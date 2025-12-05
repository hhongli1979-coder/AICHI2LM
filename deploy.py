#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TeleChat 一键本地部署脚本 (One-Click Local Deployment Script)
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import yaml
from pathlib import Path
import requests
import psutil

# 默认配置
DEFAULT_CONFIG = {
    'model_path': '../models/7B',
    'api_host': '0.0.0.0',
    'api_port': 8070,
    'web_host': '0.0.0.0',
    'web_port': 8501,
    'gpu_devices': '0',
    'check_interval': 2,
    'max_wait_time': 60
}

class TeleChatDeployer:
    """TeleChat部署管理器"""
    
    def __init__(self, config_path=None):
        self.config = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.config.update(user_config)
        
        self.api_process = None
        self.web_process = None
        self.script_dir = Path(__file__).parent.absolute()
        
    def check_dependencies(self):
        """检查依赖项"""
        print("🔍 检查依赖项...")
        
        # 检查Python版本
        if sys.version_info < (3, 7):
            print("❌ 错误: 需要Python 3.7或更高版本")
            return False
        
        # 检查必要的包
        required_packages = ['torch', 'transformers', 'fastapi', 'uvicorn', 'streamlit']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少以下Python包: {', '.join(missing_packages)}")
            print(f"📦 请运行: pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 依赖项检查通过")
        return True
    
    def check_model_path(self):
        """检查模型路径"""
        print(f"🔍 检查模型路径: {self.config['model_path']}")
        
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            print(f"❌ 错误: 模型路径不存在: {model_path}")
            print("💡 提示: 请在配置文件中设置正确的model_path")
            return False
        
        # 检查必要的模型文件
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
        missing_files = []
        
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                # 检查是否有safetensors格式
                if file_name == 'pytorch_model.bin':
                    if not any(model_path.glob('*.safetensors')):
                        missing_files.append(file_name)
                else:
                    missing_files.append(file_name)
        
        if missing_files:
            print(f"⚠️  警告: 模型目录中缺少一些文件: {', '.join(missing_files)}")
            print("📝 模型可能仍然可用，继续尝试启动...")
        
        print("✅ 模型路径检查通过")
        return True
    
    def check_port_available(self, port):
        """检查端口是否可用"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return False
        return True
    
    def wait_for_api(self):
        """等待API服务启动"""
        print(f"⏳ 等待API服务启动 (最多{self.config['max_wait_time']}秒)...")
        
        url = f"http://127.0.0.1:{self.config['api_port']}/docs"
        start_time = time.time()
        
        while time.time() - start_time < self.config['max_wait_time']:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print("✅ API服务已就绪")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(self.config['check_interval'])
        
        print("❌ 错误: API服务启动超时")
        return False
    
    def start_api_service(self):
        """启动API服务"""
        print(f"🚀 启动API服务 (端口: {self.config['api_port']})...")
        
        # 检查端口
        if not self.check_port_available(self.config['api_port']):
            print(f"❌ 错误: 端口 {self.config['api_port']} 已被占用")
            return False
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.config['gpu_devices']
        
        # 启动API服务
        service_script = self.script_dir / 'service' / 'telechat_service.py'
        if not service_script.exists():
            print(f"❌ 错误: 找不到服务脚本: {service_script}")
            return False
        
        try:
            self.api_process = subprocess.Popen(
                [sys.executable, str(service_script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.script_dir / 'service')
            )
            
            # 等待服务启动
            if not self.wait_for_api():
                self.stop_api_service()
                return False
            
            print(f"✅ API服务已启动 (PID: {self.api_process.pid})")
            print(f"📍 API文档地址: http://{self.config['api_host']}:{self.config['api_port']}/docs")
            return True
            
        except Exception as e:
            print(f"❌ 启动API服务失败: {e}")
            return False
    
    def start_web_service(self):
        """启动Web服务"""
        print(f"🚀 启动Web服务 (端口: {self.config['web_port']})...")
        
        # 检查端口
        if not self.check_port_available(self.config['web_port']):
            print(f"❌ 错误: 端口 {self.config['web_port']} 已被占用")
            return False
        
        # 启动Web服务
        web_script = self.script_dir / 'service' / 'web_demo.py'
        if not web_script.exists():
            print(f"❌ 错误: 找不到Web脚本: {web_script}")
            return False
        
        try:
            self.web_process = subprocess.Popen(
                [
                    sys.executable, '-m', 'streamlit', 'run',
                    str(web_script),
                    '--server.port', str(self.config['web_port']),
                    '--server.address', self.config['web_host']
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.script_dir / 'service')
            )
            
            # 等待服务启动
            time.sleep(5)
            
            if self.web_process.poll() is not None:
                print("❌ Web服务启动失败")
                return False
            
            print(f"✅ Web服务已启动 (PID: {self.web_process.pid})")
            print(f"📍 Web访问地址: http://{self.config['web_host']}:{self.config['web_port']}")
            return True
            
        except Exception as e:
            print(f"❌ 启动Web服务失败: {e}")
            return False
    
    def stop_api_service(self):
        """停止API服务"""
        if self.api_process:
            print("🛑 停止API服务...")
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            self.api_process = None
    
    def stop_web_service(self):
        """停止Web服务"""
        if self.web_process:
            print("🛑 停止Web服务...")
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.web_process.kill()
            self.web_process = None
    
    def stop_all_services(self):
        """停止所有服务"""
        self.stop_web_service()
        self.stop_api_service()
    
    def deploy(self):
        """执行部署"""
        print("=" * 60)
        print("🎯 TeleChat 一键本地部署")
        print("=" * 60)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 检查模型
        if not self.check_model_path():
            return False
        
        # 启动API服务
        if not self.start_api_service():
            self.stop_all_services()
            return False
        
        # 启动Web服务
        if not self.start_web_service():
            self.stop_all_services()
            return False
        
        print("\n" + "=" * 60)
        print("✨ 部署成功！")
        print("=" * 60)
        print(f"📍 API服务: http://{self.config['api_host']}:{self.config['api_port']}/docs")
        print(f"📍 Web界面: http://{self.config['web_host']}:{self.config['web_port']}")
        print("\n按 Ctrl+C 停止服务")
        print("=" * 60)
        
        return True
    
    def run(self):
        """运行部署并保持服务"""
        try:
            if not self.deploy():
                return 1
            
            # 保持运行直到用户中断
            while True:
                time.sleep(1)
                
                # 检查进程是否仍在运行
                if self.api_process and self.api_process.poll() is not None:
                    print("❌ API服务意外停止")
                    break
                
                if self.web_process and self.web_process.poll() is not None:
                    print("❌ Web服务意外停止")
                    break
        
        except KeyboardInterrupt:
            print("\n\n👋 收到停止信号...")
        
        finally:
            self.stop_all_services()
            print("✅ 所有服务已停止")
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='TeleChat 一键本地部署工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置部署
  python deploy.py
  
  # 使用自定义配置文件
  python deploy.py --config deploy_config.yaml
  
  # 指定GPU设备
  python deploy.py --gpu 0,1
  
  # 指定模型路径
  python deploy.py --model-path /path/to/model
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='配置文件路径',
        default=None
    )
    
    parser.add_argument(
        '--model-path', '-m',
        help='模型路径',
        default=None
    )
    
    parser.add_argument(
        '--gpu', '-g',
        help='GPU设备 (例如: 0 或 0,1)',
        default=None
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        help='API服务端口',
        default=None
    )
    
    parser.add_argument(
        '--web-port',
        type=int,
        help='Web服务端口',
        default=None
    )
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = TeleChatDeployer(config_path=args.config)
    
    # 应用命令行参数
    if args.model_path:
        deployer.config['model_path'] = args.model_path
    if args.gpu:
        deployer.config['gpu_devices'] = args.gpu
    if args.api_port:
        deployer.config['api_port'] = args.api_port
    if args.web_port:
        deployer.config['web_port'] = args.web_port
    
    # 运行部署
    sys.exit(deployer.run())


if __name__ == '__main__':
    main()
