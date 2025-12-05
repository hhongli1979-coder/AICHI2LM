#!/usr/bin/env python3
"""
Test script for Docker Registry Manager
测试 Docker 镜像仓库管理器的基本功能
"""

import os
import sys
import json
import tempfile
import shutil
from docker_registry_manager import DockerRegistryManager, Permission, User, ImageMetadata


def test_basic_functionality():
    """测试基本功能 (不需要 htpasswd)"""
    print("=== 测试 Docker Registry Manager 基本功能 ===\n")
    
    # 创建临时目录
    test_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {test_dir}")
    
    try:
        # 创建管理器实例
        manager = DockerRegistryManager(
            registry_url="localhost:5000",
            config_dir=test_dir
        )
        print("✓ 管理器实例创建成功\n")
        
        # 测试用户数据结构
        print("测试 1: 用户数据结构")
        test_user = User(
            username="testuser",
            password_hash="test_hash",
            permissions=[Permission.VIEW.value, Permission.PULL.value],
            email="test@example.com"
        )
        print(f"  用户名: {test_user.username}")
        print(f"  权限: {test_user.permissions}")
        print(f"  拥有 VIEW 权限: {test_user.has_permission(Permission.VIEW)}")
        print(f"  拥有 PUSH 权限: {test_user.has_permission(Permission.PUSH)}")
        print(f"  拥有 ADMIN 权限: {test_user.has_permission(Permission.ADMIN)}")
        print("✓ 用户数据结构测试通过\n")
        
        # 手动添加测试用户 (跳过 htpasswd)
        print("测试 2: 手动添加用户")
        manager.users["admin"] = User(
            username="admin",
            password_hash="htpasswd",
            permissions=[Permission.ADMIN.value],
            email="admin@test.com"
        )
        manager.users["user1"] = User(
            username="user1",
            password_hash="htpasswd",
            permissions=[Permission.VIEW.value, Permission.PULL.value],
            email="user1@test.com"
        )
        manager.users["user2"] = User(
            username="user2",
            password_hash="htpasswd",
            permissions=[Permission.VIEW.value, Permission.PULL.value],
            email="user2@test.com"
        )
        manager._save_users()
        print(f"  添加了 {len(manager.users)} 个用户")
        print("✓ 用户添加测试通过\n")
        
        # 测试用户列表
        print("测试 3: 列出用户")
        users = manager.list_users()
        for user in users:
            print(f"  - {user.username}: {', '.join(user.permissions)}")
        print("✓ 用户列表测试通过\n")
        
        # 测试镜像注册
        print("测试 4: 注册镜像")
        manager.register_image(
            "telechat", "7b-fp16",
            digest="sha256:abc123",
            created="2024-12-05T12:00:00Z",
            size=14000000000,
            allowed_users=["user1"]
        )
        manager.register_image(
            "telechat", "12b-fp16",
            digest="sha256:def456",
            created="2024-12-05T13:00:00Z",
            size=24000000000,
            allowed_users=["user1", "user2"]
        )
        manager.register_image(
            "telechat", "7b-int4",
            digest="sha256:ghi789",
            created="2024-12-05T14:00:00Z",
            size=3500000000,
            allowed_users=[]  # 所有用户可访问
        )
        print(f"  注册了 {len(manager.images)} 个镜像")
        print("✓ 镜像注册测试通过\n")
        
        # 测试访问权限检查
        print("测试 5: 检查访问权限")
        tests = [
            ("admin", "telechat", "7b-fp16", True),  # 管理员可以访问所有镜像
            ("user1", "telechat", "7b-fp16", True),  # user1 在允许列表中
            ("user2", "telechat", "7b-fp16", False), # user2 不在允许列表中
            ("user1", "telechat", "12b-fp16", True), # user1 在允许列表中
            ("user2", "telechat", "12b-fp16", True), # user2 在允许列表中
            ("user1", "telechat", "7b-int4", True),  # 所有用户可访问
            ("user2", "telechat", "7b-int4", True),  # 所有用户可访问
        ]
        
        all_passed = True
        for username, img_name, tag, expected in tests:
            result = manager.check_image_access(username, img_name, tag)
            status = "✓" if result == expected else "✗"
            all_passed = all_passed and (result == expected)
            print(f"  {status} {username} 访问 {img_name}:{tag} = {result} (期望: {expected})")
        
        if all_passed:
            print("✓ 访问权限检查测试通过\n")
        else:
            print("✗ 访问权限检查测试失败\n")
            return False
        
        # 测试列出可访问镜像
        print("测试 6: 列出用户可访问的镜像")
        for username in ["admin", "user1", "user2"]:
            images = manager.list_accessible_images(username)
            print(f"  {username} 可访问 {len(images)} 个镜像:")
            for img in images:
                print(f"    - {img.name}:{img.tag}")
        print("✓ 列出可访问镜像测试通过\n")
        
        # 测试授予访问权限
        print("测试 7: 授予访问权限")
        manager.grant_image_access("telechat", "7b-fp16", "user2")
        has_access = manager.check_image_access("user2", "telechat", "7b-fp16")
        if has_access:
            print("  ✓ user2 现在可以访问 telechat:7b-fp16")
            print("✓ 授予访问权限测试通过\n")
        else:
            print("  ✗ 授予权限失败")
            return False
        
        # 测试撤销访问权限
        print("测试 8: 撤销访问权限")
        manager.revoke_image_access("telechat", "7b-fp16", "user2")
        has_access = manager.check_image_access("user2", "telechat", "7b-fp16")
        if not has_access:
            print("  ✓ user2 不再能访问 telechat:7b-fp16")
            print("✓ 撤销访问权限测试通过\n")
        else:
            print("  ✗ 撤销权限失败")
            return False
        
        # 测试配置文件持久化
        print("测试 9: 配置文件持久化")
        users_file = os.path.join(test_dir, "users.json")
        images_file = os.path.join(test_dir, "images.json")
        
        if os.path.exists(users_file) and os.path.exists(images_file):
            with open(users_file, 'r') as f:
                users_data = json.load(f)
            with open(images_file, 'r') as f:
                images_data = json.load(f)
            
            print(f"  users.json: {len(users_data)} 个用户")
            print(f"  images.json: {len(images_data)} 个镜像")
            print("✓ 配置文件持久化测试通过\n")
        else:
            print("  ✗ 配置文件未创建")
            return False
        
        # 测试重新加载配置
        print("测试 10: 重新加载配置")
        new_manager = DockerRegistryManager(
            registry_url="localhost:5000",
            config_dir=test_dir
        )
        if len(new_manager.users) == len(manager.users) and \
           len(new_manager.images) == len(manager.images):
            print(f"  重新加载了 {len(new_manager.users)} 个用户和 {len(new_manager.images)} 个镜像")
            print("✓ 重新加载配置测试通过\n")
        else:
            print("  ✗ 配置加载不一致")
            return False
        
        print("=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50)
        return True
        
    finally:
        # 清理临时目录
        shutil.rmtree(test_dir)
        print(f"\n清理临时目录: {test_dir}")


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
