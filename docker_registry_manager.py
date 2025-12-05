#!/usr/bin/env python3
"""
Docker Registry Manager with Access Control
管理 Docker 镜像仓库并控制访问权限
"""

import os
import json
import subprocess
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict


class Permission(Enum):
    """访问权限类型"""
    VIEW = "view"      # 查看镜像列表
    PULL = "pull"      # 拉取镜像
    PUSH = "push"      # 推送镜像
    DELETE = "delete"  # 删除镜像
    ADMIN = "admin"    # 管理员权限


@dataclass
class User:
    """用户信息"""
    username: str
    password_hash: str
    permissions: List[str]
    email: Optional[str] = None

    def has_permission(self, permission: Permission) -> bool:
        """检查用户是否拥有特定权限"""
        return Permission.ADMIN.value in self.permissions or permission.value in self.permissions


@dataclass
class ImageMetadata:
    """镜像元数据"""
    name: str
    tag: str
    digest: str
    created: str
    size: int
    allowed_users: List[str]  # 允许访问的用户列表


class DockerRegistryManager:
    """Docker 镜像仓库管理器"""
    
    def __init__(self, registry_url: str = "localhost:5000", 
                 config_dir: str = "./registry"):
        self.registry_url = registry_url
        self.config_dir = config_dir
        self.users_file = os.path.join(config_dir, "users.json")
        self.images_file = os.path.join(config_dir, "images.json")
        self.htpasswd_file = os.path.join(config_dir, "auth", "htpasswd")
        
        # 创建必要的目录
        os.makedirs(os.path.join(config_dir, "auth"), exist_ok=True)
        os.makedirs(os.path.join(config_dir, "data"), exist_ok=True)
        
        # 加载用户和镜像数据
        self.users: Dict[str, User] = self._load_users()
        self.images: Dict[str, ImageMetadata] = self._load_images()

    def _load_users(self) -> Dict[str, User]:
        """加载用户数据"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {username: User(**user_data) 
                       for username, user_data in data.items()}
        return {}

    def _save_users(self):
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            data = {username: asdict(user) 
                   for username, user in self.users.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_images(self) -> Dict[str, ImageMetadata]:
        """加载镜像元数据"""
        if os.path.exists(self.images_file):
            with open(self.images_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {key: ImageMetadata(**img_data) 
                       for key, img_data in data.items()}
        return {}

    def _save_images(self):
        """保存镜像元数据"""
        with open(self.images_file, 'w', encoding='utf-8') as f:
            data = {key: asdict(img) 
                   for key, img in self.images.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_user(self, username: str, password: str, 
                 permissions: List[str], email: Optional[str] = None) -> bool:
        """
        添加用户
        
        Args:
            username: 用户名
            password: 密码
            permissions: 权限列表
            email: 邮箱（可选）
            
        Returns:
            是否添加成功
        """
        if username in self.users:
            print(f"用户 {username} 已存在")
            return False
        
        # 使用 htpasswd 生成密码哈希
        try:
            # 创建或更新 htpasswd 文件
            if os.path.exists(self.htpasswd_file):
                cmd = ["htpasswd", "-B", "-b", self.htpasswd_file, username, password]
            else:
                cmd = ["htpasswd", "-B", "-c", "-b", self.htpasswd_file, username, password]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 添加用户到内存和配置文件
            user = User(username=username, 
                       password_hash="htpasswd",  # 实际哈希在 htpasswd 文件中
                       permissions=permissions,
                       email=email)
            self.users[username] = user
            self._save_users()
            
            print(f"成功添加用户: {username}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"添加用户失败: {e}")
            return False
        except FileNotFoundError:
            print("错误: 未找到 htpasswd 命令。请安装 apache2-utils (Debian/Ubuntu) 或 httpd-tools (RHEL/CentOS)")
            return False

    def remove_user(self, username: str) -> bool:
        """
        删除用户
        
        Args:
            username: 用户名
            
        Returns:
            是否删除成功
        """
        if username not in self.users:
            print(f"用户 {username} 不存在")
            return False
        
        try:
            # 从 htpasswd 文件删除
            subprocess.run(["htpasswd", "-D", self.htpasswd_file, username],
                         check=True, capture_output=True)
            
            # 从内存和配置文件删除
            del self.users[username]
            self._save_users()
            
            print(f"成功删除用户: {username}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"删除用户失败: {e}")
            return False

    def update_user_permissions(self, username: str, 
                               permissions: List[str]) -> bool:
        """
        更新用户权限
        
        Args:
            username: 用户名
            permissions: 新的权限列表
            
        Returns:
            是否更新成功
        """
        if username not in self.users:
            print(f"用户 {username} 不存在")
            return False
        
        self.users[username].permissions = permissions
        self._save_users()
        print(f"成功更新用户 {username} 的权限")
        return True

    def list_users(self) -> List[User]:
        """列出所有用户"""
        return list(self.users.values())

    def register_image(self, name: str, tag: str, digest: str = "",
                      created: str = "", size: int = 0,
                      allowed_users: Optional[List[str]] = None) -> bool:
        """
        注册镜像元数据
        
        Args:
            name: 镜像名称
            tag: 镜像标签
            digest: 镜像摘要
            created: 创建时间
            size: 镜像大小
            allowed_users: 允许访问的用户列表（None 表示所有用户）
            
        Returns:
            是否注册成功
        """
        key = f"{name}:{tag}"
        
        if allowed_users is None:
            allowed_users = []  # 空列表表示所有用户都可以访问
        
        image = ImageMetadata(
            name=name,
            tag=tag,
            digest=digest,
            created=created,
            size=size,
            allowed_users=allowed_users
        )
        
        self.images[key] = image
        self._save_images()
        print(f"成功注册镜像: {key}")
        return True

    def check_image_access(self, username: str, image_name: str, 
                          tag: str) -> bool:
        """
        检查用户是否有权访问指定镜像
        
        Args:
            username: 用户名
            image_name: 镜像名称
            tag: 镜像标签
            
        Returns:
            是否有访问权限
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # 管理员可以访问所有镜像
        if user.has_permission(Permission.ADMIN):
            return True
        
        key = f"{image_name}:{tag}"
        
        # 如果镜像未注册，默认拒绝访问
        if key not in self.images:
            return False
        
        image = self.images[key]
        
        # 空列表表示所有用户都可以访问
        if not image.allowed_users:
            return True
        
        # 检查用户是否在允许列表中
        return username in image.allowed_users

    def list_accessible_images(self, username: str) -> List[ImageMetadata]:
        """
        列出用户可以访问的所有镜像
        
        Args:
            username: 用户名
            
        Returns:
            可访问的镜像列表
        """
        if username not in self.users:
            return []
        
        user = self.users[username]
        
        # 管理员可以查看所有镜像
        if user.has_permission(Permission.ADMIN):
            return list(self.images.values())
        
        # 普通用户只能查看有权限的镜像
        accessible_images = []
        for image in self.images.values():
            if not image.allowed_users or username in image.allowed_users:
                accessible_images.append(image)
        
        return accessible_images

    def grant_image_access(self, image_name: str, tag: str, 
                          username: str) -> bool:
        """
        授予用户访问特定镜像的权限
        
        Args:
            image_name: 镜像名称
            tag: 镜像标签
            username: 用户名
            
        Returns:
            是否授权成功
        """
        key = f"{image_name}:{tag}"
        
        if key not in self.images:
            print(f"镜像 {key} 不存在")
            return False
        
        if username not in self.users:
            print(f"用户 {username} 不存在")
            return False
        
        image = self.images[key]
        if username not in image.allowed_users:
            image.allowed_users.append(username)
            self._save_images()
            print(f"成功授予用户 {username} 访问镜像 {key} 的权限")
        else:
            print(f"用户 {username} 已经拥有访问镜像 {key} 的权限")
        
        return True

    def revoke_image_access(self, image_name: str, tag: str, 
                           username: str) -> bool:
        """
        撤销用户访问特定镜像的权限
        
        Args:
            image_name: 镜像名称
            tag: 镜像标签
            username: 用户名
            
        Returns:
            是否撤销成功
        """
        key = f"{image_name}:{tag}"
        
        if key not in self.images:
            print(f"镜像 {key} 不存在")
            return False
        
        image = self.images[key]
        if username in image.allowed_users:
            image.allowed_users.remove(username)
            self._save_images()
            print(f"成功撤销用户 {username} 访问镜像 {key} 的权限")
        else:
            print(f"用户 {username} 没有访问镜像 {key} 的权限")
        
        return True


def main():
    """示例用法"""
    # 创建管理器实例
    manager = DockerRegistryManager()
    
    # 添加管理员用户 (请使用强密码)
    manager.add_user("admin", "YOUR_STRONG_PASSWORD_HERE", [Permission.ADMIN.value], "admin@example.com")
    
    # 添加普通用户 (请使用强密码)
    manager.add_user("user1", "YOUR_PASSWORD_HERE", [Permission.VIEW.value, Permission.PULL.value], "user1@example.com")
    manager.add_user("user2", "YOUR_PASSWORD_HERE", [Permission.VIEW.value, Permission.PULL.value], "user2@example.com")
    
    # 注册镜像
    manager.register_image("telechat", "7b-fp16", allowed_users=["user1"])
    manager.register_image("telechat", "12b-fp16", allowed_users=["user1", "user2"])
    manager.register_image("telechat", "7b-int4", allowed_users=[])  # 所有用户可访问
    
    # 检查访问权限
    print(f"\nuser1 可以访问 telechat:7b-fp16? {manager.check_image_access('user1', 'telechat', '7b-fp16')}")
    print(f"user2 可以访问 telechat:7b-fp16? {manager.check_image_access('user2', 'telechat', '7b-fp16')}")
    
    # 列出用户可访问的镜像
    print(f"\nuser1 可访问的镜像:")
    for img in manager.list_accessible_images("user1"):
        print(f"  - {img.name}:{img.tag}")
    
    print(f"\nuser2 可访问的镜像:")
    for img in manager.list_accessible_images("user2"):
        print(f"  - {img.name}:{img.tag}")
    
    # 授予访问权限
    manager.grant_image_access("telechat", "7b-fp16", "user2")
    
    # 再次检查
    print(f"\n授权后 user2 可以访问 telechat:7b-fp16? {manager.check_image_access('user2', 'telechat', '7b-fp16')}")


if __name__ == "__main__":
    main()
