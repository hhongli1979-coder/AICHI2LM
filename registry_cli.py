#!/usr/bin/env python3
"""
Docker Registry Manager CLI
Docker 镜像仓库管理命令行工具
"""

import argparse
import sys
from docker_registry_manager import DockerRegistryManager, Permission


def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="TeleChat Docker 镜像仓库管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加用户 (请使用强密码)
  %(prog)s user add admin <STRONG_PASSWORD> --permissions admin --email admin@example.com
  
  # 列出所有用户
  %(prog)s user list
  
  # 注册镜像
  %(prog)s image register telechat 7b-fp16 --users user1,user2
  
  # 授予访问权限
  %(prog)s access grant telechat 7b-fp16 user3
  
  # 检查访问权限
  %(prog)s access check user1 telechat 7b-fp16
  
  # 列出用户可访问的镜像
  %(prog)s image list-accessible user1
        """
    )
    
    parser.add_argument(
        '--registry-url',
        default='localhost:5000',
        help='Docker 仓库地址 (默认: localhost:5000)'
    )
    
    parser.add_argument(
        '--config-dir',
        default='./registry',
        help='配置文件目录 (默认: ./registry)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 用户管理命令
    user_parser = subparsers.add_parser('user', help='用户管理')
    user_subparsers = user_parser.add_subparsers(dest='user_command')
    
    # 添加用户
    user_add = user_subparsers.add_parser('add', help='添加用户')
    user_add.add_argument('username', help='用户名')
    user_add.add_argument('password', help='密码')
    user_add.add_argument('--permissions', required=True,
                         help='权限列表 (逗号分隔): view,pull,push,delete,admin')
    user_add.add_argument('--email', help='邮箱地址')
    
    # 删除用户
    user_remove = user_subparsers.add_parser('remove', help='删除用户')
    user_remove.add_argument('username', help='用户名')
    
    # 更新用户权限
    user_update = user_subparsers.add_parser('update', help='更新用户权限')
    user_update.add_argument('username', help='用户名')
    user_update.add_argument('--permissions', required=True,
                           help='权限列表 (逗号分隔): view,pull,push,delete,admin')
    
    # 列出用户
    user_subparsers.add_parser('list', help='列出所有用户')
    
    # 镜像管理命令
    image_parser = subparsers.add_parser('image', help='镜像管理')
    image_subparsers = image_parser.add_subparsers(dest='image_command')
    
    # 注册镜像
    image_register = image_subparsers.add_parser('register', help='注册镜像')
    image_register.add_argument('name', help='镜像名称')
    image_register.add_argument('tag', help='镜像标签')
    image_register.add_argument('--users', help='允许访问的用户列表 (逗号分隔，留空表示所有用户)')
    image_register.add_argument('--digest', default='', help='镜像摘要')
    image_register.add_argument('--created', default='', help='创建时间')
    image_register.add_argument('--size', type=int, default=0, help='镜像大小')
    
    # 列出用户可访问的镜像
    image_list = image_subparsers.add_parser('list-accessible', 
                                            help='列出用户可访问的镜像')
    image_list.add_argument('username', help='用户名')
    
    # 访问控制命令
    access_parser = subparsers.add_parser('access', help='访问控制')
    access_subparsers = access_parser.add_subparsers(dest='access_command')
    
    # 授予访问权限
    access_grant = access_subparsers.add_parser('grant', help='授予访问权限')
    access_grant.add_argument('image_name', help='镜像名称')
    access_grant.add_argument('tag', help='镜像标签')
    access_grant.add_argument('username', help='用户名')
    
    # 撤销访问权限
    access_revoke = access_subparsers.add_parser('revoke', help='撤销访问权限')
    access_revoke.add_argument('image_name', help='镜像名称')
    access_revoke.add_argument('tag', help='镜像标签')
    access_revoke.add_argument('username', help='用户名')
    
    # 检查访问权限
    access_check = access_subparsers.add_parser('check', help='检查访问权限')
    access_check.add_argument('username', help='用户名')
    access_check.add_argument('image_name', help='镜像名称')
    access_check.add_argument('tag', help='镜像标签')
    
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 创建管理器实例
    manager = DockerRegistryManager(
        registry_url=args.registry_url,
        config_dir=args.config_dir
    )
    
    # 用户管理命令
    if args.command == 'user':
        if not args.user_command:
            parser.parse_args(['user', '-h'])
            return 1
            
        if args.user_command == 'add':
            permissions = [p.strip() for p in args.permissions.split(',')]
            manager.add_user(args.username, args.password, permissions, args.email)
            
        elif args.user_command == 'remove':
            manager.remove_user(args.username)
            
        elif args.user_command == 'update':
            permissions = [p.strip() for p in args.permissions.split(',')]
            manager.update_user_permissions(args.username, permissions)
            
        elif args.user_command == 'list':
            users = manager.list_users()
            if not users:
                print("没有用户")
            else:
                print(f"{'用户名':<20} {'权限':<30} {'邮箱':<30}")
                print("-" * 80)
                for user in users:
                    permissions_str = ", ".join(user.permissions)
                    email_str = user.email or "N/A"
                    print(f"{user.username:<20} {permissions_str:<30} {email_str:<30}")
    
    # 镜像管理命令
    elif args.command == 'image':
        if not args.image_command:
            parser.parse_args(['image', '-h'])
            return 1
            
        if args.image_command == 'register':
            allowed_users = []
            if args.users:
                allowed_users = [u.strip() for u in args.users.split(',')]
            
            manager.register_image(
                args.name, args.tag,
                digest=args.digest,
                created=args.created,
                size=args.size,
                allowed_users=allowed_users if allowed_users else None
            )
            
        elif args.image_command == 'list-accessible':
            images = manager.list_accessible_images(args.username)
            if not images:
                print(f"用户 {args.username} 没有可访问的镜像")
            else:
                print(f"用户 {args.username} 可访问的镜像:")
                print(f"{'镜像名称':<30} {'标签':<20} {'大小':<15} {'允许用户'}")
                print("-" * 100)
                for img in images:
                    size_str = f"{img.size / (1024**3):.2f} GB" if img.size > 0 else "N/A"
                    users_str = ", ".join(img.allowed_users) if img.allowed_users else "所有用户"
                    print(f"{img.name:<30} {img.tag:<20} {size_str:<15} {users_str}")
    
    # 访问控制命令
    elif args.command == 'access':
        if not args.access_command:
            parser.parse_args(['access', '-h'])
            return 1
            
        if args.access_command == 'grant':
            manager.grant_image_access(args.image_name, args.tag, args.username)
            
        elif args.access_command == 'revoke':
            manager.revoke_image_access(args.image_name, args.tag, args.username)
            
        elif args.access_command == 'check':
            has_access = manager.check_image_access(
                args.username, args.image_name, args.tag
            )
            if has_access:
                print(f"✓ 用户 {args.username} 可以访问镜像 {args.image_name}:{args.tag}")
            else:
                print(f"✗ 用户 {args.username} 无法访问镜像 {args.image_name}:{args.tag}")
            return 0 if has_access else 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
