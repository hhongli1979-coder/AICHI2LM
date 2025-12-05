# SSL 证书目录

将您的 SSL 证书文件放在此目录中。

## 文件命名建议

- `cert.pem` - SSL 证书文件
- `key.pem` - 私钥文件
- `chain.pem` - 证书链文件（可选）

## 生成自签名证书（仅用于测试）

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/C=CN/ST=Beijing/L=Beijing/O=TeleChat/CN=localhost"
```

## 使用 Let's Encrypt（推荐用于生产）

```bash
# 安装 certbot
sudo apt-get install certbot

# 获取证书
sudo certbot certonly --standalone -d your-domain.com

# 复制证书到此目录
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem key.pem
```

## 安全提醒

⚠️ **重要**：请勿将证书文件提交到版本控制系统！

此目录已添加到 `.gitignore` 中，以防止意外提交敏感文件。
