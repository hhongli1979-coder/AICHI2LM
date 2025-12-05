# TeleChat Kubernetes 部署指南

## 概述

本指南介绍如何在 Kubernetes 集群中部署 TeleChat 服务。

## 前提条件

- Kubernetes 集群（版本 1.20+）
- kubectl 命令行工具已配置
- 集群支持 GPU（NVIDIA Device Plugin）
- 持久化存储（用于模型和日志）
- Ingress Controller（可选，用于外部访问）

## 部署步骤

### 1. 准备 GPU 支持

确保 Kubernetes 集群已安装 NVIDIA Device Plugin：

```bash
# 部署 NVIDIA Device Plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# 验证 GPU 节点
kubectl get nodes -o json | jq '.items[].status.allocatable'
```

### 2. 准备模型文件

将模型文件上传到持久化存储：

```bash
# 方式一：使用 kubectl cp
kubectl create namespace telechat
kubectl run model-upload --image=busybox --restart=Never -n telechat -- sleep 3600
kubectl cp models/7B telechat/model-upload:/tmp/models/7B
kubectl delete pod model-upload -n telechat

# 方式二：使用 NFS 或对象存储
# 将模型文件上传到 NFS/S3 等存储系统
```

### 3. 创建 PersistentVolume

如果使用本地存储，需要先创建 PV：

```yaml
# models-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: telechat-models-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadOnlyMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: /data/telechat/models
---
# logs-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: telechat-logs-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: /data/telechat/logs
```

应用配置：

```bash
kubectl apply -f models-pv.yaml
kubectl apply -f logs-pv.yaml
```

### 4. 部署 TeleChat 服务

```bash
# 应用完整部署配置
kubectl apply -f kubernetes/telechat-deployment.yaml

# 查看部署状态
kubectl get all -n telechat

# 查看 Pod 日志
kubectl logs -f -n telechat -l app=telechat
```

### 5. 配置 Ingress（可选）

如果需要通过域名访问：

```bash
# 修改 Ingress 配置中的域名
vim kubernetes/telechat-deployment.yaml

# 应用配置
kubectl apply -f kubernetes/telechat-deployment.yaml

# 获取 Ingress 地址
kubectl get ingress -n telechat
```

### 6. 配置 TLS 证书（可选）

```bash
# 创建 TLS Secret
kubectl create secret tls telechat-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n telechat

# 或使用 cert-manager 自动获取证书
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

## 扩缩容

### 手动扩缩容

```bash
# 扩展到 5 个副本
kubectl scale deployment telechat-deployment -n telechat --replicas=5

# 查看副本状态
kubectl get pods -n telechat
```

### 自动扩缩容（HPA）

HPA 配置已包含在部署文件中，基于 CPU 和内存使用率自动扩缩容：

```bash
# 查看 HPA 状态
kubectl get hpa -n telechat

# 查看详细信息
kubectl describe hpa telechat-hpa -n telechat
```

## 监控和日志

### 查看 Pod 状态

```bash
# 列出所有 Pod
kubectl get pods -n telechat

# 查看 Pod 详情
kubectl describe pod <pod-name> -n telechat

# 查看 Pod 日志
kubectl logs -f <pod-name> -n telechat
```

### 健康检查

```bash
# 在集群内部检查
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n telechat -- \
  curl http://telechat-service:8070/health

# 通过 LoadBalancer 检查
curl http://<external-ip>:8070/health
```

### 资源使用监控

```bash
# 查看节点资源使用
kubectl top nodes

# 查看 Pod 资源使用
kubectl top pods -n telechat

# 查看容器资源使用
kubectl top pods -n telechat --containers
```

## 更新部署

### 滚动更新

```bash
# 更新镜像
kubectl set image deployment/telechat-deployment telechat=telechat:v2 -n telechat

# 查看滚动更新状态
kubectl rollout status deployment/telechat-deployment -n telechat

# 查看更新历史
kubectl rollout history deployment/telechat-deployment -n telechat
```

### 回滚部署

```bash
# 回滚到上一个版本
kubectl rollout undo deployment/telechat-deployment -n telechat

# 回滚到指定版本
kubectl rollout undo deployment/telechat-deployment -n telechat --to-revision=2
```

## 故障排查

### Pod 无法启动

```bash
# 查看 Pod 事件
kubectl describe pod <pod-name> -n telechat

# 查看日志
kubectl logs <pod-name> -n telechat

# 常见问题：
# - GPU 资源不足
# - 模型文件未挂载
# - 内存不足
# - 镜像拉取失败
```

### 健康检查失败

```bash
# 进入 Pod 内部调试
kubectl exec -it <pod-name> -n telechat -- /bin/bash

# 手动测试健康检查
curl http://localhost:8070/health

# 查看应用日志
tail -f /app/logs/*.log
```

### 网络问题

```bash
# 测试 Service 连接
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n telechat -- \
  curl http://telechat-service:8070/health

# 检查 Service 端点
kubectl get endpoints telechat-service -n telechat

# 检查 Ingress
kubectl describe ingress telechat-ingress -n telechat
```

## 备份和恢复

### 备份配置

```bash
# 导出所有配置
kubectl get all,configmap,secret,pvc,pv,ingress -n telechat -o yaml > telechat-backup.yaml

# 备份模型文件
kubectl cp telechat/<pod-name>:/app/models ./models-backup
```

### 恢复配置

```bash
# 恢复配置
kubectl apply -f telechat-backup.yaml

# 恢复模型文件
kubectl cp ./models-backup telechat/<pod-name>:/app/models
```

## 清理资源

```bash
# 删除所有资源
kubectl delete -f kubernetes/telechat-deployment.yaml

# 删除命名空间（会删除所有资源）
kubectl delete namespace telechat
```

## 最佳实践

1. **资源限制**：始终设置合理的 CPU 和内存限制
2. **健康检查**：配置 liveness 和 readiness 探针
3. **持久化存储**：使用可靠的存储后端（NFS、Ceph、云存储）
4. **日志收集**：集成 Fluentd 或 Filebeat 收集日志
5. **监控告警**：使用 Prometheus 和 Grafana 监控
6. **安全加固**：使用 NetworkPolicy、PodSecurityPolicy
7. **备份策略**：定期备份配置和数据
8. **测试环境**：在生产前在测试环境充分测试

## 参考资源

- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)
- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Ingress Controllers](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/)
