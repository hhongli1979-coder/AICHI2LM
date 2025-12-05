# TeleChat 监控配置示例

本目录包含可选的监控和可观测性配置。

## Prometheus 监控

### 配置文件

创建 `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'telechat'
    static_configs:
      - targets: ['telechat:8070']
    metrics_path: '/metrics'
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### 添加到 Docker Compose

在 `docker-compose.yml` 中添加：

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: telechat-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - telechat-network

  grafana:
    image: grafana/grafana:latest
    container_name: telechat-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus
    networks:
      - telechat-network

volumes:
  prometheus-data:
  grafana-data:
```

## ELK Stack (日志聚合)

### Elasticsearch + Kibana

```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    container_name: telechat-elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - telechat-network

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    container_name: telechat-kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - telechat-network

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.10.0
    container_name: telechat-filebeat
    user: root
    volumes:
      - ./monitoring/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - ./logs:/var/log/telechat:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: filebeat -e -strict.perms=false
    depends_on:
      - elasticsearch
    networks:
      - telechat-network

volumes:
  elasticsearch-data:
```

### Filebeat 配置

创建 `filebeat.yml`:

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/telechat/*.log
  fields:
    service: telechat
    
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  
setup.kibana:
  host: "kibana:5601"
```

## 使用说明

1. 创建监控目录：
```bash
mkdir -p monitoring/grafana/{dashboards,datasources}
```

2. 添加 Grafana 数据源配置：
```bash
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
```

3. 启动所有服务：
```bash
docker-compose up -d
```

4. 访问监控界面：
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Kibana: http://localhost:5601

## 告警规则

在 Prometheus 中配置告警规则，创建 `alerts.yml`:

```yaml
groups:
  - name: telechat_alerts
    interval: 30s
    rules:
      - alert: TeleChatServiceDown
        expr: up{job="telechat"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TeleChat service is down"
          description: "TeleChat service has been down for more than 1 minute"
      
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{name="telechat-service"} > 30000000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 30GB"
```

## 自定义指标

如需添加应用程序指标，可以在 TeleChat 服务中集成 Prometheus 客户端：

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 定义指标
request_count = Counter('telechat_requests_total', 'Total requests')
request_duration = Histogram('telechat_request_duration_seconds', 'Request duration')
active_requests = Gauge('telechat_active_requests', 'Active requests')

# 在 FastAPI 中添加 /metrics 端点
@app.get('/metrics')
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```
