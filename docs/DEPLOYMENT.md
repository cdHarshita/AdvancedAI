# Deployment Guide - AI Systems Lab

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended for Development)

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 10GB+ disk space

#### Steps

1. **Clone and configure**
```bash
git clone https://github.com/cdHarshita/AdvancedAI.git
cd AdvancedAI
cp .env.example .env
# Edit .env with your API keys
```

2. **Start services**
```bash
docker-compose up -d
```

3. **Verify deployment**
```bash
# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api

# Access services
# API: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9091
```

4. **Stop services**
```bash
docker-compose down
# or to remove volumes
docker-compose down -v
```

### Option 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+

#### Create Kubernetes manifests

**1. Namespace**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-systems
```

**2. ConfigMap**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-systems-config
  namespace: ai-systems
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

**3. Secret**
```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-systems-secret
  namespace: ai-systems
type: Opaque
stringData:
  OPENAI_API_KEY: "your-api-key-here"
  SECRET_KEY: "your-secret-key-here"
```

**4. Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-systems-api
  namespace: ai-systems
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-systems-api
  template:
    metadata:
      labels:
        app: ai-systems-api
    spec:
      containers:
      - name: api
        image: ai-systems:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: ai-systems-config
        - secretRef:
            name: ai-systems-secret
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**5. Service**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-systems-api
  namespace: ai-systems
spec:
  selector:
    app: ai-systems-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

**6. HorizontalPodAutoscaler**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-systems-api-hpa
  namespace: ai-systems
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-systems-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Deploy to Kubernetes
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (edit first!)
kubectl apply -f k8s/secret.yaml

# Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment
kubectl get pods -n ai-systems
kubectl get svc -n ai-systems

# View logs
kubectl logs -f deployment/ai-systems-api -n ai-systems

# Get service URL
kubectl get svc ai-systems-api -n ai-systems
```

### Option 3: AWS ECS Deployment

#### Prerequisites
- AWS CLI configured
- ECS cluster created
- ECR repository created

#### Steps

1. **Build and push Docker image**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t ai-systems:latest .

# Tag for ECR
docker tag ai-systems:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-systems:latest

# Push to ECR
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-systems:latest
```

2. **Create task definition**
```json
{
  "family": "ai-systems-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "ai-systems-api",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-systems:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-systems",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

3. **Create service**
```bash
aws ecs create-service \
  --cluster ai-systems-cluster \
  --service-name ai-systems-service \
  --task-definition ai-systems-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=ai-systems-api,containerPort=8000"
```

### Option 4: Google Cloud Run

#### Prerequisites
- gcloud CLI installed
- GCP project created
- Artifact Registry repository created

#### Steps

1. **Build and push image**
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build image
docker build -t gcr.io/PROJECT_ID/ai-systems:latest .

# Push to GCR
docker push gcr.io/PROJECT_ID/ai-systems:latest
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy ai-systems \
  --image gcr.io/PROJECT_ID/ai-systems:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production,LOG_LEVEL=INFO \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --concurrency 80
```

### Option 5: Azure Container Instances

#### Prerequisites
- Azure CLI installed
- Azure subscription
- Azure Container Registry created

#### Steps

1. **Build and push image**
```bash
# Login to ACR
az acr login --name myregistry

# Build and push
az acr build --registry myregistry --image ai-systems:latest .
```

2. **Deploy to ACI**
```bash
az container create \
  --resource-group ai-systems-rg \
  --name ai-systems-api \
  --image myregistry.azurecr.io/ai-systems:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server myregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label ai-systems-api \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production LOG_LEVEL=INFO \
  --secure-environment-variables OPENAI_API_KEY=<key>
```

## 🔧 Environment Configuration

### Development
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
API_WORKERS=1
ENABLE_METRICS=true
```

### Staging
```bash
ENVIRONMENT=staging
LOG_LEVEL=INFO
API_WORKERS=2
ENABLE_METRICS=true
```

### Production
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
API_WORKERS=4
ENABLE_METRICS=true
MAX_RETRIES=3
RATE_LIMIT_PER_MINUTE=100
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-systems-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ai-systems
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: ai-systems-api
        action: keep
```

### Grafana Dashboards
Import pre-built dashboards:
- **Dashboard ID 1860**: Node Exporter
- **Dashboard ID 3662**: Prometheus 2.0 Stats
- **Custom Dashboard**: API metrics (included in repo)

## 🔒 Security Checklist

Before deploying to production:

- [ ] Change default SECRET_KEY
- [ ] Set strong passwords for all services
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Configure log rotation
- [ ] Set up backup strategy
- [ ] Enable audit logging
- [ ] Review security groups
- [ ] Scan images for vulnerabilities
- [ ] Set resource limits
- [ ] Enable container security scanning
- [ ] Configure secrets management
- [ ] Set up alerting

## 🚨 Troubleshooting

### Common Issues

**1. Container fails to start**
```bash
# Check logs
docker-compose logs api

# Common fixes
- Verify .env file exists
- Check API keys are set
- Ensure ports are not in use
```

**2. High memory usage**
```bash
# Reduce workers
API_WORKERS=2

# Limit vector store size
# Use chunking for large documents
```

**3. Slow response times**
```bash
# Enable caching
# Add Redis for rate limiting
# Use connection pooling
# Optimize RAG chunk size
```

**4. Cost overruns**
```bash
# Set strict limits
MAX_COST_PER_REQUEST=0.10
# Use cheaper models
DEFAULT_MODEL=gpt-3.5-turbo
```

## 📈 Scaling Guide

### Horizontal Scaling
```bash
# Docker Compose
docker-compose up -d --scale api=3

# Kubernetes
kubectl scale deployment ai-systems-api --replicas=5 -n ai-systems

# AWS ECS
aws ecs update-service --cluster ai-systems-cluster --service ai-systems-service --desired-count 5
```

### Vertical Scaling
```yaml
# Increase resources
resources:
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t ai-systems:${{ github.sha }} .
      
      - name: Run tests
        run: |
          docker run ai-systems:${{ github.sha }} pytest
      
      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
          docker push ai-systems:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ai-systems-api ai-systems-api=ai-systems:${{ github.sha }} -n ai-systems
```

## 📝 Post-Deployment Validation

```bash
# 1. Health check
curl https://api.example.com/health

# 2. Metrics
curl https://api.example.com/metrics

# 3. Test endpoint
curl -X POST https://api.example.com/api/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'

# 4. Check logs
kubectl logs -f deployment/ai-systems-api -n ai-systems

# 5. Monitor costs
curl https://api.example.com/api/v1/cost/summary
```

## 🎯 Performance Tuning

### Database Optimization
```python
# Connection pooling
DATABASE_URL=postgresql://user:pass@host/db?pool_size=20&max_overflow=10
```

### Caching Strategy
```python
# Redis caching for repeated queries
CACHE_TTL=3600
CACHE_ENABLED=true
```

### Load Balancing
```nginx
# Nginx configuration
upstream ai_systems {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Production Patterns](https://kubernetes.io/docs/concepts/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
