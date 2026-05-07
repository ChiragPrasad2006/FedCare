# FedCare Setup and Installation Guide

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB free space
- OS: Windows 10/11, macOS 10.15+, or Linux

### Recommended Requirements
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50 GB free space
- GPU: NVIDIA GPU with CUDA support (for faster training)

## Step-by-Step Installation

### 1. Install Python

Download Python 3.10 or higher from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
```

### 2. Clone Repository

```bash
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare
```

### 3. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.13.0
- TensorFlow Federated 0.40.0
- PyTorch 2.0.0
- Flask 2.3.2
- And all other required dependencies

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` file with your preferred settings:
```env
NUM_ROUNDS=10
NUM_HOSPITALS=3
EPOCHS_PER_ROUND=5
BATCH_SIZE=32
LEARNING_RATE=0.001
```

### 6. Verify Installation

Test main server:
```bash
python main_server/app.py
```

Test hospital server (in new terminal):
```bash
HOSPITAL_PORT=5001 python hospital_server/app.py
```

Test communication:
```bash
python -c "import requests; print(requests.get('http://localhost:5000/health').json())"
```

## Docker Installation

### Prerequisites
- Docker Desktop ([download](https://www.docker.com/products/docker-desktop))
- Docker Compose (included in Desktop)

### Docker Setup

1. Install Docker Desktop

2. Build Docker images:
```bash
cd docker
docker-compose build
```

3. Start containers:
```bash
docker-compose up
```

4. Check container status:
```bash
docker-compose ps
```

5. View logs:
```bash
docker-compose logs -f main-server
docker-compose logs -f hospital-server-1
```

6. Stop containers:
```bash
docker-compose down
```

## Kubernetes Installation

### Prerequisites
- Docker images built and accessible
- Kubernetes cluster (minikube, Docker Desktop K8s, or cloud K8s)
- `kubectl` CLI tool

### Kubernetes Setup

1. Start Kubernetes cluster:
```bash
# Using Docker Desktop - enable in preferences
# Or using minikube
minikube start
```

2. Build and load Docker images:
```bash
docker build -t fedcare-main-server:latest -f docker/Dockerfile.main_server .
docker build -t fedcare-hospital-server:latest -f docker/Dockerfile.hospital_server .

# For minikube, load images into cluster
minikube image load fedcare-main-server:latest
minikube image load fedcare-hospital-server:latest
```

3. Deploy to Kubernetes:
```bash
kubectl apply -f kubernetes/deployment.yml
```

4. Check deployment status:
```bash
kubectl get pods -n fedcare
kubectl get svc -n fedcare
```

5. View logs:
```bash
kubectl logs -n fedcare deployment/main-server
kubectl logs -n fedcare deployment/hospital-server
```

6. Port forward for local access:
```bash
kubectl port-forward -n fedcare svc/main-server 5000:5000
kubectl port-forward -n fedcare svc/hospital-server 5001:5001
```

## Troubleshooting Installation

### Python Module Not Found

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Memory Issues

Reduce batch size or number of hospitals:
```env
BATCH_SIZE=16
NUM_HOSPITALS=2
```

### Docker Issues

Clear Docker cache:
```bash
docker system prune -a
docker volume prune
docker-compose build --no-cache
```

### Kubernetes Issues

Check cluster resources:
```bash
kubectl describe node
kubectl top pods -n fedcare
```

Check deployment events:
```bash
kubectl describe deployment main-server -n fedcare
```

## Next Steps

1. Run the quick start example: `python orchestrator.py`
2. Monitor logs in `./logs/` directory
3. Check metrics via `curl http://localhost:5000/metrics`
4. Modify configuration in `.env` for your use case
5. Explore the API endpoints documentation

## Performance Optimization

### For Faster Training
- Increase `EPOCHS_PER_ROUND`
- Increase `BATCH_SIZE`
- Use GPU acceleration (if available)

### For Scalability
- Increase `NUM_HOSPITALS` gradually
- Monitor memory usage
- Use Kubernetes for auto-scaling

### For Privacy
- Enable `ENCRYPTION_ENABLED=True` (future)
- Enable `DIFFERENTIAL_PRIVACY=True` (future)

## Useful Commands

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install specific package
pip install package-name

# Freeze dependencies
pip freeze > requirements.txt

# Run tests
python -m pytest tests/

# Format code
black .

# Lint code
flake8 .

# Start fresh
deactivate
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

For more help, check the main README.md file.
