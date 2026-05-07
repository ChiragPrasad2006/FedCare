# FedCare Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### 1. Install & Setup
```bash
# Clone
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
```

### 2. Start Servers

**Option A: Local (Development)**
```bash
# Terminal 1
python main_server/app.py

# Terminal 2
HOSPITAL_PORT=5001 python hospital_server/app.py

# Terminal 3
HOSPITAL_PORT=5002 python hospital_server/app.py

# Terminal 4
HOSPITAL_PORT=5003 python hospital_server/app.py
```

**Option B: Docker Compose**
```bash
docker-compose -f docker/docker-compose.yml up
```

**Option C: Kubernetes**
```bash
bash deploy-k8s.sh
```

### 3. Run Federated Learning
```bash
python orchestrator.py
```

---

## 📋 Common Tasks

### Check Server Health
```bash
curl http://localhost:5000/health
curl http://localhost:5001/health
```

### View Logs
```bash
# Local
tail -f logs/main_server_*.log
tail -f logs/hospital_server_*.log

# Docker
docker-compose -f docker/docker-compose.yml logs -f

# Kubernetes
kubectl logs -n fedcare deployment/main-server
```

### Get Training Metrics
```bash
curl http://localhost:5000/metrics | jq '.'
```

### Reset System
```bash
curl -X POST http://localhost:5000/reset
```

### Get System Status
```bash
curl http://localhost:5000/status | jq '.'
```

---

## 🔧 Configuration

### Edit .env File
```env
NUM_ROUNDS=10              # Federated learning rounds
NUM_HOSPITALS=3            # Number of hospitals
EPOCHS_PER_ROUND=5         # Local training epochs
BATCH_SIZE=32              # Training batch size
LEARNING_RATE=0.001        # Model learning rate
LOG_LEVEL=INFO             # Logging level
```

### Common Adjustments

**For faster training:**
```env
EPOCHS_PER_ROUND=3
BATCH_SIZE=64
NUM_HOSPITALS=2
```

**For better accuracy:**
```env
EPOCHS_PER_ROUND=10
BATCH_SIZE=16
LEARNING_RATE=0.0005
```

**For memory-constrained systems:**
```env
BATCH_SIZE=8
NUM_HOSPITALS=2
```

---

## 🧪 Testing & Examples

### Run Examples
```bash
python examples.py
```

Options:
1. Basic Flow (recommended for first-time)
2. Non-IID Data (realistic scenario)
3. Direct API Calls (testing endpoints)
4. Orchestrator (automated)

### Test Individual Components
```python
# Test model creation
from shared.models import create_simple_model, compile_model
model = create_simple_model()
model = compile_model(model)
print("✓ Model created")

# Test data generation
from data_simulation.data_generator import load_and_split_mnist_data
data, test = load_and_split_mnist_data(num_hospitals=3)
print(f"✓ Generated data for {len(data)} hospitals")

# Test communication
from shared.communication import ServerCommunicator
comm = ServerCommunicator()
print("✓ Communication module loaded")
```

---

## 📊 API Endpoints Quick Reference

### Main Server (Cloud)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/initialize` | POST | Initialize FL process |
| `/get_global_model` | GET | Get current model |
| `/submit_update` | POST | Receive hospital updates |
| `/aggregate` | POST | Aggregate weights |
| `/metrics` | GET | Get training metrics |
| `/status` | GET | System status |
| `/reset` | POST | Reset system |

### Hospital Server (Edge)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/configure` | POST | Set hospital ID |
| `/load_data` | POST | Load training data |
| `/sync_and_train` | POST | Fetch model & train |
| `/training_history` | GET | Get local metrics |
| `/status` | GET | Hospital status |

---

## 🔍 Debugging

### Issue: Servers not connecting

```bash
# Check if ports are in use
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Issue: Out of memory

```bash
# Reduce batch size
BATCH_SIZE=8 python main_server/app.py

# Reduce number of hospitals
NUM_HOSPITALS=2 python hospital_server/app.py
```

### Issue: Data loading fails

```python
# Verify data format
import numpy as np
X = np.array([...])
y = np.array([...])
print(X.shape, y.shape)  # Should be (n_samples, 28, 28, 1) and (n_samples,)
```

### Issue: SSL/Certificate errors

```bash
# For development only (not production!)
export PYTHONHTTPSVERIFY=0
```

---

## 📈 Monitoring

### Key Metrics to Track

1. **Loss Convergence**
   ```bash
   curl http://localhost:5000/metrics | jq '.metrics[] | select(.round==0) | .metrics.loss'
   ```

2. **Accuracy Improvement**
   ```bash
   curl http://localhost:5000/metrics | jq '.metrics[] | .metrics.accuracy'
   ```

3. **Training Time**
   - Monitor logs for round completion times

4. **Resource Usage**
   - CPU: top/Activity Monitor
   - Memory: free/Memory tab
   - Network: iftop/Network tab

### Example Monitoring Script
```bash
#!/bin/bash
while true; do
    clear
    curl -s http://localhost:5000/status | jq '.'
    curl -s http://localhost:5000/metrics | jq '.metrics[-1]'
    sleep 5
done
```

---

## 🐳 Docker Commands

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up

# Stop services
docker-compose -f docker/docker-compose.yml down

# View logs
docker-compose -f docker/docker-compose.yml logs -f service-name

# Execute command in container
docker-compose -f docker/docker-compose.yml exec main-server bash

# View resource usage
docker stats

# Clean up
docker system prune -a
```

---

## ☸️ Kubernetes Commands

```bash
# Deploy
kubectl apply -f kubernetes/deployment.yml

# Check status
kubectl get pods -n fedcare
kubectl get svc -n fedcare

# View logs
kubectl logs -n fedcare deployment/main-server
kubectl logs -n fedcare deployment/hospital-server

# Port forward
kubectl port-forward -n fedcare svc/main-server 5000:5000
kubectl port-forward -n fedcare svc/hospital-server 5001:5001

# Scale
kubectl scale deployment hospital-server --replicas=5 -n fedcare

# Delete
kubectl delete namespace fedcare

# Get resource usage
kubectl top pods -n fedcare
```

---

## 🎯 Performance Tips

### Faster Training
- Increase `EPOCHS_PER_ROUND`
- Increase `BATCH_SIZE` (if memory available)
- Use GPU (if available)
- Reduce `NUM_HOSPITALS`

### Better Convergence
- Decrease `LEARNING_RATE`
- Increase `NUM_HOSPITALS`
- Use more training data
- Run more rounds

### Lower Memory
- Decrease `BATCH_SIZE`
- Decrease `NUM_HOSPITALS`
- Use simpler model
- Use float32 instead of float64

---

## 📚 File Structure Reference

```
FedCare/
├── main_server/app.py          # Cloud aggregation server
├── hospital_server/app.py      # Edge training server
├── shared/                      # Shared utilities
│   ├── config.py               # Configuration
│   ├── communication.py         # Network communication
│   ├── models.py               # Model definitions
│   └── logger.py               # Logging setup
├── data_simulation/
│   └── data_generator.py       # Data generation
├── docker/
│   ├── Dockerfile.main_server
│   ├── Dockerfile.hospital_server
│   └── docker-compose.yml
├── kubernetes/
│   └── deployment.yml          # K8s manifests
├── orchestrator.py             # FL orchestration
├── examples.py                 # Example usage
├── requirements.txt            # Dependencies
├── .env.example                # Configuration template
└── docs/                        # Documentation
    ├── README.md
    ├── ARCHITECTURE.md
    ├── API.md
    ├── INSTALLATION.md
    └── QUICKREF.md (this file)
```

---

## 🔗 Links & Resources

- **GitHub**: https://github.com/ChiragPrasad2006/FedCare
- **TensorFlow Federated**: https://www.tensorflow.org/federated
- **Docker**: https://www.docker.com/
- **Kubernetes**: https://kubernetes.io/
- **HIPAA Compliance**: https://www.hhs.gov/hipaa/

---

## ❓ FAQ

**Q: Can I use different hardware?**
A: Yes! The system auto-adapts to available resources.

**Q: How many hospitals can I run?**
A: Tested with 3-10. More requires more resources/distributed setup.

**Q: Can I use real patient data?**
A: Yes, but ensure HIPAA/GDPR compliance and data security.

**Q: What models are supported?**
A: Any TensorFlow/Keras model. Customize in `shared/models.py`.

**Q: Can I deploy to AWS?**
A: Yes! Use ECS, EKS, or SageMaker. See AWS docs.

---

Last Updated: May 2024

For more help, see the full documentation in `/docs/`
