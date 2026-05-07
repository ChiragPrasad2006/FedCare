# FedCare Troubleshooting Guide

## Common Issues and Solutions

### Server Issues

#### ❌ "Address already in use" / Port conflict

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
HOSPITAL_PORT=5010 python hospital_server/app.py
```

---

#### ❌ "Connection refused"

**Error Message:**
```
requests.exceptions.ConnectionError: 
Failed to establish connection to http://localhost:5000
```

**Solutions:**

1. Check if server is running:
```bash
curl http://localhost:5000/health
ps aux | grep app.py  # macOS/Linux
tasklist | findstr python  # Windows
```

2. Start servers:
```bash
python main_server/app.py  # Terminal 1
HOSPITAL_PORT=5001 python hospital_server/app.py  # Terminal 2
```

3. Check firewall:
```bash
# macOS
sudo lsof -i :5000

# Windows
netsh advfirewall firewall show rule name=all | findstr 5000
```

---

#### ❌ "Timeout" errors

**Error Message:**
```
requests.exceptions.Timeout: 
Connection timeout after 30s
```

**Solutions:**

1. Server might be slow, increase timeout:
```python
from shared.communication import ServerCommunicator
comm = ServerCommunicator(timeout=60)  # 60 seconds
```

2. Check server logs:
```bash
tail -f logs/main_server_*.log
```

3. Check system resources:
```bash
top  # macOS/Linux
Task Manager  # Windows
```

---

### Data Issues

#### ❌ "No training data provided"

**Error Message:**
```json
{
  "error": "No training data provided"
}
```

**Solution:**

Ensure data is in correct format:
```python
import numpy as np
from data_simulation.data_generator import load_and_split_mnist_data

# Load data correctly
hospital_data, test_data = load_and_split_mnist_data(num_hospitals=3)
X_train, y_train = hospital_data[0]

# Verify format
assert isinstance(X_train, np.ndarray)
assert isinstance(y_train, np.ndarray)
assert len(X_train) > 0
print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

# Convert to list for JSON
X_list = X_train.reshape(X_train.shape[0], -1).tolist()
y_list = y_train.tolist()
```

---

#### ❌ "Data shape mismatch"

**Error Message:**
```
ValueError: Cannot reshape array of size X into shape (Y,)
```

**Solution:**

Verify data shape matches model input:
```python
from shared.models import create_simple_model

model = create_simple_model(input_shape=(28, 28, 1), num_classes=10)

# Verify input shape
X_train.shape  # Should be (n_samples, 28, 28, 1)
y_train.shape  # Should be (n_samples,)

# Reshape if needed
if len(X_train.shape) == 2:  # (n_samples, 784)
    X_train = X_train.reshape(-1, 28, 28, 1)
```

---

### Memory Issues

#### ❌ "MemoryError" or "Out of memory"

**Error Message:**
```
MemoryError: Unable to allocate 1.23 GiB for an array
```

**Solutions:**

1. **Reduce batch size:**
```bash
# In .env
BATCH_SIZE=8  # Instead of 32
```

2. **Reduce number of hospitals:**
```bash
# In .env
NUM_HOSPITALS=2  # Instead of 3
```

3. **Clear memory:**
```bash
# Restart Python kernel
# Clear Docker cache: docker system prune
# Restart system
```

4. **Monitor memory usage:**
```bash
# macOS/Linux
watch -n 1 free -h

# Windows
Get-Process | Select-Object ProcessName, @{Name="MemoryMB"; Expression={[math]::Round($_.WorkingSet/1MB, 2)}}
```

---

#### ❌ "CUDA out of memory" (GPU)

**Error Message:**
```
RuntimeError: CUDA out of memory. 
Tried to allocate 1.23 GiB
```

**Solutions:**

1. Use CPU instead:
```bash
export CUDA_VISIBLE_DEVICES=""  # Disable GPU
```

2. Reduce model size or batch size

3. Clear GPU memory:
```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

---

### Model Issues

#### ❌ "Model compilation error"

**Error Message:**
```
ValueError: Unknown optimizer: adam
```

**Solution:**

Ensure TensorFlow is properly imported:
```python
from shared.models import compile_model

# Models are pre-configured
model = compile_model(model, learning_rate=0.001)

# Or compile manually
model.compile(
    optimizer='adam',  # or keras.optimizers.Adam()
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

#### ❌ "Model weight mismatch"

**Error Message:**
```
ValueError: Layer weight has shape (X,) but the model expects (Y,)
```

**Solution:**

Ensure weights come from same model architecture:
```python
from shared.models import create_simple_model

# Use same model creation function
model1 = create_simple_model()
model2 = create_simple_model()

# Get and set weights
weights = model1.get_weights()
model2.set_weights(weights)  # OK
```

---

### Network Issues

#### ❌ "Certificate verification failed"

**Error Message:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**

For development only (not production):
```bash
export PYTHONHTTPSVERIFY=0
# or
python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context"
```

Better solution: Use proper certificates:
```python
import requests
requests.post(url, verify=True)  # Use proper SSL certs
```

---

#### ❌ "Network is unreachable"

**Error Message:**
```
ConnectionError: Network is unreachable
```

**Solutions:**

1. Check network connectivity:
```bash
ping localhost
ping 127.0.0.1
```

2. Check Docker network (if using Docker):
```bash
docker network ls
docker network inspect fedcare-network
```

3. For Kubernetes:
```bash
kubectl get pods -n fedcare
kubectl describe pod <pod-name> -n fedcare
```

---

### Docker Issues

#### ❌ "Docker: command not found"

**Solution:**

Install Docker from: https://www.docker.com/products/docker-desktop

---

#### ❌ "Cannot connect to Docker daemon"

**Error Message:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**

```bash
# macOS/Linux
sudo service docker start
# or
docker run hello-world

# macOS
open /Applications/Docker.app

# Windows
Start Docker Desktop application
```

---

#### ❌ "Docker container exits immediately"

**Solution:**

Check container logs:
```bash
docker-compose -f docker/docker-compose.yml logs main-server
docker-compose -f docker/docker-compose.yml logs hospital-server-1
```

Common causes:
- Missing dependencies
- Port conflict
- Configuration error

---

### Kubernetes Issues

#### ❌ "kubectl: command not found"

**Solution:**

Install kubectl: https://kubernetes.io/docs/tasks/tools/

---

#### ❌ "No resources found in fedcare namespace"

**Error Message:**
```
No resources found in fedcare namespace
```

**Solution:**

Deploy first:
```bash
kubectl apply -f kubernetes/deployment.yml
kubectl get pods -n fedcare
```

---

#### ❌ "Pod stuck in Pending state"

**Error Message:**
```
kubectl get pods -n fedcare
# Output: Pod is Pending
```

**Solution:**

Check resources and events:
```bash
kubectl describe pod <pod-name> -n fedcare
kubectl top nodes  # Check node resources
kubectl get events -n fedcare  # Check events
```

---

### Python Issues

#### ❌ "ModuleNotFoundError"

**Error Message:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**

Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install tensorflow tensorflow-federated torch numpy flask
```

---

#### ❌ "venv activation doesn't work"

**Solution:**

Correct activation:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

---

### Orchestration Issues

#### ❌ "Orchestrator script fails"

**Solution:**

Run servers first:
```bash
# Terminal 1
python main_server/app.py

# Terminal 2
HOSPITAL_PORT=5001 python hospital_server/app.py
HOSPITAL_PORT=5002 python hospital_server/app.py
HOSPITAL_PORT=5003 python hospital_server/app.py

# Terminal 3
python orchestrator.py
```

---

#### ❌ "Some hospitals failed training"

**Error Message:**
```
Some hospitals failed training
```

**Debugging:**

Check hospital logs:
```bash
tail -f logs/hospital_server_*.log
# Look for specific errors
```

Verify hospital is configured:
```bash
curl http://localhost:5001/status
```

---

## Performance Issues

#### ⚠️ "Training is very slow"

**Optimization:**

1. Increase batch size:
```env
BATCH_SIZE=64  # Faster but needs more memory
```

2. Reduce epochs:
```env
EPOCHS_PER_ROUND=3  # Faster but less training
```

3. Reduce hospitals:
```env
NUM_HOSPITALS=2  # Parallel training faster
```

4. Use GPU:
```bash
# Check if GPU available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Logging and Debugging

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### View Detailed Logs

```bash
# Main server
tail -f logs/main_server_*.log | grep ERROR

# All logs
tail -f logs/*.log

# Filter by timestamp
grep "2024-05-07" logs/main_server_*.log
```

### Capture Full Output

```bash
# Run with output capture
python main_server/app.py 2>&1 | tee server.log
```

---

## Getting Help

If issues persist:

1. **Check documentation**: `/docs/`
2. **Review logs**: `./logs/`
3. **Test components separately**: `examples.py`
4. **Open GitHub issue**: Include error message and logs
5. **Check FAQ**: Above section

---

Last Updated: May 2024
