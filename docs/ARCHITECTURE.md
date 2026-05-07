# FedCare System Architecture

## Overview

FedCare is a federated learning system designed for privacy-preserving healthcare AI. It enables multiple hospitals to collaboratively train machine learning models without sharing sensitive patient data.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud / Main Server                      │
│  - Global Model Manager                                     │
│  - Weight Aggregator                                        │
│  - Metrics Collector                                        │
│  - REST API Server                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Hospital│ │Hospital│ │Hospital│
    │Server 1│ │Server 2│ │Server 3│
    ├────────┤ ├────────┤ ├────────┤
    │Local   │ │Local   │ │Local   │
    │Model   │ │Model   │ │Model   │
    │Trainer │ │Trainer │ │Trainer │
    └────────┘ └────────┘ └────────┘
```

## Components

### 1. Main Server (Cloud)

**Purpose**: Orchestrate federated learning and aggregate model updates

**Responsibilities**:
- Initialize and maintain global model
- Distribute model to hospitals
- Receive and validate model updates
- Aggregate weights from all hospitals
- Track metrics and statistics
- Manage training rounds

**APIs**:
- `/initialize` - Start federated learning process
- `/get_global_model` - Distribute current model
- `/submit_update` - Receive hospital updates
- `/aggregate` - Perform weight aggregation
- `/metrics` - Query training metrics
- `/status` - Get system status

**Database**: In-memory (can be extended to MongoDB/PostgreSQL)

### 2. Hospital Server (Edge)

**Purpose**: Train models on local patient data securely

**Responsibilities**:
- Receive global model from cloud
- Train on local, private data
- Send only model weights (not data)
- Track local training metrics
- Handle data loading and preprocessing

**APIs**:
- `/configure` - Set hospital identity
- `/load_data` - Receive training data
- `/sync_and_train` - Fetch model, train, submit update
- `/training_history` - Query local metrics
- `/status` - Get hospital status

**Data**: Local only, never transmitted outside hospital

### 3. Shared Utilities

**Files**:
- `config.py` - Configuration management
- `communication.py` - Network communication
- `models.py` - Neural network definitions
- `logger.py` - Logging setup

**Key Functions**:
- Model creation and compilation
- Weight serialization/deserialization
- Weight averaging for aggregation
- HTTP communication with retries

## Data Flow

### Training Round Process

```
1. Initialization
   └─> Main Server creates global model

2. Model Distribution
   └─> Main Server broadcasts model to hospitals

3. Local Training
   ├─> Hospital 1: fetch model → train locally → compute updates
   ├─> Hospital 2: fetch model → train locally → compute updates
   └─> Hospital 3: fetch model → train locally → compute updates

4. Update Submission
   ├─> Hospital 1: send weights + metrics to Main Server
   ├─> Hospital 2: send weights + metrics to Main Server
   └─> Hospital 3: send weights + metrics to Main Server

5. Aggregation
   └─> Main Server: average all weights → create new global model

6. Next Round
   └─> Repeat from step 2
```

## Privacy Considerations

### Current Implementation

- ✓ Raw patient data never leaves hospital servers
- ✓ Only model weights are transmitted
- ✓ No server can access raw training data
- ✓ Hospitals remain completely autonomous

### Future Enhancements

- Differential Privacy: Add noise to prevent inference attacks
- Secure Aggregation: Cryptographic aggregation without central server
- Homomorphic Encryption: Compute on encrypted data
- Federated Analytics: Privacy-preserving data analysis

## Communication Protocol

### Weight Serialization

```python
# Serialize
weights = model.get_weights()  # List of numpy arrays
weights_bytes = pickle.dumps(weights)
weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')

# Transmit
POST /submit_update
{
    "hospital_id": "hospital_1",
    "weights": "base64_string",
    "metrics": {...}
}

# Deserialize
weights_bytes = base64.b64decode(weights_b64)
weights = pickle.loads(weights_bytes)
model.set_weights(weights)
```

### Metrics Format

```json
{
    "loss": 0.4532,
    "accuracy": 0.8921,
    "epochs": 5,
    "samples": 5000
}
```

## Model Architecture

### Default Model (Simple CNN)

```
Input (28x28x1)
  ↓
Conv2D (32 filters, 3x3)
  ↓
MaxPool (2x2)
  ↓
Conv2D (64 filters, 3x3)
  ↓
MaxPool (2x2)
  ↓
Conv2D (64 filters, 3x3)
  ↓
Flatten
  ↓
Dense (64, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (10, Softmax)
  ↓
Output (10 classes)
```

### Model Aggregation

```
Global Model v0 {w1₀, w2₀, ..., wₙ₀}
         ↓
    [Each Hospital trains locally]
         ↓
Hospital 1: {w1₁, w2₁, ..., wₙ₁}
Hospital 2: {w1₂, w2₂, ..., wₙ₂}
Hospital 3: {w1₃, w2₃, ..., wₙ₃}
         ↓
Average weights:
w_avg = (w_hospital1 + w_hospital2 + w_hospital3) / 3
         ↓
Global Model v1 {w1_avg, w2_avg, ..., wₙ_avg}
```

## Scalability

### Current Limitations
- In-memory storage
- Single aggregation server
- No load balancing

### Future Scaling Options

1. **Horizontal Scaling**
   - Multiple main server replicas
   - Load balancer for distribution
   - Shared cache (Redis)
   - Database persistence

2. **Edge Computing**
   - Regional aggregators
   - Hierarchical federation
   - Local aggregation at region level

3. **Kubernetes Deployment**
   - Auto-scaling pods
   - Service mesh for communication
   - Persistent volume claims
   - Distributed storage

## Performance Metrics

### Monitoring Points

1. **Training Performance**
   - Loss per round
   - Accuracy per round
   - Convergence speed

2. **System Performance**
   - Model aggregation time
   - Network communication time
   - Total round time

3. **Resource Utilization**
   - CPU usage per server
   - Memory consumption
   - Network bandwidth

## Security

### Current Mechanisms
- No authentication (for development)
- CORS enabled for testing
- No HTTPS (development only)

### Production Requirements
- API authentication (OAuth2/JWT)
- HTTPS/TLS encryption
- Rate limiting
- Input validation
- Audit logging

## Deployment Scenarios

### 1. Local Development
```
Single machine running all servers
Maximum 3 hospitals (resource limited)
Testing and development
```

### 2. Docker Compose
```
Multiple containers on single host
Production-like networking
Easy scaling
Good for testing
```

### 3. Kubernetes
```
Multi-node cluster
Auto-scaling
Load balancing
Monitoring and logging
True production setup
```

### 4. Cloud Deployment
```
AWS ECS/EKS
Azure Kubernetes Service
Google Cloud Run
Managed services
Global distribution
```

## Fault Tolerance

### Current Implementation
- Retry logic with exponential backoff
- Request timeouts
- Graceful error handling

### Improvements Needed
- Health checks
- Automatic failover
- Replica synchronization
- Distributed transactions
- Message queue (RabbitMQ/Kafka)

## Extensibility

### Easy to Add
1. New model architectures
2. Different datasets
3. Custom aggregation algorithms
4. Privacy mechanisms
5. New deployment targets

### Plugin Architecture
```
models.py - Add new models
data_simulator/ - Add new datasets
aggregators/ - Add aggregation algorithms
privacy/ - Add privacy mechanisms
backends/ - Add storage backends
```

## Testing Strategy

### Unit Tests
- Model creation
- Weight averaging
- Serialization

### Integration Tests
- Server-to-server communication
- Full training round
- Error handling

### Load Tests
- Multiple hospitals
- Large models
- High-frequency updates

## Monitoring and Logging

### Log Levels
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Warning messages for unusual situations
- ERROR: Error messages for failures

### Key Metrics to Track
- Model loss/accuracy trends
- Round completion time
- Network latency
- Server resource usage
- Error rates

## References

### Papers
- FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data
- Differential Privacy: A Survey of Results
- Secure Aggregation in Federated Learning

### Frameworks
- TensorFlow Federated
- PySyft
- FATE (Federated AI Technology Enabler)

### Standards
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- HL7 (Health Level 7)

---

Last Updated: May 2024
