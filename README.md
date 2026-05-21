README: FedCare - Federated Learning for Privacy-Preserving Healthcare

FedCare is an advanced system for training machine learning models on sensitive patient data without exposing the raw data. It uses federated learning to keep data decentralized while building a collaborative global model.

## Project Overview

### Architecture
- **Main Server (Cloud)**: Central aggregation server that coordinates federated learning rounds
- **Hospital Servers (Edge)**: Local edge servers at each hospital that train models on local patient data
- **Communication Layer**: Secure communication between servers via REST APIs
- **Data Simulation**: Synthetic and real dataset support for testing

### Key Features
1. Privacy-preserving machine learning
2. Decentralized data handling
3. Multi-hospital collaboration
4. Real-time model aggregation
5. Comprehensive logging and metrics
6. Docker and Kubernetes support
7. Non-IID (heterogeneous) data handling

## Project Structure

```
FedCare/
├── main_server/          # Cloud aggregation server
│   └── app.py           # Main Flask application
├── hospital_server/      # Edge training servers
│   └── app.py           # Hospital Flask application
├── shared/              # Shared utilities
│   ├── config.py        # Configuration
│   ├── communication.py  # Server communication
│   ├── models.py        # Neural network models
│   └── logger.py        # Logging setup
├── docker/              # Docker configurations
│   ├── Dockerfile.main_server
│   ├── Dockerfile.hospital_server
│   └── docker-compose.yml
├── kubernetes/          # Kubernetes manifests
│   └── deployment.yml
├── data_simulation/     # Data generation
│   └── data_generator.py
├── orchestrator.py      # Federated learning orchestration
├── requirements.txt     # Python dependencies
└── docs/               # Documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Docker & Docker Compose (optional)
- Kubernetes cluster (optional)

### Setup

1. Clone the repository
```bash
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

## Quick Start

### Option 1: Run Locally (Development)

1. Start Main Server in one terminal
```bash
python main_server/app.py
```

2. Start Hospital Servers in separate terminals
```bash
HOSPITAL_PORT=5001 python hospital_server/app.py
HOSPITAL_PORT=5002 python hospital_server/app.py
HOSPITAL_PORT=5003 python hospital_server/app.py
```

3. Run Orchestration Script
```bash
python orchestrator.py
```

4. Open the dashboards
```text
Main server:      http://localhost:5000/
Hospital server:  http://localhost:5001/
Hospital server:  http://localhost:5002/
```

### Option 2: Run with Docker Compose

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up

# Run orchestrator
python orchestrator.py
```

### Option 3: Deploy to Kubernetes

```bash
# Follow the full local Kubernetes guide
# docs/LOCAL_KUBERNETES_DEPLOYMENT.md
```

Recommended for college demos:

- run Kubernetes locally with `Minikube` or `Docker Desktop`
- use your own machine's CPU and RAM instead of paid cloud compute
- expose dashboards with `kubectl port-forward`

## API Endpoints

### Main Server

- `GET /health` - Health check
- `GET /` - Main dashboard UI
- `POST /initialize` - Initialize federated learning
- `GET /get_global_model` - Get current global model
- `POST /submit_update` - Receive hospital updates
- `POST /aggregate` - Trigger aggregation
- `GET /metrics` - Get training metrics
- `GET /dashboard_data` - Dashboard data feed
- `GET /status` - Get current status
- `POST /reset` - Reset system

### Hospital Server

- `GET /health` - Health check
- `GET /` - Hospital dashboard UI
- `POST /configure` - Configure hospital
- `POST /load_data` - Load local training data
- `POST /load_demo_training_data` - Generate demo training data
- `POST /upload_patient_records` - Upload patient records for anonymization preview
- `POST /generate_demo_records` - Generate demo patient records
- `GET /patient_records` - Retrieve raw or anonymized patient records
- `POST /sync_and_train` - Fetch model and train
- `GET /training_history` - Get training history
- `GET /dashboard_data` - Dashboard data feed
- `GET /status` - Get hospital status

## Usage Examples

### Python API

```python
from orchestrator import FederatedLearningOrchestrator
from data_simulation.data_generator import load_and_split_mnist_data

# Initialize
orchestrator = FederatedLearningOrchestrator(
    main_server_url="http://localhost:5000",
    hospital_urls=[
        "http://localhost:5001",
        "http://localhost:5002",
        "http://localhost:5003"
    ]
)

# Initialize system
orchestrator.initialize_system()

# Load data
hospital_data, test_data = load_and_split_mnist_data(num_hospitals=3)
orchestrator.load_hospital_data(hospital_data)

# Run federated learning
orchestrator.run_federated_learning(num_rounds=10)

# Get metrics
metrics = orchestrator.get_metrics()
print(metrics)
```

### cURL Commands

```bash
# Health check
curl http://localhost:5000/health

# Initialize
curl -X POST http://localhost:5000/initialize

# Get global model
curl http://localhost:5000/get_global_model

# Configure hospital
curl -X POST http://localhost:5001/configure \
  -H "Content-Type: application/json" \
  -d '{"hospital_id": "hospital_1"}'

# Load data
curl -X POST http://localhost:5001/load_data \
  -H "Content-Type: application/json" \
  -d @data.json

# Sync and train
curl -X POST http://localhost:5001/sync_and_train \
  -H "Content-Type: application/json" \
  -d '{"round": 0}'

# Get metrics
curl http://localhost:5000/metrics
```

## Configuration

Edit `.env` file to configure:

- `NUM_ROUNDS`: Number of federated learning rounds (default: 10)
- `NUM_HOSPITALS`: Number of hospital servers (default: 3)
- `EPOCHS_PER_ROUND`: Local training epochs per round (default: 5)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Model learning rate (default: 0.001)

## Data Simulation

### Generate MNIST Data

```python
from data_simulation.data_generator import load_and_split_mnist_data

hospital_data, test_data = load_and_split_mnist_data(
    num_hospitals=3,
    test_size=0.2
)
```

### Generate Non-IID Data

```python
from data_simulation.data_generator import create_non_iid_data

# Simulates realistic federated scenario with heterogeneous data
hospital_data = create_non_iid_data(
    num_hospitals=3,
    num_samples_per_hospital=1000,
    num_classes=10
)
```

## Architecture Details

### Federated Learning Process

1. **Initialization**: Main server initializes global model
2. **Distribution**: Hospitals fetch current global model
3. **Local Training**: Each hospital trains model on local data
4. **Update Submission**: Hospitals send model weights to main server
5. **Aggregation**: Main server averages weights from all hospitals
6. **Next Round**: Process repeats for specified rounds

### Privacy Considerations

- Raw patient data never leaves hospital servers
- Only model weights are transmitted
- Communication can be encrypted (future enhancement)
- Differential privacy support (future enhancement)

## Presentation Dashboards

FedCare now includes built-in dashboards for presentations and demos:

- hospital-side patient record upload and anonymization preview
- local training controls and history
- main-server privacy removal summary
- federated training status and accuracy

## Google Cloud Hosting

For a presentation-friendly deployment with `1` main server and `2` hospital servers on Google Cloud, see:

- `docs/GOOGLE_CLOUD_DEPLOYMENT.md`

Important:

- Google Cloud usually still requires a billing account and payment method, even when a service has a free tier
- TensorFlow-based services can exceed free limits, so local Kubernetes is the safer no-cost option for student projects

## Local Kubernetes Hosting

For the recommended no-cloud-cost setup with `1` main server and `2` hospital servers on your own machine, see:

- `docs/LOCAL_KUBERNETES_DEPLOYMENT.md`

## Monitoring and Logging

Logs are saved to `./logs/` directory with timestamps.

### View Logs

```bash
# Main server logs
tail -f logs/main_server_*.log

# Hospital server logs
tail -f logs/hospital_server_*.log
```

## Performance Metrics

Monitor:
- Loss per round
- Accuracy per round
- Training time per hospital
- Aggregation time
- Network communication overhead

Metrics are available via `/metrics` endpoint on main server.

## Troubleshooting

### Servers not connecting

1. Check if servers are running
   ```bash
   curl http://localhost:5000/health
   curl http://localhost:5001/health
   ```

2. Check logs for errors
   ```bash
   tail -f logs/main_server_*.log
   ```

3. Verify network connectivity
   ```bash
   ping localhost
   ```

### Data loading fails

1. Ensure data is in correct format (numpy arrays)
2. Check data shape matches model input
3. Verify sufficient memory available

### Memory issues

1. Reduce `BATCH_SIZE` in .env
2. Reduce `NUM_HOSPITALS`
3. Increase available memory

## AWS Deployment (Future)

Support for AWS SageMaker deployment with:
- Managed training jobs
- Distributed training
- Automatic scaling
- Model registry integration

## Advanced Features (Future)

- Differential Privacy
- Secure Multi-Party Computation
- Advanced data heterogeneity handling
- Custom aggregation algorithms
- Real HIPAA-compliant encryption
- MongoDB backend for persistence
- Web dashboard for monitoring

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## License

MIT License - See LICENSE file

## Citation

If you use FedCare in research, please cite:

```
@software{fedcare2024,
  title={FedCare: Federated Learning for Privacy-Preserving Healthcare},
  author={Chirag Prasad},
  year={2024}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

## Acknowledgments

- TensorFlow Federated team
- Open source community
- Healthcare privacy advocates

---

**Last Updated**: May 2026
