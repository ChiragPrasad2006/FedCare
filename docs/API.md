# FedCare API Documentation

## Overview

FedCare provides REST APIs for federated learning operations. All endpoints accept and return JSON.

## Base URLs

- **Main Server**: `http://localhost:5000`
- **Hospital Servers**: `http://localhost:500X` (where X is 1, 2, 3, etc.)

## Authentication

Currently, no authentication is required. Future versions will support OAuth2 and API keys.

## Main Server Endpoints

### 1. Health Check

Check if the main server is running.

```http
GET /health
```

**Response (200)**
```json
{
  "status": "healthy",
  "timestamp": "2024-05-07T12:30:45.123456",
  "current_round": 0
}
```

### 2. Initialize

Initialize the federated learning system and reset the global model.

```http
POST /initialize
```

**Body**: None

**Response (200)**
```json
{
  "message": "Federated learning initialized",
  "model_initialized": true,
  "start_round": 0
}
```

### 3. Get Global Model

Retrieve the current global model weights.

```http
GET /get_global_model
```

**Response (200)**
```json
{
  "weights": "base64_encoded_weights_string",
  "round": 0,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 4. Submit Update

Receive model updates from hospital servers.

```http
POST /submit_update
```

**Request Body**
```json
{
  "hospital_id": "hospital_1",
  "weights": "base64_encoded_weights",
  "metrics": {
    "loss": 0.4532,
    "accuracy": 0.8921,
    "epochs": 5,
    "samples": 5000
  }
}
```

**Response (200)**
```json
{
  "message": "Update received",
  "hospital_id": "hospital_1",
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

**Error Response (400)**
```json
{
  "error": "hospital_id required"
}
```

### 5. Aggregate

Trigger model aggregation and advance to next round.

```http
POST /aggregate
```

**Request Body** (optional)
```json
{
  "trigger_next_round": true
}
```

**Response (200)**
```json
{
  "message": "Aggregation completed",
  "round": 1,
  "success": true,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 6. Get Metrics

Retrieve all training metrics from the system.

```http
GET /metrics
```

**Response (200)**
```json
{
  "metrics": [
    {
      "round": 0,
      "hospital_id": "hospital_1",
      "metrics": {
        "loss": 0.4532,
        "accuracy": 0.8921,
        "epochs": 5,
        "samples": 5000
      },
      "timestamp": "2024-05-07T12:30:45.123456"
    }
  ],
  "total_rounds": 1,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 7. Get Status

Get current system status and statistics.

```http
GET /status
```

**Response (200)**
```json
{
  "current_round": 0,
  "pending_hospital_updates": 2,
  "total_hospitals_reporting": 1,
  "model_initialized": true,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 8. Reset

Reset the entire federated learning system.

```http
POST /reset
```

**Response (200)**
```json
{
  "message": "System reset successfully",
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

## Hospital Server Endpoints

### 1. Health Check

```http
GET /health
```

**Response (200)**
```json
{
  "status": "healthy",
  "hospital_id": "hospital_1",
  "current_round": 0,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 2. Configure

Configure the hospital server with an ID.

```http
POST /configure
```

**Request Body**
```json
{
  "hospital_id": "hospital_1"
}
```

**Response (200)**
```json
{
  "message": "Hospital configured successfully",
  "hospital_id": "hospital_1"
}
```

### 3. Load Data

Load local training data into the hospital.

```http
POST /load_data
```

**Request Body**
```json
{
  "X_train": [[28x28 image as flat array], ...],
  "y_train": [0, 1, 2, ...]
}
```

**Response (200)**
```json
{
  "message": "Data loaded successfully",
  "num_samples": 5000
}
```

**Error Response (400)**
```json
{
  "error": "No training data provided"
}
```

### 4. Sync and Train

Fetch the global model and train locally.

```http
POST /sync_and_train
```

**Request Body** (optional)
```json
{
  "round": 0
}
```

**Response (200)**
```json
{
  "message": "Sync and train completed successfully",
  "round": 0,
  "metrics": {
    "loss": 0.4532,
    "accuracy": 0.8921,
    "epochs": 5,
    "samples": 5000
  },
  "hospital_id": "hospital_1",
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 5. Get Training History

Retrieve training history for this hospital.

```http
GET /training_history
```

**Response (200)**
```json
{
  "hospital_id": "hospital_1",
  "history": [
    {
      "round": 0,
      "metrics": {
        "loss": 0.4532,
        "accuracy": 0.8921,
        "epochs": 5,
        "samples": 5000
      },
      "timestamp": "2024-05-07T12:30:45.123456"
    }
  ],
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

### 6. Get Status

Get current hospital status.

```http
GET /status
```

**Response (200)**
```json
{
  "hospital_id": "hospital_1",
  "current_round": 0,
  "model_initialized": true,
  "has_data": true,
  "num_training_rounds": 1,
  "timestamp": "2024-05-07T12:30:45.123456"
}
```

## Error Handling

All errors follow this format:

```json
{
  "error": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `404`: Not Found
- `500`: Server Error
- `503`: Service Unavailable

## Rate Limiting

Currently not implemented. Future versions may include rate limiting.

## Examples

### Python Example

```python
import requests
import json

# Initialize
response = requests.post('http://localhost:5000/initialize')
print(response.json())

# Get model
response = requests.get('http://localhost:5000/get_global_model')
print(response.json())

# Get metrics
response = requests.get('http://localhost:5000/metrics')
metrics = response.json()
print(json.dumps(metrics, indent=2))
```

### cURL Example

```bash
# Initialize
curl -X POST http://localhost:5000/initialize

# Get model
curl http://localhost:5000/get_global_model

# Submit update
curl -X POST http://localhost:5000/submit_update \
  -H "Content-Type: application/json" \
  -d '{
    "hospital_id": "hospital_1",
    "weights": "base64_string",
    "metrics": {"loss": 0.45, "accuracy": 0.89}
  }'

# Get metrics
curl http://localhost:5000/metrics | jq '.'
```

### JavaScript Example

```javascript
// Initialize
fetch('http://localhost:5000/initialize', {
  method: 'POST'
})
.then(r => r.json())
.then(data => console.log(data));

// Get metrics
fetch('http://localhost:5000/metrics')
  .then(r => r.json())
  .then(data => console.log(data));
```

## Data Formats

### Weights Format

Weights are serialized using Python pickle and encoded in base64. When receiving, decode base64 and deserialize with pickle.

```python
import pickle
import base64

# Encode
weights_bytes = pickle.dumps(weights)
weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')

# Decode
weights_bytes = base64.b64decode(weights_b64)
weights = pickle.loads(weights_bytes)
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

## Webhooks

Not currently implemented. Future versions may support webhooks for event notifications.

## Versioning

Current API Version: 1.0

Breaking changes will increment the major version number.

## Support

For API issues and questions:
1. Check the documentation
2. Review example code
3. Check server logs
4. Open a GitHub issue

---

Last Updated: May 2024
