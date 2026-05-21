"""
Hospital Server (Edge) - Trains model on local data
"""
import os
import requests
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    setup_logger, create_federated_model, compile_model,
    get_model_weights, set_model_weights,
    MAIN_SERVER_HOST, MAIN_SERVER_PORT, EPOCHS_PER_ROUND,
    BATCH_SIZE, LEARNING_RATE, MAIN_SERVER_URL
)
from shared.privacy import analyze_and_anonymize_records, generate_demo_patient_records
from data_simulation.data_generator import generate_synthetic_patient_data

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logger("hospital_server")

# Global variables
hospital_id = None
local_model = None
current_round = 0
training_history = []
local_data = None
patient_records = []
anonymized_patient_records = []
privacy_snapshot = {
    'total_records': 0,
    'detected_pii_items': 0,
    'redacted_items': 0,
    'residual_risk_items': 0,
    'removal_accuracy': 100.0,
    'processed_at': None,
}


def initialize_local_model():
    """Initialize the local model"""
    global local_model
    logger.info("Initializing local model...")
    local_model = create_federated_model()
    local_model = compile_model(local_model, learning_rate=LEARNING_RATE)
    logger.info("Local model initialized successfully")


def get_main_server_base_url():
    """Resolve the main server base URL for local and cloud deployments."""
    if MAIN_SERVER_URL:
        return MAIN_SERVER_URL.rstrip('/')
    return f"http://{MAIN_SERVER_HOST}:{MAIN_SERVER_PORT}"


def summarize_training():
    """Build a concise training summary for the dashboard."""
    latest_metrics = training_history[-1]['metrics'] if training_history else {}
    avg_accuracy = (
        round(sum(item['metrics'].get('accuracy', 0) for item in training_history) / len(training_history), 4)
        if training_history else 0.0
    )
    return {
        'latest_metrics': latest_metrics,
        'avg_accuracy': avg_accuracy,
        'rounds_completed': len(training_history),
        'has_training_data': local_data is not None and len(local_data[0]) > 0,
    }


def publish_privacy_summary():
    """Send privacy snapshot and anonymized preview to the main server."""
    if not hospital_id:
        return False

    payload = {
        'hospital_id': hospital_id,
        'privacy_summary': privacy_snapshot,
        'sample_records': anonymized_patient_records[:5],
        'hospital_profile': {
            'hospital_id': hospital_id,
            'records_uploaded': len(patient_records),
            'training_rounds': len(training_history),
            'latest_sync': datetime.now().isoformat(),
        },
    }

    try:
        url = f"{get_main_server_base_url()}/submit_privacy_summary"
        response = requests.post(url, json=payload, timeout=15)
        return response.status_code == 200
    except Exception as exc:
        logger.warning(f"Unable to publish privacy summary to main server: {exc}")
        return False


def fetch_global_model():
    """Fetch global model from main server"""
    global local_model
    
    try:
        url = f"{get_main_server_base_url()}/get_global_model"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            from shared.communication import ServerCommunicator
            comm = ServerCommunicator()
            weights = comm.receive_model_weights(data['weights'])
            
            if weights is not None:
                set_model_weights(local_model, weights)
                logger.info(f"Successfully fetched and applied global model for round {data['round']}")
                return True
        else:
            logger.error(f"Failed to fetch global model: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching global model: {e}")
        return False


def train_on_local_data():
    """Train model on local data"""
    global local_model, current_round, training_history
    
    if local_model is None:
        logger.error("Local model not initialized")
        return None
    
    if local_data is None or len(local_data) == 0:
        logger.warning("No local data available for training")
        return None
    
    try:
        logger.info(f"Starting local training for round {current_round}")
        
        # Unpack data
        X_train, y_train = local_data
        
        # Train model
        history = local_model.fit(
            X_train, y_train,
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        # Extract metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        metrics = {
            'loss': float(final_loss),
            'accuracy': float(final_accuracy),
            'epochs': EPOCHS_PER_ROUND,
            'samples': len(X_train)
        }
        
        training_history.append({
            'round': current_round,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Local training completed - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during local training: {e}")
        return None


def submit_update():
    """Submit trained model update to main server"""
    global local_model, current_round, hospital_id
    
    if local_model is None:
        logger.error("Local model not trained")
        return False
    
    try:
        url = f"{get_main_server_base_url()}/submit_update"
        
        weights = get_model_weights(local_model)
        
        from shared.communication import ServerCommunicator
        comm = ServerCommunicator()
        
        import pickle
        import base64
        weights_bytes = pickle.dumps(weights)
        weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
        
        # Get latest metrics
        metrics = {}
        if training_history:
            metrics = training_history[-1]['metrics']
        
        payload = {
            'hospital_id': hospital_id,
            'weights': weights_b64,
            'metrics': metrics,
            'round': current_round
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Successfully submitted update to main server")
            return True
        else:
            logger.error(f"Failed to submit update: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error submitting update: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'hospital_id': hospital_id,
        'current_round': current_round,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def dashboard():
    """Serve hospital dashboard UI."""
    return render_template(
        'dashboard.html',
        default_hospital_id=hospital_id or 'hospital_1',
        main_server_url=get_main_server_base_url(),
    )


@app.route('/configure', methods=['POST'])
def configure():
    """Configure hospital server"""
    global hospital_id
    
    try:
        data = request.get_json()
        hospital_id = data.get('hospital_id', 'hospital_1')
        
        logger.info(f"Hospital configured with ID: {hospital_id}")
        
        initialize_local_model()
        
        return jsonify({
            'message': 'Hospital configured successfully',
            'hospital_id': hospital_id
        }), 200
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/load_data', methods=['POST'])
def load_data():
    """Load local training data"""
    global local_data
    
    try:
        data = request.get_json()
        
        # Expect data as lists (will be converted to numpy arrays)
        X_train = np.array(data.get('X_train', []))
        y_train = np.array(data.get('y_train', []))
        
        if len(X_train) == 0:
            return jsonify({'error': 'No training data provided'}), 400
        
        local_data = (X_train, y_train)
        
        logger.info(f"Loaded local data: {len(X_train)} samples")
        
        return jsonify({
            'message': 'Data loaded successfully',
            'num_samples': len(X_train)
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sync_and_train', methods=['POST'])
def sync_and_train():
    """Fetch global model and train on local data"""
    global current_round
    
    try:
        data = request.get_json() or {}
        current_round = data.get('round', current_round)
        
        logger.info(f"Starting sync and train for round {current_round}")
        
        # Fetch global model
        if not fetch_global_model():
            return jsonify({'error': 'Failed to fetch global model'}), 500
        
        # Train on local data
        metrics = train_on_local_data()
        if metrics is None:
            return jsonify({'error': 'Training failed'}), 500
        
        # Submit update
        if not submit_update():
            return jsonify({'error': 'Failed to submit update'}), 500

        publish_privacy_summary()
        
        return jsonify({
            'message': 'Sync and train completed successfully',
            'round': current_round,
            'metrics': metrics,
            'hospital_id': hospital_id,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in sync_and_train: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/training_history', methods=['GET'])
def get_training_history():
    """Get training history"""
    return jsonify({
        'hospital_id': hospital_id,
        'history': training_history,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/status', methods=['GET'])
def get_status():
    """Get current hospital status"""
    return jsonify({
        'hospital_id': hospital_id,
        'current_round': current_round,
        'model_initialized': local_model is not None,
        'has_data': local_data is not None and len(local_data[0]) > 0,
        'num_training_rounds': len(training_history),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """Provide combined dashboard data for the hospital UI."""
    return jsonify({
        'hospital_id': hospital_id,
        'status': {
            'current_round': current_round,
            'model_initialized': local_model is not None,
            'has_data': local_data is not None and len(local_data[0]) > 0,
            'num_training_rounds': len(training_history),
            'timestamp': datetime.now().isoformat(),
        },
        'training': summarize_training(),
        'privacy': privacy_snapshot,
        'records': {
            'raw_count': len(patient_records),
            'anonymized_count': len(anonymized_patient_records),
            'preview': anonymized_patient_records[:6],
        },
        'main_server_url': get_main_server_base_url(),
    }), 200


@app.route('/patient_records', methods=['GET'])
def get_patient_records():
    """Retrieve uploaded patient records for the dashboard."""
    view = request.args.get('view', 'anonymized')
    limit = max(1, min(int(request.args.get('limit', 20)), 100))
    dataset = patient_records if view == 'raw' else anonymized_patient_records
    return jsonify({
        'hospital_id': hospital_id,
        'view': view,
        'count': len(dataset),
        'records': dataset[:limit],
        'privacy': privacy_snapshot,
    }), 200


@app.route('/upload_patient_records', methods=['POST'])
def upload_patient_records():
    """Upload patient records for presentation and privacy analysis."""
    global patient_records, anonymized_patient_records, privacy_snapshot

    try:
        data = request.get_json() or {}
        records = data.get('records', data if isinstance(data, list) else [])

        if not isinstance(records, list) or not records:
            return jsonify({'error': 'Provide a non-empty list of patient records'}), 400

        patient_records = records
        anonymized_patient_records, privacy_snapshot = analyze_and_anonymize_records(records)
        publish_privacy_summary()

        return jsonify({
            'message': 'Patient records uploaded successfully',
            'privacy': privacy_snapshot,
            'preview': anonymized_patient_records[:5],
        }), 200
    except Exception as exc:
        logger.error(f"Error uploading patient records: {exc}")
        return jsonify({'error': str(exc)}), 500


@app.route('/generate_demo_records', methods=['POST'])
def generate_demo_records():
    """Generate presentation-friendly demo records."""
    global patient_records, anonymized_patient_records, privacy_snapshot

    data = request.get_json() or {}
    count = max(3, min(int(data.get('count', 8)), 20))
    generated_records = generate_demo_patient_records(hospital_id or 'hospital', count=count)
    patient_records = generated_records
    anonymized_patient_records, privacy_snapshot = analyze_and_anonymize_records(generated_records)
    publish_privacy_summary()

    return jsonify({
        'message': 'Demo patient records generated',
        'count': len(patient_records),
        'privacy': privacy_snapshot,
        'preview': anonymized_patient_records[:5],
    }), 200


@app.route('/load_demo_training_data', methods=['POST'])
def load_demo_training_data():
    """Generate lightweight training data for demos."""
    global local_data

    data = request.get_json() or {}
    sample_count = max(120, min(int(data.get('count', 240)), 1000))
    X_train, y_train = generate_synthetic_patient_data(num_samples=sample_count)
    local_data = (X_train.reshape(-1, 28, 28, 1), y_train)

    return jsonify({
        'message': 'Demo training data generated',
        'num_samples': len(local_data[0]),
        'shape': list(local_data[0].shape),
    }), 200


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('PORT', os.getenv('HOSPITAL_PORT', 5001)))
    
    logger.info(f"Starting Hospital Server on localhost:{port}")
    initialize_local_model()
    app.run(host='0.0.0.0', port=port, debug=False)
