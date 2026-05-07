"""
Hospital Server (Edge) - Trains model on local data
"""
import os
import json
import logging
import requests
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    setup_logger, create_simple_model, compile_model,
    get_model_weights, set_model_weights,
    MAIN_SERVER_HOST, MAIN_SERVER_PORT, EPOCHS_PER_ROUND,
    BATCH_SIZE, LEARNING_RATE
)

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


def initialize_local_model():
    """Initialize the local model"""
    global local_model
    logger.info("Initializing local model...")
    local_model = create_simple_model()
    local_model = compile_model(local_model, learning_rate=LEARNING_RATE)
    logger.info("Local model initialized successfully")


def fetch_global_model():
    """Fetch global model from main server"""
    global local_model
    
    try:
        url = f"http://{MAIN_SERVER_HOST}:{MAIN_SERVER_PORT}/get_global_model"
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
        url = f"http://{MAIN_SERVER_HOST}:{MAIN_SERVER_PORT}/submit_update"
        
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


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('HOSPITAL_PORT', 5001))
    
    logger.info(f"Starting Hospital Server on localhost:{port}")
    initialize_local_model()
    app.run(host='0.0.0.0', port=port, debug=False)
