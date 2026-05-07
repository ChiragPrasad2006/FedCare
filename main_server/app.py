"""
Main Server (Cloud) - Aggregates model updates from hospitals
"""
import os
import json
import logging
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    setup_logger, create_simple_model, compile_model, 
    get_model_weights, set_model_weights, average_weights,
    MAIN_SERVER_HOST, MAIN_SERVER_PORT, NUM_ROUNDS, NUM_HOSPITALS,
    LEARNING_RATE
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logger("main_server")

# Global variables
global_model = None
current_round = 0
hospital_updates = {}
training_metrics = []
lock = threading.Lock()


def initialize_global_model():
    """Initialize the global model"""
    global global_model
    logger.info("Initializing global model...")
    global_model = create_simple_model()
    global_model = compile_model(global_model, learning_rate=LEARNING_RATE)
    logger.info("Global model initialized successfully")


def aggregate_weights(round_num):
    """Aggregate weights from all hospitals"""
    global global_model, hospital_updates
    
    with lock:
        logger.info(f"Aggregating weights for round {round_num}...")
        
        if not hospital_updates:
            logger.warning("No hospital updates received for aggregation")
            return False
        
        # Collect all weights
        weights_list = []
        hospital_count = 0
        
        for hospital_id, data in hospital_updates.items():
            if 'weights' in data:
                try:
                    from shared.communication import ServerCommunicator
                    comm = ServerCommunicator()
                    weights = comm.receive_model_weights(data['weights'])
                    if weights is not None:
                        weights_list.append(weights)
                        hospital_count += 1
                except Exception as e:
                    logger.error(f"Error processing weights from {hospital_id}: {e}")
        
        logger.info(f"Received weights from {hospital_count} hospitals")
        
        if weights_list:
            # Average weights
            averaged_weights = average_weights(weights_list)
            set_model_weights(global_model, averaged_weights)
            logger.info(f"Model aggregated successfully with {hospital_count} updates")
            
            # Log metrics
            for hospital_id, data in hospital_updates.items():
                if 'metrics' in data:
                    training_metrics.append({
                        'round': round_num,
                        'hospital_id': hospital_id,
                        'metrics': data['metrics'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Clear hospital updates for next round
            hospital_updates.clear()
            return True
        
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'current_round': current_round
    }), 200


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize federated learning process"""
    global current_round
    
    logger.info("Received initialization request")
    current_round = 0
    hospital_updates.clear()
    initialize_global_model()
    
    return jsonify({
        'message': 'Federated learning initialized',
        'model_initialized': True,
        'start_round': 0
    }), 200


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Get current global model weights"""
    if global_model is None:
        initialize_global_model()
    
    try:
        weights = get_model_weights(global_model)
        from shared.communication import ServerCommunicator
        comm = ServerCommunicator()
        
        import pickle
        import base64
        weights_bytes = pickle.dumps(weights)
        weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
        
        logger.info(f"Sending global model for round {current_round}")
        
        return jsonify({
            'weights': weights_b64,
            'round': current_round,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving global model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receive model updates from hospitals"""
    global current_round, hospital_updates
    
    try:
        data = request.get_json()
        hospital_id = data.get('hospital_id')
        
        if not hospital_id:
            return jsonify({'error': 'hospital_id required'}), 400
        
        logger.info(f"Received update from hospital {hospital_id}")
        
        with lock:
            hospital_updates[hospital_id] = {
                'weights': data.get('weights'),
                'metrics': data.get('metrics', {}),
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify({
            'message': 'Update received',
            'hospital_id': hospital_id,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in submit_update: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/aggregate', methods=['POST'])
def aggregate():
    """Trigger aggregation and start new round"""
    global current_round
    
    try:
        data = request.get_json() or {}
        trigger_next_round = data.get('trigger_next_round', True)
        
        logger.info(f"Starting aggregation for round {current_round}")
        
        success = aggregate_weights(current_round)
        
        if success and trigger_next_round:
            current_round += 1
            logger.info(f"Advanced to round {current_round}")
        
        return jsonify({
            'message': 'Aggregation completed',
            'round': current_round,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in aggregate: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get training metrics"""
    try:
        return jsonify({
            'metrics': training_metrics,
            'total_rounds': current_round,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    with lock:
        pending_updates = len(hospital_updates)
    
    return jsonify({
        'current_round': current_round,
        'pending_hospital_updates': pending_updates,
        'total_hospitals_reporting': len(hospital_updates),
        'model_initialized': global_model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the federated learning process"""
    global current_round, hospital_updates, training_metrics, global_model
    
    logger.info("Resetting federated learning process...")
    
    with lock:
        current_round = 0
        hospital_updates.clear()
        training_metrics.clear()
        global_model = None
    
    initialize_global_model()
    
    return jsonify({
        'message': 'System reset successfully',
        'timestamp': datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    logger.info(f"Starting Main Server on {MAIN_SERVER_HOST}:{MAIN_SERVER_PORT}")
    initialize_global_model()
    app.run(host=MAIN_SERVER_HOST, port=MAIN_SERVER_PORT, debug=False)
