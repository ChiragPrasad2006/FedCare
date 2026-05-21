"""
Main Server (Cloud) - Aggregates model updates from hospitals
"""
import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    setup_logger, create_federated_model, compile_model, 
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
privacy_reports = {}
lock = threading.Lock()


def initialize_global_model():
    """Initialize the global model"""
    global global_model
    logger.info("Initializing global model...")
    global_model = create_federated_model()
    global_model = compile_model(global_model, learning_rate=LEARNING_RATE)
    logger.info("Global model initialized successfully")


def summarize_metrics():
    """Build summary metrics for dashboard consumption."""
    if not training_metrics:
        return {
            'latest_round_accuracy': 0.0,
            'latest_round_loss': 0.0,
            'average_accuracy': 0.0,
            'average_loss': 0.0,
        }

    latest_round = max(item['round'] for item in training_metrics)
    latest_items = [item for item in training_metrics if item['round'] == latest_round]

    latest_round_accuracy = float(np.mean([
        item['metrics'].get('accuracy', 0.0) for item in latest_items
    ])) if latest_items else 0.0
    latest_round_loss = float(np.mean([
        item['metrics'].get('loss', 0.0) for item in latest_items
    ])) if latest_items else 0.0

    return {
        'latest_round_accuracy': round(latest_round_accuracy * 100, 2),
        'latest_round_loss': round(latest_round_loss, 4),
        'average_accuracy': round(float(np.mean([
            item['metrics'].get('accuracy', 0.0) for item in training_metrics
        ])) * 100, 2),
        'average_loss': round(float(np.mean([
            item['metrics'].get('loss', 0.0) for item in training_metrics
        ])), 4),
    }


def summarize_privacy():
    """Aggregate privacy statistics reported by hospitals."""
    if not privacy_reports:
        return {
            'connected_hospitals': 0,
            'overall_removal_accuracy': 0.0,
            'total_records_processed': 0,
            'total_detected_pii': 0,
            'total_residual_risk': 0,
        }

    reports = list(privacy_reports.values())
    total_detected = sum(item['privacy_summary'].get('detected_pii_items', 0) for item in reports)
    total_redacted = sum(item['privacy_summary'].get('redacted_items', 0) for item in reports)
    total_residual = sum(item['privacy_summary'].get('residual_risk_items', 0) for item in reports)
    total_records = sum(item['privacy_summary'].get('total_records', 0) for item in reports)

    overall_accuracy = (
        round(max(total_redacted - total_residual, 0) / total_detected * 100, 2)
        if total_detected else 100.0
    )

    return {
        'connected_hospitals': len(reports),
        'overall_removal_accuracy': overall_accuracy,
        'total_records_processed': total_records,
        'total_detected_pii': total_detected,
        'total_residual_risk': total_residual,
    }


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


@app.route('/', methods=['GET'])
def dashboard():
    """Serve main server dashboard UI."""
    return render_template('dashboard.html')


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


@app.route('/submit_privacy_summary', methods=['POST'])
def submit_privacy_summary():
    """Receive anonymization and patient-record privacy summaries from hospitals."""
    try:
        data = request.get_json() or {}
        hospital_id = data.get('hospital_id')
        if not hospital_id:
            return jsonify({'error': 'hospital_id required'}), 400

        with lock:
            privacy_reports[hospital_id] = {
                'hospital_id': hospital_id,
                'privacy_summary': data.get('privacy_summary', {}),
                'sample_records': data.get('sample_records', []),
                'hospital_profile': data.get('hospital_profile', {}),
                'timestamp': datetime.now().isoformat(),
            }

        return jsonify({
            'message': 'Privacy summary received',
            'hospital_id': hospital_id,
            'timestamp': datetime.now().isoformat(),
        }), 200
    except Exception as exc:
        logger.error(f"Error in submit_privacy_summary: {exc}")
        return jsonify({'error': str(exc)}), 500


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


@app.route('/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """Provide dashboard-friendly aggregated system state."""
    with lock:
        pending_updates = len(hospital_updates)
        hospital_snapshots = list(privacy_reports.values())

    return jsonify({
        'federated_status': {
            'current_round': current_round,
            'pending_hospital_updates': pending_updates,
            'total_hospitals_reporting': len(hospital_updates),
            'model_initialized': global_model is not None,
            'timestamp': datetime.now().isoformat(),
        },
        'training_summary': summarize_metrics(),
        'privacy_summary': summarize_privacy(),
        'hospital_privacy_reports': hospital_snapshots,
        'training_metrics': training_metrics[-12:],
    }), 200


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the federated learning process"""
    global current_round, hospital_updates, training_metrics, global_model, privacy_reports
    
    logger.info("Resetting federated learning process...")
    
    with lock:
        current_round = 0
        hospital_updates.clear()
        training_metrics.clear()
        privacy_reports.clear()
        global_model = None
    
    initialize_global_model()
    
    return jsonify({
        'message': 'System reset successfully',
        'timestamp': datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', str(MAIN_SERVER_PORT)))
    logger.info(f"Starting Main Server on {MAIN_SERVER_HOST}:{port}")
    initialize_global_model()
    app.run(host=MAIN_SERVER_HOST, port=port, debug=False)
