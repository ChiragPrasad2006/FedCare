"""
Main Server - hosts the global model, anonymizes uploaded records, and routes them.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    LEARNING_RATE,
    MAIN_SERVER_HOST,
    MAIN_SERVER_PORT,
    NUM_HOSPITALS,
    NUM_ROUNDS,
    PERSONALIZATION_ENABLED,
    PROXIMAL_MU,
    average_weights,
    compile_model,
    create_federated_model,
    get_model_weights,
    set_model_weights,
    setup_logger,
    HOSPITAL_REGISTRY_TTL_SECONDS,
)
from shared.mnist_data import bootstrap_model_on_mnist, mnist_dataset_status
from shared.privacy import analyze_and_anonymize_records


app = Flask(__name__)
CORS(app)
logger = setup_logger("main_server")

global_model = None
current_round = 0
hospital_updates = {}
training_metrics = []
personalization_metrics = []  # Track personalization results from all hospitals
processed_record_routes = {}
transfer_history = []
active_hospitals = {}
mnist_bootstrap = {
    "ready": False,
    "dataset": "MNIST",
    "sample_count": 0,
    "accuracy": 0.0,
    "loss": 0.0,
    "epochs": 0,
    "error": None,
}
lock = threading.Lock()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def initialize_global_model():
    """Initialize and warm-start the global model with MNIST."""
    global global_model, mnist_bootstrap
    logger.info("Initializing global model")
    global_model = create_federated_model()
    global_model = compile_model(global_model, learning_rate=LEARNING_RATE)
    try:
        mnist_bootstrap = bootstrap_model_on_mnist(global_model)
        mnist_bootstrap["ready"] = True
        mnist_bootstrap["timestamp"] = now_iso()
    except Exception as exc:
        mnist_bootstrap = {
            "ready": False,
            "dataset": "MNIST",
            "sample_count": 0,
            "accuracy": 0.0,
            "loss": 0.0,
            "epochs": 0,
            "error": str(exc),
            "timestamp": now_iso(),
        }
        logger.error("MNIST bootstrap failed: %s", exc)


def ensure_model_ready():
    if global_model is None:
        initialize_global_model()


def prune_inactive_hospitals():
    cutoff = datetime.utcnow() - timedelta(seconds=HOSPITAL_REGISTRY_TTL_SECONDS)
    inactive = [
        hospital_id
        for hospital_id, info in active_hospitals.items()
        if datetime.fromisoformat(info["last_seen"].replace("Z", "")) < cutoff
    ]
    for hospital_id in inactive:
        active_hospitals.pop(hospital_id, None)


def active_hospital_ids():
    prune_inactive_hospitals()
    return sorted(active_hospitals.keys())


def summarize_metrics():
    if not training_metrics:
        return {"latest_round_accuracy": 0.0, "latest_round_loss": 0.0}

    latest_round = max(item["round"] for item in training_metrics)
    latest_items = [item for item in training_metrics if item["round"] == latest_round]
    latest_round_accuracy = float(np.mean([item["metrics"].get("accuracy", 0.0) for item in latest_items]))
    latest_round_loss = float(np.mean([item["metrics"].get("loss", 0.0) for item in latest_items]))
    return {
        "latest_round_accuracy": round(latest_round_accuracy * 100, 2),
        "latest_round_loss": round(latest_round_loss, 4),
    }


def summarize_routes():
    queued_by_hospital = {
        hospital_id: sum(len(batch.get("records", [])) for batch in batches)
        for hospital_id, batches in processed_record_routes.items()
    }
    return {
        "total_records_waiting": sum(queued_by_hospital.values()),
        "queued_by_hospital": queued_by_hospital,
        "recent_transfers": transfer_history[-12:],
    }


def register_hospital(hospital_id: str, server_url: str | None = None):
    with lock:
        active_hospitals[hospital_id] = {
            "hospital_id": hospital_id,
            "server_url": server_url or "",
            "last_seen": now_iso(),
        }


def aggregate_weights(round_num):
    global global_model
    with lock:
        if not hospital_updates:
            return False

        weights_list = []
        from shared.communication import ServerCommunicator

        communicator = ServerCommunicator()
        for hospital_id, data in hospital_updates.items():
            if "weights" not in data:
                continue
            weights = communicator.receive_model_weights(data["weights"])
            if weights is not None:
                weights_list.append(weights)

        if not weights_list:
            return False

        averaged_weights = average_weights(weights_list)
        set_model_weights(global_model, averaged_weights)

        for hospital_id, data in hospital_updates.items():
            training_metrics.append(
                {
                    "round": round_num,
                    "hospital_id": hospital_id,
                    "metrics": data.get("metrics", {}),
                    "timestamp": now_iso(),
                }
            )

        hospital_updates.clear()
        return True


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
def health_check():
    ensure_model_ready()
    return jsonify(
        {
            "status": "healthy",
            "timestamp": now_iso(),
            "mnist_ready": mnist_bootstrap["ready"],
            "current_round": current_round,
        }
    ), 200


@app.route("/register_hospital", methods=["POST"])
def register_hospital_endpoint():
    data = request.get_json() or {}
    hospital_id = (data.get("hospital_id") or "").strip()
    if not hospital_id:
        return jsonify({"error": "hospital_id required"}), 400

    register_hospital(hospital_id, data.get("server_url"))
    return jsonify({"message": "Hospital registered", "hospital_id": hospital_id, "timestamp": now_iso()}), 200


@app.route("/hospital_directory", methods=["GET"])
def hospital_directory():
    exclude = (request.args.get("exclude") or "").strip()
    hospitals = [hospital_id for hospital_id in active_hospital_ids() if hospital_id != exclude]
    return jsonify({"hospitals": hospitals, "timestamp": now_iso()}), 200


@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    ensure_model_ready()
    try:
        import base64
        import pickle

        weights = get_model_weights(global_model)
        weights_b64 = base64.b64encode(pickle.dumps(weights)).decode("utf-8")
        return jsonify({"weights": weights_b64, "round": current_round, "timestamp": now_iso()}), 200
    except Exception as exc:
        logger.error("Error retrieving global model: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/submit_update", methods=["POST"])
def submit_update():
    data = request.get_json() or {}
    hospital_id = (data.get("hospital_id") or "").strip()
    if not hospital_id:
        return jsonify({"error": "hospital_id required"}), 400

    register_hospital(hospital_id, data.get("server_url"))
    with lock:
        hospital_updates[hospital_id] = {
            "weights": data.get("weights"),
            "metrics": data.get("metrics", {}),
            "timestamp": now_iso(),
        }

    return jsonify({"message": "Update received", "hospital_id": hospital_id, "timestamp": now_iso()}), 200


@app.route("/aggregate", methods=["POST"])
def aggregate():
    global current_round
    ensure_model_ready()
    success = aggregate_weights(current_round)
    if success:
        current_round += 1
    return jsonify({"message": "Aggregation completed", "success": success, "round": current_round}), 200


@app.route("/trigger_personalization", methods=["POST"])
def trigger_personalization():
    """
    Trigger personalization phase on all active hospitals after aggregation.
    Optional: use FedProx or standard fine-tuning based on PERSONALIZATION_ENABLED.
    
    Expected JSON payload:
    {
        "round": <round_number>,
        "use_fedprox": <true/false>,  # defaults to PERSONALIZATION_ENABLED
        "hospitals": [<list of hospital IDs>]  # optional, defaults to all active
    }
    """
    try:
        data = request.get_json() or {}
        round_num = data.get("round", current_round)
        use_fedprox = data.get("use_fedprox", PERSONALIZATION_ENABLED)
        hospital_ids = data.get("hospitals", active_hospital_ids())
        
        if not hospital_ids:
            return jsonify({"error": "No active hospitals available"}), 400
        
        results = {}
        for hospital_id in hospital_ids:
            if hospital_id not in active_hospitals:
                results[hospital_id] = {"status": "inactive"}
                continue
            
            try:
                hospital_url = active_hospitals[hospital_id].get("server_url", "")
                if not hospital_url:
                    results[hospital_id] = {"status": "no_url"}
                    continue
                
                # Determine endpoint based on FedProx flag
                endpoint = "/personalize_fedprox" if use_fedprox else "/personalize"
                
                response = requests.post(
                    f"{hospital_url}{endpoint}",
                    json={"round": round_num, "use_fedprox": use_fedprox},
                    timeout=60
                )
                
                if response.status_code == 200:
                    personalization_metrics.append({
                        "round": round_num,
                        "hospital_id": hospital_id,
                        "metrics": response.json().get("metrics", {}),
                        "timestamp": now_iso()
                    })
                    results[hospital_id] = {
                        "status": "completed",
                        "metrics": response.json().get("metrics", {})
                    }
                else:
                    results[hospital_id] = {
                        "status": "failed",
                        "reason": response.json().get("error", "unknown")
                    }
            except Exception as exc:
                logger.error(f"Error triggering personalization for {hospital_id}: {exc}")
                results[hospital_id] = {"status": "error", "reason": str(exc)}
        
        return jsonify({
            "message": "Personalization triggered",
            "round": round_num,
            "personalization_type": "fedprox" if use_fedprox else "standard",
            "results": results,
            "timestamp": now_iso()
        }), 200
        
    except Exception as exc:
        logger.error("Error in personalization trigger: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/personalization_metrics", methods=["GET"])
def get_personalization_metrics():
    """Retrieve aggregated personalization metrics from all hospitals"""
    limit = request.args.get("limit", 20, type=int)
    round_num = request.args.get("round", None, type=int)
    
    if round_num is not None:
        metrics = [m for m in personalization_metrics if m["round"] == round_num]
    else:
        metrics = personalization_metrics[-limit:]
    
    return jsonify({
        "personalization_metrics": metrics,
        "total_personalization_rounds": len(set(m["round"] for m in personalization_metrics)),
        "timestamp": now_iso()
    }), 200


@app.route("/submit_patient_records", methods=["POST"])
def submit_patient_records():
    """Receive raw hospital records, anonymize them, then route to the chosen hospital."""
    data = request.get_json() or {}
    source_hospital_id = (data.get("source_hospital_id") or "").strip()
    destination_hospital_id = (data.get("destination_hospital_id") or "").strip()
    records = data.get("records", [])

    if not source_hospital_id or not destination_hospital_id:
        return jsonify({"error": "source_hospital_id and destination_hospital_id are required"}), 400
    if source_hospital_id == destination_hospital_id:
        return jsonify({"error": "source and destination hospitals must be different"}), 400
    if destination_hospital_id not in active_hospital_ids():
        return jsonify({"error": "destination hospital is not currently active"}), 400
    if not isinstance(records, list) or not records:
        return jsonify({"error": "records must be a non-empty list"}), 400

    anonymized_records, privacy_summary = analyze_and_anonymize_records(records)
    batch = {
        "source_hospital_id": source_hospital_id,
        "destination_hospital_id": destination_hospital_id,
        "records": anonymized_records,
        "privacy_summary": privacy_summary,
        "timestamp": now_iso(),
    }

    with lock:
        processed_record_routes.setdefault(destination_hospital_id, []).append(batch)
        transfer_history.append(
            {
                "source_hospital_id": source_hospital_id,
                "destination_hospital_id": destination_hospital_id,
                "record_count": len(anonymized_records),
                "removal_accuracy": privacy_summary.get("removal_accuracy", 0.0),
                "timestamp": batch["timestamp"],
            }
        )

    return jsonify(
        {
            "message": "Records anonymized by main server and routed",
            "destination_hospital_id": destination_hospital_id,
            "record_count": len(anonymized_records),
            "privacy_summary": privacy_summary,
            "preview": anonymized_records[:5],
            "timestamp": batch["timestamp"],
        }
    ), 200


@app.route("/processed_records/<hospital_id>", methods=["GET"])
def get_processed_records(hospital_id):
    consume = request.args.get("consume", "false").lower() == "true"
    with lock:
        batches = list(processed_record_routes.get(hospital_id, []))
        if consume:
            processed_record_routes[hospital_id] = []

    return jsonify(
        {
            "hospital_id": hospital_id,
            "batch_count": len(batches),
            "record_count": sum(len(batch.get("records", [])) for batch in batches),
            "batches": batches,
            "timestamp": now_iso(),
        }
    ), 200


@app.route("/dashboard_data", methods=["GET"])
def get_dashboard_data():
    ensure_model_ready()
    return jsonify(
        {
            "mnist_bootstrap": mnist_bootstrap,
            "mnist_dataset": mnist_dataset_status(),
            "active_hospitals": [active_hospitals[hospital_id] for hospital_id in active_hospital_ids()],
            "route_summary": summarize_routes(),
            "training_summary": summarize_metrics(),
            "current_round": current_round,
            "num_rounds_configured": NUM_ROUNDS,
            "timestamp": now_iso(),
        }
    ), 200


@app.route("/reset", methods=["POST"])
def reset():
    global current_round, global_model
    with lock:
        current_round = 0
        hospital_updates.clear()
        training_metrics.clear()
        processed_record_routes.clear()
        transfer_history.clear()
        global_model = None
    initialize_global_model()
    return jsonify({"message": "System reset successfully", "timestamp": now_iso()}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", str(MAIN_SERVER_PORT)))
    logger.info("Starting Main Server on %s:%s", MAIN_SERVER_HOST, port)
    initialize_global_model()
    app.run(host=MAIN_SERVER_HOST, port=port, debug=False)
