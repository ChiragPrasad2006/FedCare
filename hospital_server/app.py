"""
Hospital Server - local MNIST training plus simple upload-and-route workflow.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    BATCH_SIZE,
    EPOCHS_PER_ROUND,
    HOSPITAL_ID,
    HOSPITAL_SERVER_PORT,
    LEARNING_RATE,
    MAIN_SERVER_HOST,
    MAIN_SERVER_PORT,
    MAIN_SERVER_URL,
    NUM_HOSPITALS,
    compile_model,
    create_federated_model,
    get_model_weights,
    set_model_weights,
    setup_logger,
)
from shared.mnist_data import get_hospital_mnist_split


app = Flask(__name__)
CORS(app)
logger = setup_logger("hospital_server")

hospital_id = HOSPITAL_ID or "hospital_1"
local_model = None
current_round = 0
training_history = []
local_data = None
local_data_summary = {}
last_submission = None
received_processed_batches = []


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def get_main_server_base_url():
    if MAIN_SERVER_URL:
        return MAIN_SERVER_URL.rstrip("/")
    return f"http://{MAIN_SERVER_HOST}:{MAIN_SERVER_PORT}"


def initialize_local_model():
    global local_model
    local_model = create_federated_model()
    local_model = compile_model(local_model, learning_rate=LEARNING_RATE)


def load_hospital_mnist_data(sample_count: int = 1200):
    global local_data, local_data_summary
    images, labels, summary = get_hospital_mnist_split(
        hospital_id=hospital_id,
        num_hospitals=NUM_HOSPITALS,
        sample_count=sample_count,
    )
    local_data = (images, labels)
    local_data_summary = summary


def register_with_main_server():
    try:
        requests.post(
            f"{get_main_server_base_url()}/register_hospital",
            json={
                "hospital_id": hospital_id,
                "server_url": f"http://{hospital_id}:{HOSPITAL_SERVER_PORT}",
            },
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Unable to register hospital with main server: %s", exc)


def bootstrap_hospital():
    initialize_local_model()
    load_hospital_mnist_data()
    register_with_main_server()


def fetch_global_model():
    try:
        response = requests.get(f"{get_main_server_base_url()}/get_global_model", timeout=30)
        if response.status_code != 200:
            return False

        from shared.communication import ServerCommunicator

        payload = response.json()
        weights = ServerCommunicator().receive_model_weights(payload["weights"])
        if weights is None:
            return False
        set_model_weights(local_model, weights)
        return True
    except Exception as exc:
        logger.error("Error fetching global model: %s", exc)
        return False


def train_on_local_data():
    global training_history
    if local_model is None or local_data is None:
        return None

    images, labels = local_data
    history = local_model.fit(
        images,
        labels,
        epochs=EPOCHS_PER_ROUND,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    metrics = {
        "loss": float(history.history["loss"][-1]),
        "accuracy": float(history.history["accuracy"][-1]),
        "epochs": EPOCHS_PER_ROUND,
        "samples": len(images),
    }
    training_history.append({"round": current_round, "metrics": metrics, "timestamp": now_iso()})
    return metrics


def submit_update():
    try:
        import base64
        import pickle

        weights_b64 = base64.b64encode(pickle.dumps(get_model_weights(local_model))).decode("utf-8")
        payload = {
            "hospital_id": hospital_id,
            "weights": weights_b64,
            "metrics": training_history[-1]["metrics"] if training_history else {},
            "round": current_round,
        }
        response = requests.post(f"{get_main_server_base_url()}/submit_update", json=payload, timeout=30)
        return response.status_code == 200
    except Exception as exc:
        logger.error("Error submitting update: %s", exc)
        return False


def sync_processed_inbox():
    global received_processed_batches
    try:
        response = requests.get(
            f"{get_main_server_base_url()}/processed_records/{hospital_id}?consume=true",
            timeout=15,
        )
        if response.status_code != 200:
            return {"batch_count": 0, "record_count": 0, "batches": []}
        payload = response.json()
        batches = payload.get("batches", [])
        if batches:
            received_processed_batches.extend(batches)
        return payload
    except Exception as exc:
        logger.warning("Unable to sync processed inbox: %s", exc)
        return {"batch_count": 0, "record_count": 0, "batches": []}


def fetch_active_hospitals():
    register_with_main_server()
    try:
        response = requests.get(
            f"{get_main_server_base_url()}/hospital_directory?exclude={hospital_id}",
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("hospitals", [])
    except Exception as exc:
        logger.warning("Unable to fetch active hospitals: %s", exc)
    return []


def parse_uploaded_records():
    if request.files.get("records_file"):
        upload = request.files["records_file"]
        content = upload.read().decode("utf-8")
        if upload.filename.lower().endswith(".json"):
            payload = json.loads(content)
            return payload["records"] if isinstance(payload, dict) and "records" in payload else payload
        if upload.filename.lower().endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content))
            return list(reader)
        raise ValueError("Only .json and .csv files are supported")

    data = request.get_json() or {}
    records = data.get("records", data if isinstance(data, list) else [])
    return records


@app.route("/", methods=["GET"])
def dashboard():
    return render_template(
        "dashboard.html",
        default_hospital_id=hospital_id,
        main_server_url=get_main_server_base_url(),
    )


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "hospital_id": hospital_id,
            "mnist_loaded": local_data_summary.get("sample_count", 0) > 0,
            "timestamp": now_iso(),
        }
    ), 200


@app.route("/configure", methods=["POST"])
def configure():
    global hospital_id, current_round, training_history, last_submission, received_processed_batches
    data = request.get_json() or {}
    hospital_id = (data.get("hospital_id") or hospital_id or "hospital_1").strip()
    current_round = 0
    training_history.clear()
    last_submission = None
    received_processed_batches = []
    bootstrap_hospital()
    return jsonify({"message": "Hospital configured", "hospital_id": hospital_id, "timestamp": now_iso()}), 200


@app.route("/upload_patient_records", methods=["POST"])
def upload_patient_records():
    global last_submission
    try:
        destination_hospital_id = (
            request.form.get("destination_hospital_id")
            if request.files
            else (request.get_json() or {}).get("destination_hospital_id", "")
        ).strip()
        records = parse_uploaded_records()

        if not destination_hospital_id:
            return jsonify({"error": "destination_hospital_id is required"}), 400
        if not isinstance(records, list) or not records:
            return jsonify({"error": "Provide a non-empty JSON or CSV upload"}), 400

        register_with_main_server()
        response = requests.post(
            f"{get_main_server_base_url()}/submit_patient_records",
            json={
                "source_hospital_id": hospital_id,
                "destination_hospital_id": destination_hospital_id,
                "records": records,
            },
            timeout=30,
        )
        payload = response.json()
        if response.status_code != 200:
            return jsonify({"error": payload.get("error", "Upload failed")}), response.status_code

        last_submission = {
            "destination_hospital_id": destination_hospital_id,
            "record_count": payload.get("record_count", 0),
            "privacy_summary": payload.get("privacy_summary", {}),
            "preview": payload.get("preview", []),
            "timestamp": payload.get("timestamp", now_iso()),
        }
        return jsonify({"message": payload["message"], "submission": last_submission}), 200
    except Exception as exc:
        logger.error("Error uploading patient records: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/sync_and_train", methods=["POST"])
def sync_and_train():
    global current_round
    if not fetch_global_model():
        return jsonify({"error": "Failed to fetch global model"}), 500
    metrics = train_on_local_data()
    if metrics is None:
        return jsonify({"error": "Training failed"}), 500
    if not submit_update():
        return jsonify({"error": "Failed to submit update"}), 500
    current_round += 1
    return jsonify({"message": "Training completed", "metrics": metrics, "round": current_round}), 200


@app.route("/dashboard_data", methods=["GET"])
def get_dashboard_data():
    sync_payload = sync_processed_inbox()
    active_hospitals = fetch_active_hospitals()
    return jsonify(
        {
            "hospital_id": hospital_id,
            "mnist": {
                "loaded": local_data_summary.get("sample_count", 0) > 0,
                "sample_count": local_data_summary.get("sample_count", 0),
                "label_distribution": local_data_summary.get("label_distribution", {}),
            },
            "active_hospitals": active_hospitals,
            "last_submission": last_submission,
            "received": {
                "new_batch_count": sync_payload.get("batch_count", 0),
                "new_record_count": sync_payload.get("record_count", 0),
                "history": received_processed_batches[-12:],
            },
            "training": training_history[-6:],
            "main_server_url": get_main_server_base_url(),
            "timestamp": now_iso(),
        }
    ), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", str(HOSPITAL_SERVER_PORT)))
    logger.info("Starting Hospital Server on 0.0.0.0:%s", port)
    bootstrap_hospital()
    app.run(host="0.0.0.0", port=port, debug=False)
