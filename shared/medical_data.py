"""
Shared Medical dataset helpers for server bootstrap and hospital training splits.
Supports MedMNIST (e.g., pathmnist) with a fallback to standard MNIST.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

import numpy as np

_MEDICAL_LOCK = threading.Lock()
_MEDICAL_CACHE = {
    "train_images": None,
    "train_labels": None,
    "test_images": None,
    "test_labels": None,
    "dataset_name": None,
    "input_shape": None,
    "num_classes": None,
}


def _normalize_images(images: np.ndarray, channels: int) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    if len(images.shape) == 3 and channels == 1:
        return images.reshape(-1, 28, 28, 1)
    elif len(images.shape) == 3 and channels == 3:
        # Unexpected, but handle just in case
        return images.reshape(-1, 28, 28, 3)
    return images


def ensure_medical_data_loaded() -> Dict[str, tuple]:
    """Load medical dataset once and keep it in memory for all processes."""
    with _MEDICAL_LOCK:
        if _MEDICAL_CACHE["train_images"] is None:
            import medmnist
            
            # Defaulting to pathmnist
            info = medmnist.INFO['pathmnist']
            DataClass = getattr(medmnist, info['python_class'])
            
            # Download and load locally so it can be committed to GitHub
            data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            os.makedirs(data_root, exist_ok=True)
            
            train_dataset = DataClass(split='train', download=True, root=data_root)
            test_dataset = DataClass(split='test', download=True, root=data_root)
            train_images = train_dataset.imgs
            train_labels = train_dataset.labels.squeeze()
            test_images = test_dataset.imgs
            test_labels = test_dataset.labels.squeeze()
            
            # Shuffle training data to ensure balanced class distribution across splits and bootstrap
            np.random.seed(42)
            indices = np.arange(len(train_images))
            np.random.shuffle(indices)
            train_images = train_images[indices]
            train_labels = train_labels[indices]
            
            channels = info['n_channels']
            _MEDICAL_CACHE["train_images"] = _normalize_images(train_images, channels)
            _MEDICAL_CACHE["train_labels"] = train_labels.astype(np.int64)
            _MEDICAL_CACHE["test_images"] = _normalize_images(test_images, channels)
            _MEDICAL_CACHE["test_labels"] = test_labels.astype(np.int64)
            
            _MEDICAL_CACHE["dataset_name"] = "MedMNIST (pathmnist)"
            _MEDICAL_CACHE["input_shape"] = (28, 28, channels)
            _MEDICAL_CACHE["num_classes"] = len(info['label'])

    return {
        "train": (_MEDICAL_CACHE["train_images"], _MEDICAL_CACHE["train_labels"]),
        "test": (_MEDICAL_CACHE["test_images"], _MEDICAL_CACHE["test_labels"]),
        "metadata": {
            "name": _MEDICAL_CACHE["dataset_name"],
            "input_shape": _MEDICAL_CACHE["input_shape"],
            "num_classes": _MEDICAL_CACHE["num_classes"],
        }
    }


def medical_dataset_status() -> Dict:
    """Return a lightweight readiness summary for dashboards."""
    data = ensure_medical_data_loaded()
    train_images, train_labels = data["train"]
    test_images, test_labels = data["test"]
    return {
        "ready": True,
        "dataset_name": data["metadata"]["name"],
        "train_samples": int(len(train_images)),
        "test_samples": int(len(test_images)),
        "num_classes": data["metadata"]["num_classes"],
        "image_shape": list(data["metadata"]["input_shape"]),
        "label_distribution": label_distribution(train_labels),
        "test_label_distribution": label_distribution(test_labels),
    }


def resolve_hospital_index(hospital_id: str | None, num_hospitals: int) -> int:
    """Map a hospital id to a stable split index."""
    if num_hospitals <= 0:
        return 0

    identifier = (hospital_id or "").strip().lower()
    digit_suffix = "".join(ch for ch in identifier if ch.isdigit())
    if digit_suffix:
        return (max(int(digit_suffix), 1) - 1) % num_hospitals

    if identifier:
        return sum(ord(ch) for ch in identifier) % num_hospitals

    return 0


def get_hospital_medical_split(
    hospital_id: str | None,
    num_hospitals: int,
    sample_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return a stable partition of the medical training set for a hospital."""
    data = ensure_medical_data_loaded()
    train_images, train_labels = data["train"]
    index = resolve_hospital_index(hospital_id, num_hospitals)

    partitions = np.array_split(np.arange(len(train_images)), num_hospitals)
    selected_indices = partitions[index]
    if sample_count is not None and sample_count > 0:
        selected_indices = selected_indices[:sample_count]

    images = train_images[selected_indices]
    labels = train_labels[selected_indices]
    summary = {
        "hospital_index": index,
        "hospital_id": hospital_id,
        "sample_count": int(len(images)),
        "label_distribution": label_distribution(labels),
        "source": data["metadata"]["name"],
    }
    return images, labels, summary


def label_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Return class counts for labels."""
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def bootstrap_model_on_medical_data(model, sample_count: int = 2048, epochs: int = 1, batch_size: int = 64) -> Dict:
    """Warm-start a model on a small data subset before the main server is exposed."""
    data = ensure_medical_data_loaded()
    train_images, train_labels = data["train"]
    
    actual_samples = min(sample_count, len(train_images))
    limited_images = train_images[:actual_samples]
    limited_labels = train_labels[:actual_samples]

    history = model.fit(
        limited_images,
        limited_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    final_loss = float(history.history["loss"][-1])
    final_accuracy = float(history.history["accuracy"][-1])
    return {
        "trained": True,
        "dataset": data["metadata"]["name"],
        "sample_count": int(len(limited_images)),
        "epochs": int(epochs),
        "accuracy": round(final_accuracy * 100, 2),
        "loss": round(final_loss, 4),
    }
