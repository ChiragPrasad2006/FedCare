"""
Shared MNIST dataset helpers for server bootstrap and hospital training splits.
"""
from __future__ import annotations

import threading
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.datasets import mnist


_MNIST_LOCK = threading.Lock()
_MNIST_CACHE = {
    "train_images": None,
    "train_labels": None,
    "test_images": None,
    "test_labels": None,
}


def _normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    return images.reshape(-1, 28, 28, 1)


def ensure_mnist_loaded() -> Dict[str, tuple]:
    """Load MNIST once and keep it in memory for all processes."""
    with _MNIST_LOCK:
        if _MNIST_CACHE["train_images"] is None:
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            _MNIST_CACHE["train_images"] = _normalize_images(train_images)
            _MNIST_CACHE["train_labels"] = train_labels.astype(np.int64)
            _MNIST_CACHE["test_images"] = _normalize_images(test_images)
            _MNIST_CACHE["test_labels"] = test_labels.astype(np.int64)

    return {
        "train": (_MNIST_CACHE["train_images"], _MNIST_CACHE["train_labels"]),
        "test": (_MNIST_CACHE["test_images"], _MNIST_CACHE["test_labels"]),
    }


def mnist_dataset_status() -> Dict:
    """Return a lightweight readiness summary for dashboards."""
    data = ensure_mnist_loaded()
    train_images, train_labels = data["train"]
    test_images, test_labels = data["test"]
    return {
        "ready": True,
        "train_samples": int(len(train_images)),
        "test_samples": int(len(test_images)),
        "num_classes": int(len(np.unique(train_labels))),
        "image_shape": list(train_images.shape[1:]),
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


def get_hospital_mnist_split(
    hospital_id: str | None,
    num_hospitals: int,
    sample_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return a stable partition of the MNIST training set for a hospital."""
    data = ensure_mnist_loaded()
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
        "source": "MNIST",
    }
    return images, labels, summary


def label_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Return class counts for digit labels."""
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def bootstrap_model_on_mnist(model, sample_count: int = 2048, epochs: int = 1, batch_size: int = 64) -> Dict:
    """Warm-start a model on a small MNIST subset before the main server is exposed."""
    data = ensure_mnist_loaded()
    train_images, train_labels = data["train"]
    limited_images = train_images[:sample_count]
    limited_labels = train_labels[:sample_count]

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
        "dataset": "MNIST",
        "sample_count": int(len(limited_images)),
        "epochs": int(epochs),
        "accuracy": round(final_accuracy * 100, 2),
        "loss": round(final_loss, 4),
    }
