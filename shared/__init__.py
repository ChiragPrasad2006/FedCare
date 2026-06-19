"""
Shared utilities for FedCare
"""
from .config import *
from .communication import ServerCommunicator
from .models import (
    create_federated_model,
    create_simple_model,
    compile_model,
    get_model_weights,
    set_model_weights,
    average_weights,
    create_fedprox_model,
    compute_proximal_term,
    train_with_fedprox,
    personalize_model
)
from .logger import setup_logger

__all__ = [
    'ServerCommunicator',
    'create_federated_model',
    'create_simple_model',
    'compile_model',
    'get_model_weights',
    'set_model_weights',
    'average_weights',
    'create_fedprox_model',
    'compute_proximal_term',
    'train_with_fedprox',
    'personalize_model',
    'setup_logger'
]
