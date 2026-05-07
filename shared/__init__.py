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
    average_weights
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
    'setup_logger'
]
