"""
Neural network models for federated learning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_federated_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN model for federated learning"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_simple_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simpler model for faster training"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compile the model with optimizer and loss"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_model_weights(model):
    """Extract model weights"""
    return model.get_weights()


def set_model_weights(model, weights):
    """Set model weights"""
    model.set_weights(weights)
    return model


def average_weights(weights_list):
    """Average multiple weight arrays (for model aggregation)"""
    if not weights_list:
        return None
    
    averaged_weights = []
    num_models = len(weights_list)
    
    # Get number of layers
    num_layers = len(weights_list[0])
    
    for layer_idx in range(num_layers):
        layer_weights_sum = weights_list[0][layer_idx].copy().astype(float)
        
        for weights in weights_list[1:]:
            layer_weights_sum += weights[layer_idx].astype(float)
        
        averaged_weights.append(layer_weights_sum / num_models)
    
    return averaged_weights
