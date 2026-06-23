"""
Neural network models for federated learning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_federated_model(input_shape=(28, 28, 3), num_classes=10, l2_reg=0.0001):
    """
    Create an optimized CNN model for medical images (MedMNIST).
    Uses Batch Normalization, Dropout, and L2 regularization for robust federated training.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_simple_model(input_shape=(28, 28, 3), num_classes=10):
    """Alias for backwards compatibility if needed, but uses the same proper model."""
    return create_federated_model(input_shape, num_classes)


class LearningRateScheduler:
    """Create adaptive learning rate schedules for medical training"""
    @staticmethod
    def warmup_cosine_decay(initial_lr=0.001, total_steps=1000, warmup_steps=100):
        import numpy as np
        def scheduler(step):
            if step < warmup_steps:
                return initial_lr * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        return keras.optimizers.schedules.LambdaSchedule(scheduler)


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
<<<<<<< Updated upstream
=======


# ============================================================================
# FedProx Personalization Support
# ============================================================================

def create_fedprox_model(input_shape=(28, 28, 3), num_classes=10):
    """Create model compatible with FedProx (same as simple_model for now)"""
    return create_simple_model(input_shape=input_shape, num_classes=num_classes)


def compute_proximal_term(current_weights, global_weights):
    """
    Compute FedProx proximal regularization term: ||w - w_t||^2
    This encourages local models to stay close to global model.
    
    Args:
        current_weights: list of current model weight arrays
        global_weights: list of global model weight arrays
    
    Returns:
        float: proximal loss term (sum of squared differences)
    """
    if current_weights is None or global_weights is None:
        return 0.0
    
    proximal_loss = 0.0
    for current, global_w in zip(current_weights, global_weights):
        proximal_loss += tf.reduce_sum(tf.square(current - global_w)).numpy()
    
    return float(proximal_loss)


def train_with_fedprox(
    model,
    X_train,
    y_train,
    global_weights,
    epochs=3,
    batch_size=32,
    learning_rate=0.001,
    proximal_mu=0.01,
    verbose=0
):
    """
    Train model with FedProx regularization.
    Adds proximal term to prevent drift from global model.
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        global_weights: Global model weights to stay close to
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        proximal_mu: Coefficient for proximal term (higher = stronger regularization)
        verbose: Verbosity level
    
    Returns:
        dict: Training metrics {loss, accuracy, proximal_term}
    """
    import numpy as np
    
    if global_weights is None:
        # If no global model, train normally
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        return {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'proximal_term': 0.0,
            'is_personalized': False
        }
    
    # Store initial global weights
    initial_global = [w.copy() for w in global_weights]
    
    # Create custom training loop with proximal regularization
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    train_losses = []
    train_accuracies = []
    proximal_terms = []
    
    # Convert to TensorFlow dataset for batching
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=len(X_train))
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_proximal = []
        
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # Standard loss
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
                
                # Add FedProx proximal term
                current_weights = model.trainable_weights
                proximal_loss = 0.0
                
                for curr_w, glob_w in zip(current_weights, initial_global):
                    proximal_loss += tf.reduce_sum(
                        tf.square(curr_w - glob_w)
                    )
                
                proximal_loss = proximal_mu * proximal_loss
                
                # Total loss = task loss + proximal regularization
                total_loss = loss + proximal_loss
            
            # Update weights
            gradients = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            
            # Track metrics
            epoch_losses.append(float(loss.numpy()))
            epoch_accuracies.append(
                float(keras.metrics.SparseCategoricalAccuracy()(y_batch, predictions).numpy())
            )
            epoch_proximal.append(float(proximal_loss.numpy()))
        
        train_losses.append(np.mean(epoch_losses))
        train_accuracies.append(np.mean(epoch_accuracies))
        proximal_terms.append(np.mean(epoch_proximal))
        
        if verbose > 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {train_losses[-1]:.4f}, "
                f"Accuracy: {train_accuracies[-1]:.4f}, "
                f"Proximal: {proximal_terms[-1]:.6f}"
            )
    
    return {
        'loss': float(train_losses[-1]),
        'accuracy': float(train_accuracies[-1]),
        'proximal_term': float(proximal_terms[-1]),
        'is_personalized': True
    }


def personalize_model(
    model,
    X_train,
    y_train,
    global_weights,
    epochs=3,
    batch_size=32,
    learning_rate=0.0005,  # Lower LR for personalization
    verbose=0
):
    """
    Fine-tune model locally after receiving global model.
    Uses lower learning rate and fewer epochs.
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        global_weights: Global model weights as starting point
        epochs: Fine-tuning epochs
        batch_size: Batch size
        learning_rate: Fine-tuning learning rate (typically lower)
        verbose: Verbosity level
    
    Returns:
        dict: Fine-tuning metrics
    """
    # First set to global weights
    set_model_weights(model, global_weights)
    
    # Then fine-tune locally
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return {
        'loss': float(history.history['loss'][-1]),
        'accuracy': float(history.history['accuracy'][-1]),
        'epochs': epochs,
        'personalization_type': 'standard_finetune'
    }
>>>>>>> Stashed changes
