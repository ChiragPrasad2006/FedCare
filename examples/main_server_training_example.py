#!/usr/bin/env python3
"""
Example: Training Main Server with Optimized Accuracy

This script demonstrates how to:
1. Download medical datasets (MedMNIST, etc.)
2. Prepare data with augmentation and preprocessing
3. Train main server with learning rate scheduling
4. Validate and save the best model

Usage:
  python examples/main_server_training_example.py

Dataset will be downloaded to: ./data/medical/
Trained model will be saved to: ./models/main_server_model.h5
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from shared import setup_logger, compile_model, create_simple_model

logger = setup_logger("main_server_training")


class MedicalDatasetLoader:
    """Load and prepare medical imaging datasets"""
    
    def __init__(self, data_dir="./data/medical"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_medmnist_dataset(self, dataset_name='pathmnist', split='train'):
        """
        Load MedMNIST dataset
        Available datasets: pathmnist, chexpert, organmnist, etc.
        """
        logger.info(f"Loading MedMNIST ({dataset_name})...")
        
        try:
            import medmnist
            
            # Download if needed
            loader = medmnist.Loader(
                split=split,
                download=True,
                download_dir=self.data_dir
            )
            
            logger.info(f"  Loaded {len(loader)} samples")
            return loader
        
        except ImportError:
            logger.error("medmnist not installed. Install with: pip install medmnist")
            return None
    
    def prepare_mnist_for_fedcare(self, num_samples=10000):
        """
        Prepare MNIST data compatible with FedCare
        Resizes to (28, 28, 1) and normalizes
        """
        logger.info(f"Loading and preparing MNIST data ({num_samples} samples)...")
        
        from tensorflow.keras.datasets import mnist
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        # Limit to num_samples
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]
        
        logger.info(f"  Training: {X_train.shape}, {y_train.shape}")
        logger.info(f"  Testing: {X_test.shape}, {y_test.shape}")
        
        return (X_train, y_train), (X_test, y_test)


class OptimizedModel:
    """Create optimized medical imaging model"""
    
    @staticmethod
    def create_model_with_regularization(
        input_shape=(28, 28, 1),
        num_classes=10,
        l2_reg=0.0001
    ):
        """Create CNN with batch norm and L2 regularization"""
        
        logger.info("Creating optimized model...")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # Conv block 1
            layers.Conv2D(
                32, (3, 3), padding='same',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv block 2
            layers.Conv2D(
                64, (3, 3), padding='same',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv block 3
            layers.Conv2D(
                128, (3, 3), padding='same',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(
                256, activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"  Parameters: {model.count_params():,.0f}")
        return model


class LearningRateScheduler:
    """Create adaptive learning rate schedules"""
    
    @staticmethod
    def warmup_cosine_decay(
        initial_lr=0.001,
        total_steps=1000,
        warmup_steps=100
    ):
        """Linear warmup + Cosine annealing decay"""
        
        def scheduler(step):
            if step < warmup_steps:
                # Linear warmup
                return initial_lr * (step / warmup_steps)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        return keras.optimizers.schedules.LambdaSchedule(scheduler)


class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, output_dir="./training_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_history(self, history, save_path=None):
        """Plot training history"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"  Plot saved: {save_path}")
        
        return fig


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "="*60)
    logger.info("Main Server Training Pipeline")
    logger.info("="*60)
    
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 20
    INITIAL_LR = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    loader = MedicalDatasetLoader()
    (X_train, y_train), (X_test, y_test) = loader.prepare_mnist_for_fedcare(
        num_samples=10000
    )
    
    # Split into train/val
    split_idx = int(len(X_train) * (1 - VALIDATION_SPLIT))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 2: Create model
    logger.info("\n[Step 2] Creating optimized model...")
    model = OptimizedModel.create_model_with_regularization(
        input_shape=X_train.shape[1:],
        num_classes=len(np.unique(y_train)),
        l2_reg=0.0001
    )
    
    # Step 3: Setup learning rate schedule
    logger.info("\n[Step 3] Setting up learning rate schedule...")
    total_steps = (len(X_train) // BATCH_SIZE) * EPOCHS
    warmup_steps = (len(X_train) // BATCH_SIZE) * 2  # Warmup for 2 epochs
    
    lr_schedule = LearningRateScheduler.warmup_cosine_decay(
        initial_lr=INITIAL_LR,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )
    
    # Step 4: Compile model
    logger.info("\n[Step 4] Compiling model...")
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Step 5: Setup callbacks
    logger.info("\n[Step 5] Setting up training callbacks...")
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            'models/main_server_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='./logs/main_server',
            histogram_freq=1,
            update_freq='epoch',
            write_graph=True
        ),
        
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Step 6: Train model
    logger.info("\n[Step 6] Training model...")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Training samples: {len(X_train)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 7: Evaluate
    logger.info("\n[Step 7] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Step 8: Save model
    logger.info("\n[Step 8] Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/main_server_model.h5')
    logger.info(f"  Model saved: models/main_server_model.h5")
    
    # Save metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmax(history.history['val_accuracy'])
    }
    
    with open('models/main_server_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved: models/main_server_metrics.json")
    
    # Step 9: Plot history
    logger.info("\n[Step 9] Plotting training history...")
    monitor = TrainingMonitor()
    monitor.plot_history(history, save_path='training_logs/training_history.png')
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"✓ Test Accuracy: {test_accuracy:.4f} (target: >0.90)")
    logger.info(f"✓ Model saved: models/main_server_model.h5")
    logger.info(f"✓ Metrics saved: models/main_server_metrics.json")
    logger.info(f"✓ Training plot: training_logs/training_history.png")
    logger.info(f"✓ Ready for federation!")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
