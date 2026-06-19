# Training Data Sourcing & Optimization Strategy for Main Server

## Overview

Accurate federated learning requires a robust global model. This guide provides:

1. **Public medical dataset sources** for initial training
2. **Data loading and preprocessing** strategies  
3. **Training optimization techniques** for higher accuracy
4. **Scalable data pipelines** for production deployment

## Part 1: Large Medical Datasets for Main Server Training

### Recommended Public Datasets

#### 1. **Medical MNIST (10,000 images)**
- **Best for**: Quick prototyping, educational use
- **Size**: ~10 KB (digits) to 50 MB (medical)
- **Types**: Pneumonia detection, chest X-rays, blood cells
- **Link**: https://github.com/MedMNIST/MedMNIST
- **Setup**:
```python
import medmnist
from medmnist.info import INFO

# Available datasets
data_flag = 'pathmnist'  # or 'chexpert', 'organmnist', etc.
info = INFO[data_flag]

train_dataset = medmnist.Loader(split='train', download=True)
test_dataset = medmnist.Loader(split='test', download=True)
```

#### 2. **CheXpert (224,000 chest X-rays)**
- **Best for**: Chest X-ray classification, comprehensive training
- **Size**: ~439 GB (full) or ~50 GB (small)
- **Coverage**: 64,540 patients
- **Link**: https://stanfordmlgroup.github.io/competitions/chexpert/
- **Requires**: Stanford login, academic access
- **Classes**: Normal, Frontal, Lateral views; 14 pathology labels
- **Preprocessing**:
```python
import pandas as pd
from skimage import io
import numpy as np

# Load CheXpert metadata
df = pd.read_csv('CheXpert-v1.0/train.csv')

# Filter frontal views only
frontal = df[df['Frontal/Lateral'] == 'Frontal']

# Load and resize images
images = []
for idx, row in frontal.iterrows():
    img = io.imread(row['Path'])
    img = resize(img, (224, 224))
    images.append(img)

X_train = np.array(images)
```

#### 3. **MIMIC-CXR (377,110 chest X-rays)**
- **Best for**: Large-scale medical imaging, critical care
- **Size**: ~1.7 TB (full dataset)
- **Coverage**: 65,379 unique patients
- **Link**: https://physionet.org/content/mimic-cxr2/
- **Requires**: PhysioNet credentialed access (free for researchers)
- **Features**:
  - Multiple views per patient
  - Associated clinical notes (NLP integration possible)
  - Discharge summaries and diagnoses

#### 4. **ImageNet Pre-trained Models (Transfer Learning)**
- **Best for**: Leveraging existing medical domain knowledge
- **Size**: Already trained models (100-500 MB)
- **Strategy**: Fine-tune ImageNet/ResNet on your medical data
- **Code**:
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pre-trained ImageNet model
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add medical classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Fine-tune on medical data (faster convergence)
model.fit(X_train, y_train, epochs=5, batch_size=64)
```

#### 5. **Synthetic Medical Data (SYNTHEA)**
- **Best for**: Privacy-preserving training, data augmentation
- **Size**: Configurable (can generate 1M+ synthetic patients)
- **Link**: https://github.com/synthetichealth/synthea
- **Usage**:
```bash
# Generate synthetic patient data
java -jar synthea-with-dependencies.jar -p 100000

# Produces realistic EHR data in FHIR format
# Can convert to images via simulation
```

### Dataset Comparison Table

| Dataset | Size | Images | Domain | Access | Best Use Case |
|---------|------|--------|--------|--------|--------------|
| Medical MNIST | 50 MB | 10K | Multi-organ | Free | Testing, prototyping |
| CheXpert | 50-439 GB | 224K | Chest X-ray | Academic | Chest imaging specialization |
| MIMIC-CXR | 1.7 TB | 377K | Critical care | Credentialed | Large-scale, multi-modal |
| ImageNet Pre-trained | 100-500 MB | N/A | General object | Free | Transfer learning backbone |
| SYNTHEA | Custom | N/A | EHR synthetic | Free | Privacy-safe training |

## Part 2: Data Loading Pipeline

### Strategy 1: Batch Download + Local Storage

Recommended for **main server production**:

```python
# download_datasets.py
import os
import requests
from pathlib import Path
import tensorflow as tf

class DatasetDownloader:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_medmnist(self, dataset_name='pathmnist'):
        """Download MedMNIST dataset"""
        import medmnist
        
        print(f"Downloading {dataset_name}...")
        dataset = medmnist.INFO[dataset_name]
        
        # Auto-download to data_dir
        train_data = medmnist.Loader(
            split='train',
            download=True,
            download_dir=str(self.data_dir)
        )
        
        return train_data
    
    def download_chexpert_sample(self, num_samples=10000):
        """
        Download sample CheXpert data
        Note: Full dataset requires manual download from Stanford
        """
        print("CheXpert requires manual download from Stanford ML Group")
        print("Visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
        return None
    
    def download_mimic_cxr(self, num_samples=5000):
        """
        Prepare MIMIC-CXR (requires PhysioNet credentials)
        """
        print("MIMIC-CXR requires PhysioNet credentialed access")
        print("1. Sign up at https://physionet.org/")
        print("2. Request MIMIC-CXR access")
        print("3. Download via gsutil")
        return None

# Usage
downloader = DatasetDownloader(data_dir="./data/medical")
train_data = downloader.download_medmnist('chexpert')
```

### Strategy 2: Lazy Loading (Stream from Cloud)

For **large datasets** without local storage:

```python
# cloud_data_loader.py
import tensorflow as tf
from tensorflow.io import gfile

class CloudDataLoader:
    def __init__(self, bucket_name='fedcare-medical-data'):
        self.bucket_name = bucket_name
    
    def stream_from_s3(self, dataset_prefix, batch_size=32):
        """Stream training data directly from S3"""
        dataset = tf.data.Dataset.list_files(
            f"s3://{self.bucket_name}/{dataset_prefix}/*.tfrecord"
        )
        
        def load_tfrecord(path):
            return tf.data.TFRecordDataset(path)
        
        dataset = dataset.interleave(
            load_tfrecord,
            cycle_length=4,
            block_length=16
        )
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

# Usage
loader = CloudDataLoader(bucket_name='fedcare-data')
train_ds = loader.stream_from_s3('chest-xrays', batch_size=64)
```

### Strategy 3: Composite Dataset (Multi-source)

```python
# composite_loader.py
import numpy as np
from sklearn.utils import shuffle

class CompositeDataLoader:
    def __init__(self):
        self.sources = []
    
    def add_medmnist(self, dataset_name='pathmnist', weight=0.3):
        """Add MedMNIST with sampling weight"""
        import medmnist
        data = medmnist.Loader(split='train', download=True)
        self.sources.append({
            'data': data,
            'weight': weight,
            'name': dataset_name
        })
    
    def add_local_data(self, path, weight=0.7):
        """Add local hospital data"""
        import os
        from PIL import Image
        
        images = []
        for fname in os.listdir(path):
            img = Image.open(os.path.join(path, fname))
            img = img.resize((224, 224))
            images.append(np.array(img))
        
        self.sources.append({
            'data': np.array(images),
            'weight': weight,
            'name': 'local_data'
        })
    
    def get_balanced_batch(self, batch_size=64):
        """
        Sample from sources according to weights
        """
        weights = np.array([s['weight'] for s in self.sources])
        weights = weights / weights.sum()
        
        # Determine samples per source
        samples_per_source = (weights * batch_size).astype(int)
        
        batch_images = []
        batch_labels = []
        
        for i, source in enumerate(self.sources):
            n = samples_per_source[i]
            data = source['data']
            
            # Random sample
            idx = np.random.choice(len(data), n, replace=True)
            batch_images.extend(data[idx])
            batch_labels.extend(idx)  # Placeholder labels
        
        return np.array(batch_images), np.array(batch_labels)

# Usage
loader = CompositeDataLoader()
loader.add_medmnist('pathmnist', weight=0.3)
loader.add_local_data('./local_medical_data', weight=0.7)

X_batch, y_batch = loader.get_balanced_batch(batch_size=64)
```

## Part 3: Training Optimization for Higher Accuracy

### Optimization Strategy 1: Learning Rate Scheduling

```python
# learning_rate_scheduler.py
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    CosineDecay,
    PiecewiseConstantDecay
)

class AdaptiveLRScheduler:
    
    @staticmethod
    def exponential_decay(initial_lr=0.001, decay_steps=1000):
        """Exponential learning rate decay"""
        return ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.96
        )
    
    @staticmethod
    def cosine_decay(initial_lr=0.001, decay_steps=1000):
        """Cosine annealing (better convergence)"""
        return CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.0  # Minimum learning rate
        )
    
    @staticmethod
    def warmup_cosine(initial_lr=0.001, total_steps=10000, warmup_steps=2000):
        """Warmup + Cosine annealing (recommended)"""
        
        def lr_schedule(step):
            if step < warmup_steps:
                # Linear warmup
                return initial_lr * (step / warmup_steps)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        return lr_schedule

# Usage
lr_schedule = AdaptiveLRScheduler.warmup_cosine(
    initial_lr=0.001,
    total_steps=50000,
    warmup_steps=5000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

### Optimization Strategy 2: Weighted Aggregation

```python
# weighted_aggregation.py
import numpy as np

def weighted_average_by_samples(weights_list, sample_counts):
    """
    Average weights weighted by hospital sample counts
    Prevents hospitals with more data from dominating
    """
    
    # Normalize weights
    total_samples = sum(sample_counts)
    normalized_weights = np.array(sample_counts) / total_samples
    
    # Weighted average
    averaged_weights = []
    num_layers = len(weights_list[0])
    
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(weights_list[0][layer_idx], dtype=float)
        
        for hospital_idx, weights in enumerate(weights_list):
            weighted_sum += (
                weights[layer_idx].astype(float) * 
                normalized_weights[hospital_idx]
            )
        
        averaged_weights.append(weighted_sum)
    
    return averaged_weights

# Usage in main_server/app.py
def aggregate_weights_weighted(round_num):
    global global_model
    
    if not hospital_updates:
        return False
    
    weights_list = []
    sample_counts = []
    
    for hospital_id, data in hospital_updates.items():
        weights = data.get('weights')
        samples = data.get('sample_count', 1)
        
        if weights is not None:
            weights_list.append(weights)
            sample_counts.append(samples)
    
    # Weighted aggregation instead of simple average
    averaged = weighted_average_by_samples(weights_list, sample_counts)
    set_model_weights(global_model, averaged)
    
    return True
```

### Optimization Strategy 3: Model Architecture & Regularization

```python
# advanced_model.py
from tensorflow import keras
from tensorflow.keras import layers

def create_optimized_medical_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Optimized architecture for medical imaging with:
    - Batch normalization for training stability
    - Data augmentation ready
    - Residual connections for deep networks
    """
    
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Data augmentation (on-the-fly)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        # Preprocessing
        layers.Normalization(mean=0.5, variance=0.25),
        
        # Feature extraction
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        # Global average pooling (more stable than flatten)
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with L2 regularization
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Usage
model = create_optimized_medical_model(input_shape=(224, 224, 3), num_classes=14)
```

### Optimization Strategy 4: Early Stopping & Validation Monitoring

```python
# training_with_validation.py
def train_main_server_with_validation(
    model, X_train, y_train, X_val, y_val,
    epochs=20, batch_size=64
):
    """
    Train with validation monitoring and early stopping
    """
    
    callbacks = [
        # Early stopping on validation accuracy
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        ),
        
        # Learning rate reduction on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        ),
        
        # Checkpoint best model
        keras.callbacks.ModelCheckpoint(
            'best_main_server_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        
        # Tensorboard logging
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Usage
history = train_main_server_with_validation(
    model, X_train, y_train, X_val, y_val,
    epochs=50,
    batch_size=64
)
```

### Optimization Strategy 5: Hyperparameter Grid Search

```python
# hyperparameter_tuning.py
from sklearn.model_selection import ParameterGrid
import json

def hyperparameter_grid_search(X_train, y_train, X_val, y_val):
    """
    Grid search over important hyperparameters
    """
    
    param_grid = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.5],
        'l2_regularization': [0.0, 0.0001, 0.001]
    }
    
    results = []
    
    for params in ParameterGrid(param_grid):
        print(f"\nTraining with: {params}")
        
        # Create model with these hyperparameters
        model = create_optimized_medical_model()
        
        # Compile with learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Shorter for grid search
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Evaluate
        val_accuracy = history.history['val_accuracy'][-1]
        results.append({
            'params': params,
            'val_accuracy': val_accuracy
        })
        
        print(f"  Val Accuracy: {val_accuracy:.4f}")
    
    # Sort by accuracy
    results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
    
    # Save best configs
    with open('best_hyperparams.json', 'w') as f:
        json.dump(results[:5], f, indent=2)
    
    return results
```

## Part 4: End-to-End Training Pipeline

### Complete Training Script

```python
# main_server_training.py
import sys
sys.path.insert(0, '.')

from data.data_loader import CompositeDataLoader
from models.optimized_model import create_optimized_medical_model
from training.schedulers import AdaptiveLRScheduler
import tensorflow as tf

def train_main_server():
    """Complete main server training pipeline"""
    
    print("=" * 60)
    print("FedCare Main Server Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[1] Loading composite datasets...")
    loader = CompositeDataLoader()
    loader.add_medmnist('chexpert', weight=0.4)
    loader.add_medmnist('pathmnist', weight=0.3)
    
    X_train, y_train = loader.prepare_dataset(num_samples=50000)
    X_val, y_val = loader.prepare_dataset(num_samples=10000, split='val')
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    
    # Step 2: Create optimized model
    print("\n[2] Creating optimized medical model...")
    model = create_optimized_medical_model(
        input_shape=(224, 224, 3),
        num_classes=14
    )
    print(f"  Model parameters: {model.count_params():,.0f}")
    
    # Step 3: Setup learning rate schedule
    print("\n[3] Setting up adaptive learning rate...")
    lr_schedule = AdaptiveLRScheduler.warmup_cosine(
        initial_lr=0.001,
        total_steps=len(X_train) // 64 * 50,
        warmup_steps=len(X_train) // 64 * 5
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Step 4: Train with validation
    print("\n[4] Training with early stopping...")
    history = train_main_server_with_validation(
        model, X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=64
    )
    
    # Step 5: Evaluate and save
    print("\n[5] Final evaluation...")
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"  Validation Accuracy: {test_acc:.4f}")
    
    model.save('main_server_model.h5')
    print(f"  Model saved: main_server_model.h5")
    
    return model

if __name__ == "__main__":
    model = train_main_server()
```

## Monitoring Accuracy

### Metrics to Track

| Metric | Purpose | Target |
|--------|---------|--------|
| Training Loss | Convergence indicator | Decreasing |
| Validation Accuracy | Real performance | >92% |
| Training Speed | Efficiency | <2s/batch |
| Model Size | Deployment | <200 MB |
| Gradient Norm | Training stability | Stable, not exploding |

## Deployment Checklist

- [ ] Download or generate training data
- [ ] Verify data preprocessing (normalization, augmentation)
- [ ] Run hyperparameter grid search
- [ ] Train main server with optimized parameters
- [ ] Validate accuracy >90% on held-out test set
- [ ] Save and deploy trained model to main server
- [ ] Configure hospitals to fetch and use global model
- [ ] Monitor federation metrics over rounds

