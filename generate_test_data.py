"""
Generate test data from MNIST or synthetic data for all hospital servers
Run this before starting the orchestrator to pre-load training data
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def save_hospital_data(hospital_data, output_dir="data_simulation"):
    """Save hospital data to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (X, y) in enumerate(hospital_data):
        hospital_id = f"hospital_{i+1}"
        filename = os.path.join(output_dir, f"test_data_{hospital_id}.json")
        
        # Flatten images for JSON storage
        X_flat = X.reshape(X.shape[0], -1).tolist()
        y_list = y.tolist()
        
        data = {
            "hospital_id": hospital_id,
            "num_samples": len(X),
            "input_shape": [28, 28, 1],
            "num_classes": 10,
            "X_train": X_flat,
            "y_train": y_list
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"✓ Saved {filename}")
        print(f"  - Samples: {len(X)}")
        print(f"  - Shape: {X.shape}")
        print()

def print_data_summary(hospital_data):
    """Print summary of hospital data"""
    print("\n" + "="*60)
    print("HOSPITAL DATA SUMMARY")
    print("="*60 + "\n")
    
    for i, (X, y) in enumerate(hospital_data):
        hospital_id = f"hospital_{i+1}"
        print(f"{hospital_id}:")
        print(f"  Training samples: {len(X)}")
        print(f"  Input shape: {X.shape}")
        print(f"  Data type: {X.dtype}")
        print(f"  Pixel range: [{X.min():.3f}, {X.max():.3f}]")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Classes present: {len(unique)}")
        print(f"  Label distribution:")
        for label, count in zip(unique, counts):
            print(f"    Class {label}: {count} samples")
        print()

def load_mnist_from_tensorflow(num_hospitals=3):
    """Load MNIST from TensorFlow"""
    try:
        from tensorflow.keras.datasets import mnist
        
        print("Downloading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Reshape
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Split among hospitals
        hospital_data = []
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Split
        samples_per_hospital = len(X_train) // num_hospitals
        
        for i in range(num_hospitals):
            start_idx = i * samples_per_hospital
            if i == num_hospitals - 1:
                end_idx = len(X_train)
            else:
                end_idx = (i + 1) * samples_per_hospital
            
            X_hosp = X_train[start_idx:end_idx]
            y_hosp = y_train[start_idx:end_idx]
            
            hospital_data.append((X_hosp, y_hosp))
        
        return hospital_data
    except ImportError:
        return None

def generate_synthetic_mnist_data(num_hospitals=3, samples_per_hospital=2000):
    """Generate synthetic MNIST-like data"""
    print(f"Generating synthetic MNIST-like data ({samples_per_hospital} samples per hospital)...")
    hospital_data = []
    
    for h in range(num_hospitals):
        # Create realistic digit-like data
        X = []
        y = []
        
        for digit in range(10):
            # Generate samples per digit per hospital
            num_samples = samples_per_hospital // 10
            
            # Create a pattern for each digit
            for _ in range(num_samples):
                # Generate a 28x28 image
                img = np.zeros((28, 28), dtype=np.float32)
                
                # Add some random noise and structure
                noise = np.random.normal(0, 0.1, (28, 28))
                
                # Add digit pattern based on digit value
                center_x, center_y = 14, 14
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        # Create circular patterns for digits
                        if dist < 8 + digit:
                            img[i, j] = max(0, 1 - (dist / (8 + digit)) + np.random.normal(0, 0.05))
                
                img = np.clip(img + noise, 0, 1).astype(np.float32)
                X.append(img.reshape(28, 28, 1))
                y.append(digit)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        hospital_data.append((X, y))
    
    return hospital_data

def main():
    print("\n" + "="*60)
    print("FEDCARE TEST DATA GENERATOR")
    print("="*60 + "\n")
    
    # Try to load MNIST first
    hospital_data = load_mnist_from_tensorflow(num_hospitals=3)
    
    if hospital_data is None:
        print("\nℹ️  TensorFlow not installed, using synthetic MNIST-like data\n")
        hospital_data = generate_synthetic_mnist_data(num_hospitals=3, samples_per_hospital=2000)
    
    # Print summary
    print_data_summary(hospital_data)
    
    # Save to files
    print("Saving data to JSON files...")
    save_hospital_data(hospital_data)
    
    print("="*60)
    print("✓ Test data generation complete!")
    print("="*60)
    print("\nYou can now:")
    print("1. Load this data using the HTTP API or orchestrator")
    print("2. Use with examples.py for manual testing")
    print("3. Use with orchestrator.py for full federated learning\n")

if __name__ == "__main__":
    main()
