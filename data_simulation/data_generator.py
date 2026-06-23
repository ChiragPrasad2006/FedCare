"""
Simulate patient data for federated learning
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_patient_data(num_samples=200, num_features=784, num_classes=10):
    """Generate synthetic patient data"""
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


def load_and_split_medical_data(num_hospitals=3, test_size=0.2):
    """Load Medical Data and split among hospitals"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.medical_data import ensure_medical_data_loaded
    
    data = ensure_medical_data_loaded()
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    
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
    
    return hospital_data, (X_test, y_test)


def load_and_split_custom_data(X, y, num_hospitals=3):
    """Split custom data among hospitals"""
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    hospital_data = []
    
    # Split
    samples_per_hospital = len(X) // num_hospitals
    
    for i in range(num_hospitals):
        start_idx = i * samples_per_hospital
        if i == num_hospitals - 1:
            end_idx = len(X)
        else:
            end_idx = (i + 1) * samples_per_hospital
        
        X_hosp = X[start_idx:end_idx]
        y_hosp = y[start_idx:end_idx]
        
        hospital_data.append((X_hosp, y_hosp))
    
    return hospital_data


def create_non_iid_data(num_hospitals=3, num_samples_per_hospital=1000, num_classes=10):
    """
    Create Non-IID (Independent and Identically Distributed) data
    to simulate realistic federated scenario where each hospital has different data distribution
    """
    hospital_data = []
    
    for h in range(num_hospitals):
        # Each hospital gets different class distribution
        classes_for_hospital = np.random.choice(
            num_classes, 
            size=5, 
            replace=False
        )
        
        X_hosp = []
        y_hosp = []
        
        for class_id in classes_for_hospital:
            # Generate samples for this class
            num_samples = num_samples_per_hospital // len(classes_for_hospital)
            X_class = np.random.rand(num_samples, 784).astype(np.float32)
            y_class = np.full(num_samples, class_id)
            
            X_hosp.append(X_class)
            y_hosp.append(y_class)
        
        X_hosp = np.vstack(X_hosp)
        y_hosp = np.concatenate(y_hosp)
        
        # Shuffle
        idx = np.random.permutation(len(X_hosp))
        X_hosp = X_hosp[idx]
        y_hosp = y_hosp[idx]
        
        hospital_data.append((X_hosp.reshape(-1, 28, 28, 1), y_hosp))
    
    return hospital_data


def get_hospital_dataloader(hospital_data, hospital_id, batch_size=32):
    """Get data for specific hospital"""
    if 0 <= hospital_id < len(hospital_data):
        return hospital_data[hospital_id]
    return None


if __name__ == "__main__":
    # Test data generation
    print("Generating test data...")
    
    # Generate Medical split
    hospital_data, test_data = load_and_split_medical_data(num_hospitals=3)
    print(f"Generated data for {len(hospital_data)} hospitals")
    print(f"Hospital 1 data shape: {hospital_data[0][0].shape}")
    
    # Generate non-IID data
    print("\nGenerating non-IID data...")
    non_iid_data = create_non_iid_data(num_hospitals=3)
    print(f"Generated non-IID data for {len(non_iid_data)} hospitals")
