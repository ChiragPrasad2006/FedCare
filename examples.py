"""
Example usage of FedCare system - Quick Start Guide
"""
import requests
import time
import json
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_simulation.data_generator import load_and_split_mnist_data, create_non_iid_data
from orchestrator import FederatedLearningOrchestrator

def example_basic_flow():
    """
    Example 1: Basic federated learning flow
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Federated Learning Flow")
    print("="*60 + "\n")
    
    # Configuration
    MAIN_SERVER_URL = "http://localhost:5000"
    HOSPITAL_URLS = [
        "http://localhost:5001",
        "http://localhost:5002",
        "http://localhost:5003"
    ]
    
    try:
        # 1. Check if main server is running
        print("1. Checking main server health...")
        response = requests.get(f"{MAIN_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Main server is running")
        else:
            print("   ✗ Main server not responding")
            return
        
        # 2. Initialize system
        print("\n2. Initializing federated learning system...")
        response = requests.post(f"{MAIN_SERVER_URL}/initialize")
        print(f"   ✓ {response.json()['message']}")
        
        # 3. Configure hospitals
        print("\n3. Configuring hospital servers...")
        for i, hospital_url in enumerate(HOSPITAL_URLS):
            try:
                response = requests.post(
                    f"{hospital_url}/configure",
                    json={"hospital_id": f"hospital_{i+1}"}
                )
                print(f"   ✓ hospital_{i+1} configured")
            except:
                print(f"   ✗ hospital_{i+1} not responding")
        
        # 4. Generate and load data
        print("\n4. Generating and loading training data...")
        hospital_data, test_data = load_and_split_mnist_data(num_hospitals=3)
        
        for i, (X_train, y_train) in enumerate(hospital_data):
            print(f"   Loading data into hospital_{i+1}...")
            print(f"   - Samples: {len(X_train)}")
            print(f"   - Shape: {X_train.shape}")
            
            # Convert to list for JSON
            X_train_list = X_train.reshape(X_train.shape[0], -1).tolist()
            y_train_list = y_train.tolist()
            
            try:
                response = requests.post(
                    f"{HOSPITAL_URLS[i]}/load_data",
                    json={"X_train": X_train_list, "y_train": y_train_list},
                    timeout=120
                )
                print(f"   ✓ hospital_{i+1}: {response.json()['num_samples']} samples loaded")
            except Exception as e:
                print(f"   ✗ Error loading data: {e}")
        
        # 5. Run training rounds
        print("\n5. Running federated learning rounds...")
        for round_num in range(3):  # Run 3 rounds as demo
            print(f"\n   ROUND {round_num + 1}")
            print(f"   " + "-"*40)
            
            # Sync and train all hospitals
            for i, hospital_url in enumerate(HOSPITAL_URLS):
                try:
                    response = requests.post(
                        f"{hospital_url}/sync_and_train",
                        json={"round": round_num},
                        timeout=120
                    )
                    result = response.json()
                    metrics = result.get('metrics', {})
                    print(
                        f"   hospital_{i+1}: "
                        f"Loss={metrics.get('loss', 0):.4f}, "
                        f"Accuracy={metrics.get('accuracy', 0):.4f}"
                    )
                except Exception as e:
                    print(f"   hospital_{i+1}: Error - {e}")
            
            # Aggregate
            try:
                response = requests.post(f"{MAIN_SERVER_URL}/aggregate")
                print(f"   ✓ Model aggregated - Round {response.json()['round']}")
            except Exception as e:
                print(f"   ✗ Aggregation error: {e}")
            
            time.sleep(2)
        
        # 6. Get final metrics
        print("\n6. Final Metrics:")
        try:
            response = requests.get(f"{MAIN_SERVER_URL}/metrics")
            metrics = response.json()['metrics']
            
            # Group by round
            by_round = {}
            for m in metrics:
                round_num = m['round']
                if round_num not in by_round:
                    by_round[round_num] = []
                by_round[round_num].append(m)
            
            for round_num in sorted(by_round.keys()):
                round_metrics = by_round[round_num]
                avg_loss = np.mean([m['metrics']['loss'] for m in round_metrics])
                avg_acc = np.mean([m['metrics']['accuracy'] for m in round_metrics])
                print(f"   Round {round_num}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
        except Exception as e:
            print(f"   Error getting metrics: {e}")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_non_iid_data():
    """
    Example 2: Using non-IID (heterogeneous) data
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Non-IID Data Distribution")
    print("="*60 + "\n")
    
    try:
        print("Generating non-IID data for 3 hospitals...")
        hospital_data = create_non_iid_data(
            num_hospitals=3,
            num_samples_per_hospital=500,
            num_classes=10
        )
        
        for i, (X, y) in enumerate(hospital_data):
            unique_classes = np.unique(y)
            print(f"Hospital {i+1}:")
            print(f"  - Samples: {len(X)}")
            print(f"  - Shape: {X.shape}")
            print(f"  - Classes: {sorted(unique_classes.tolist())}")
            print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        print("\n✓ Non-IID data generated successfully!")
        print("Note: Each hospital has different class distribution")
        print("This simulates realistic federated learning scenario")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_api_calls():
    """
    Example 3: Direct API calls
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Direct API Calls")
    print("="*60 + "\n")
    
    MAIN_SERVER_URL = "http://localhost:5000"
    
    try:
        # Health check
        print("1. Health Check:")
        response = requests.get(f"{MAIN_SERVER_URL}/health")
        print(json.dumps(response.json(), indent=2))
        
        # System status
        print("\n2. System Status:")
        response = requests.get(f"{MAIN_SERVER_URL}/status")
        print(json.dumps(response.json(), indent=2))
        
        # Get model info
        print("\n3. Global Model Info:")
        response = requests.get(f"{MAIN_SERVER_URL}/get_global_model")
        result = response.json()
        print(f"   Round: {result['round']}")
        print(f"   Weights size: {len(result['weights'])} bytes (base64)")
        print(f"   Timestamp: {result['timestamp']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_orchestrator():
    """
    Example 4: Using the orchestrator for automated flow
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Using Orchestrator")
    print("="*60 + "\n")
    
    try:
        orchestrator = FederatedLearningOrchestrator(
            main_server_url="http://localhost:5000",
            hospital_urls=[
                "http://localhost:5001",
                "http://localhost:5002",
                "http://localhost:5003"
            ]
        )
        
        # Initialize
        print("1. Initializing system...")
        if not orchestrator.initialize_system():
            print("   ✗ Failed to initialize")
            return
        print("   ✓ System initialized")
        
        # Load data
        print("\n2. Loading training data...")
        hospital_data, test_data = load_and_split_mnist_data(num_hospitals=3)
        if not orchestrator.load_hospital_data(hospital_data):
            print("   ✗ Failed to load data")
            return
        print("   ✓ Data loaded")
        
        # Run federated learning
        print("\n3. Running federated learning (2 rounds)...")
        orchestrator.run_federated_learning(num_rounds=2)
        
        # Get report
        print("\n4. Final Report:")
        orchestrator.print_final_report()
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run examples"""
    print("\n" + "="*70)
    print("FedCare - Federated Learning Examples")
    print("="*70)
    
    print("\nAvailable examples:")
    print("1. Basic Flow (recommended)")
    print("2. Non-IID Data")
    print("3. Direct API Calls")
    print("4. Orchestrator (automated)")
    print("0. Run all")
    
    choice = input("\nSelect example to run (0-4): ").strip()
    
    if choice == "1":
        example_basic_flow()
    elif choice == "2":
        example_non_iid_data()
    elif choice == "3":
        example_api_calls()
    elif choice == "4":
        example_orchestrator()
    elif choice == "0":
        example_non_iid_data()
        example_api_calls()
        example_basic_flow()
        # Uncomment to run orchestrator
        # example_orchestrator()
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
