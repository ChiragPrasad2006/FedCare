"""
Orchestration script for federated learning process
"""
import requests
import time
import json
import logging
from typing import List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedLearningOrchestrator:
    """Orchestrates the federated learning process"""
    
    def __init__(self, main_server_url: str, hospital_urls: List[str]):
        self.main_server_url = main_server_url
        self.hospital_urls = hospital_urls
        self.num_hospitals = len(hospital_urls)
        self.max_retries = 3
        self.timeout = 30
    
    def _make_request(self, method: str, url: str, json_data: dict = None, timeout: int = None):
        """Make HTTP request with retry logic"""
        if timeout is None:
            timeout = self.timeout
        
        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = requests.get(url, timeout=timeout)
                elif method == 'POST':
                    response = requests.post(url, json=json_data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                if response.status_code in [200, 201]:
                    return response.json()
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def initialize_system(self):
        """Initialize the federated learning system"""
        logger.info("Initializing FedCare system...")
        
        # Initialize main server
        result = self._make_request('POST', f"{self.main_server_url}/initialize", {})
        if not result:
            logger.error("Failed to initialize main server")
            return False
        
        logger.info("Main server initialized")
        
        # Configure hospitals
        for i, hospital_url in enumerate(self.hospital_urls):
            hospital_id = f"hospital_{i+1}"
            config_data = {"hospital_id": hospital_id}
            
            result = self._make_request('POST', f"{hospital_url}/configure", config_data)
            if result:
                logger.info(f"Configured {hospital_id}")
            else:
                logger.error(f"Failed to configure {hospital_id}")
        
        return True
    
    def load_hospital_data(self, hospital_data_list: List[tuple]):
        """Load data into hospital servers"""
        logger.info(f"Loading data into {self.num_hospitals} hospital servers...")
        
        for i, (X_train, y_train) in enumerate(hospital_data_list):
            if i >= self.num_hospitals:
                break
            
            # Convert to list for JSON serialization
            X_train_list = X_train.tolist()
            y_train_list = y_train.tolist()
            
            data_payload = {
                "X_train": X_train_list,
                "y_train": y_train_list
            }
            
            result = self._make_request(
                'POST',
                f"{self.hospital_urls[i]}/load_data",
                data_payload,
                timeout=60  # Longer timeout for data loading
            )
            
            if result:
                logger.info(f"Data loaded into hospital_{i+1}: {result.get('num_samples')} samples")
            else:
                logger.error(f"Failed to load data into hospital_{i+1}")
        
        return True
    
    def run_federated_learning(self, num_rounds: int, enable_personalization: bool = False):
        """
        Run federated learning for specified rounds with optional personalization.
        
        Args:
            num_rounds: Number of FL rounds
            enable_personalization: Enable FedProx personalization after aggregation
        """
        logger.info(f"Starting federated learning for {num_rounds} rounds...")
        if enable_personalization:
            logger.info("Personalization enabled - will use FedProx after each aggregation")
        
        for round_num in range(num_rounds):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num + 1}/{num_rounds}")
            logger.info(f"{'='*60}")
            
            # Step 1: Hospitals sync and train
            logger.info("Step 1: Hospitals syncing and training...")
            all_successful = True
            
            for i, hospital_url in enumerate(self.hospital_urls):
                hospital_id = f"hospital_{i+1}"
                
                result = self._make_request(
                    'POST',
                    f"{hospital_url}/sync_and_train",
                    {"round": round_num},
                    timeout=120
                )
                
                if result:
                    metrics = result.get('metrics', {})
                    logger.info(
                        f"{hospital_id} completed training - "
                        f"Loss: {metrics.get('loss', 'N/A'):.4f}, "
                        f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}"
                    )
                else:
                    logger.error(f"{hospital_id} training failed")
                    all_successful = False
            
            if not all_successful:
                logger.warning("Some hospitals failed training")
            
            # Step 2: Main server aggregates
            logger.info("Step 2: Main server aggregating updates...")
            result = self._make_request(
                'POST',
                f"{self.main_server_url}/aggregate",
                {"trigger_next_round": True}
            )
            
            if result:
                logger.info(f"Aggregation completed - Advanced to round {result.get('round')}")
            else:
                logger.error("Aggregation failed")
            
            # Step 3: Optional personalization phase
            if enable_personalization:
                logger.info("Step 3: Hospitals personalizing models (FedProx)...")
                result = self._make_request(
                    'POST',
                    f"{self.main_server_url}/trigger_personalization",
                    {
                        "round": round_num,
                        "use_fedprox": True
                    }
                )
                
                if result:
                    results = result.get('results', {})
                    completed = sum(1 for r in results.values() if r.get('status') == 'completed')
                    logger.info(f"Personalization completed for {completed}/{len(results)} hospitals")
                    
                    # Log personalization metrics
                    for hospital_id, res in results.items():
                        if res.get('status') == 'completed':
                            metrics = res.get('metrics', {})
                            logger.info(
                                f"  {hospital_id} personalization - "
                                f"Loss: {metrics.get('loss', 'N/A'):.4f}, "
                                f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}"
                            )
                else:
                    logger.error("Personalization trigger failed")
            
            # Wait a bit before next round
            time.sleep(2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Federated learning completed!")
        logger.info(f"{'='*60}\n")
    
    def get_metrics(self):
        """Get metrics from main server"""
        result = self._make_request('GET', f"{self.main_server_url}/metrics")
        if result:
            return result.get('metrics', [])
        return []
    
    def get_hospital_history(self):
        """Get training history from all hospitals"""
        history = {}
        
        for i, hospital_url in enumerate(self.hospital_urls):
            hospital_id = f"hospital_{i+1}"
            result = self._make_request('GET', f"{hospital_url}/training_history")
            
            if result:
                history[hospital_id] = result.get('history', [])
        
        return history
    
    def print_final_report(self):
        """Print final training report"""
        logger.info("\n" + "="*60)
        logger.info("FEDERATED LEARNING FINAL REPORT")
        logger.info("="*60)
        
        # Get metrics
        metrics = self.get_metrics()
        history = self.get_hospital_history()
        
        # Print by round
        if metrics:
            rounds_data = {}
            for metric in metrics:
                round_num = metric.get('round')
                if round_num not in rounds_data:
                    rounds_data[round_num] = []
                rounds_data[round_num].append(metric)
            
            for round_num in sorted(rounds_data.keys()):
                logger.info(f"\nRound {round_num + 1}:")
                round_metrics = rounds_data[round_num]
                
                avg_loss = np.mean([m['metrics'].get('loss', 0) for m in round_metrics])
                avg_accuracy = np.mean([m['metrics'].get('accuracy', 0) for m in round_metrics])
                
                logger.info(f"  Average Loss: {avg_loss:.4f}")
                logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
                logger.info(f"  Hospitals reporting: {len(round_metrics)}")
        
        logger.info("\n" + "="*60)
        logger.info("Report completed")
        logger.info("="*60 + "\n")


def main():
    """Main orchestration function"""
    import sys
    sys.path.insert(0, '.')
    
    from data_simulation.data_generator import load_and_split_mnist_data
    
    # Configuration
    MAIN_SERVER_URL = "http://localhost:5000"
    HOSPITAL_URLS = [
        "http://localhost:5001",
        "http://localhost:5002",
        "http://localhost:5003"
    ]
    NUM_ROUNDS = 5
    NUM_HOSPITALS = 3
    
    # Initialize orchestrator
    orchestrator = FederatedLearningOrchestrator(MAIN_SERVER_URL, HOSPITAL_URLS)
    
    # Wait for servers to be ready
    logger.info("Waiting for servers to be ready...")
    for _ in range(30):
        try:
            response = requests.get(f"{MAIN_SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                logger.info("Main server is ready")
                break
        except:
            time.sleep(1)
    
    # Initialize system
    if not orchestrator.initialize_system():
        logger.error("Failed to initialize system")
        return
    
    # Generate and load data
    logger.info("Generating MNIST data...")
    hospital_data, test_data = load_and_split_mnist_data(num_hospitals=NUM_HOSPITALS)
    
    if not orchestrator.load_hospital_data(hospital_data):
        logger.error("Failed to load hospital data")
        return
    
    # Run federated learning
    orchestrator.run_federated_learning(NUM_ROUNDS)
    
    # Print final report
    orchestrator.print_final_report()


if __name__ == "__main__":
    main()
