#!/usr/bin/env python3
"""
Example: FedProx Personalization Workflow

This script demonstrates how to use FedProx personalization
to improve local hospital accuracy while maintaining federation cohesion.

Usage:
  python examples/fedprox_personalization_example.py

Requirements:
  - Main server running on http://localhost:5000
  - 3 hospital servers running on ports 5001, 5002, 5003
  - PERSONALIZATION_ENABLED=True in .env
"""

import sys
import os
import json
import requests
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import setup_logger

logger = setup_logger("fedprox_example")


class FedProxPersonalizationDemo:
    """Demonstrates FedProx personalization workflow"""
    
    def __init__(self, main_url="http://localhost:5000"):
        self.main_url = main_url
        self.hospital_urls = [
            "http://localhost:5001",
            "http://localhost:5002",
            "http://localhost:5003"
        ]
    
    def run_training_round(self, round_num):
        """Execute one training round without personalization"""
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ROUND {round_num} (No Personalization)")
        logger.info(f"{'='*60}")
        
        # Step 1: Hospitals train
        logger.info("[1] Hospitals training locally...")
        for i, url in enumerate(self.hospital_urls):
            try:
                response = requests.post(
                    f"{url}/sync_and_train",
                    json={"round": round_num},
                    timeout=120
                )
                if response.status_code == 200:
                    metrics = response.json()['metrics']
                    logger.info(
                        f"  Hospital {i+1}: loss={metrics['loss']:.4f}, "
                        f"accuracy={metrics['accuracy']:.4f}"
                    )
            except Exception as e:
                logger.error(f"  Hospital {i+1} failed: {e}")
        
        # Step 2: Main server aggregates
        logger.info("[2] Main server aggregating...")
        try:
            response = requests.post(
                f"{self.main_url}/aggregate",
                json={},
                timeout=60
            )
            if response.status_code == 200:
                logger.info(f"  Aggregation completed")
        except Exception as e:
            logger.error(f"  Aggregation failed: {e}")
    
    def run_personalization_round(self, round_num, use_fedprox=True):
        """Execute personalization phase"""
        logger.info(f"\n{'='*60}")
        logger.info(f"PERSONALIZATION ROUND {round_num} (FedProx)")
        logger.info(f"{'='*60}")
        
        # Get individual hospital personalization results
        logger.info("[1] Triggering FedProx personalization...")
        try:
            response = requests.post(
                f"{self.main_url}/trigger_personalization",
                json={
                    "round": round_num,
                    "use_fedprox": use_fedprox
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                results = result['results']
                
                completed = sum(1 for r in results.values() if r['status'] == 'completed')
                logger.info(f"  Completed: {completed}/{len(results)} hospitals")
                
                # Log results
                for hospital_id, res in results.items():
                    if res['status'] == 'completed':
                        metrics = res['metrics']
                        logger.info(
                            f"    {hospital_id}: loss={metrics.get('loss', 'N/A'):.4f}, "
                            f"accuracy={metrics.get('accuracy', 'N/A'):.4f}"
                        )
                    else:
                        logger.warning(f"    {hospital_id}: {res['status']}")
        except Exception as e:
            logger.error(f"  Personalization trigger failed: {e}")
        
        # Get aggregated metrics
        logger.info("[2] Fetching aggregated personalization metrics...")
        try:
            response = requests.get(
                f"{self.main_url}/personalization_metrics?round={round_num}",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                metrics = data['personalization_metrics']
                logger.info(f"  Total personalization results: {len(metrics)}")
        except Exception as e:
            logger.error(f"  Failed to fetch metrics: {e}")
    
    def compare_with_and_without_personalization(self, num_rounds=5):
        """
        Compare FL performance with and without personalization
        """
        logger.info("\n" + "="*60)
        logger.info("FedProx Personalization Comparison")
        logger.info("="*60)
        
        # Phase 1: Training without personalization
        logger.info("\nPHASE 1: Without Personalization")
        logger.info("-" * 40)
        
        for round_num in range(num_rounds // 2):
            self.run_training_round(round_num)
            time.sleep(1)
        
        # Get metrics without personalization
        try:
            response = requests.get(
                f"{self.main_url}/metrics",
                timeout=30
            )
            if response.status_code == 200:
                metrics_without = response.json()
                logger.info(f"\nMetrics WITHOUT personalization: {metrics_without}")
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
        
        # Phase 2: Training WITH personalization
        logger.info("\nPHASE 2: With FedProx Personalization")
        logger.info("-" * 40)
        
        for round_num in range(num_rounds // 2, num_rounds):
            self.run_training_round(round_num)
            self.run_personalization_round(round_num, use_fedprox=True)
            time.sleep(1)
        
        # Get metrics with personalization
        try:
            response = requests.get(
                f"{self.main_url}/metrics",
                timeout=30
            )
            if response.status_code == 200:
                metrics_with = response.json()
                logger.info(f"\nMetrics WITH personalization: {metrics_with}")
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info("✓ Training without personalization: Rounds 0-2")
        logger.info("✓ Training with FedProx personalization: Rounds 3-4")
        logger.info("\nObservations:")
        logger.info("- Global model accuracy may decrease slightly with personalization")
        logger.info("- Individual hospital accuracy improves significantly")
        logger.info("- Convergence speed typically improves")
        logger.info("- Better adaptation to local data distributions")
    
    def monitor_personalization_history(self, hospital_id="hospital_1"):
        """Monitor personalization history for a specific hospital"""
        logger.info(f"\nMonitoring personalization history for {hospital_id}...")
        
        for i in range(5):
            try:
                response = requests.get(
                    f"http://localhost:{5001 + i}/personalization_history?limit=5",
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    history = data['personalization_history']
                    
                    logger.info(f"\n{hospital_id} Personalization History:")
                    for entry in history[-3:]:
                        metrics = entry['metrics']
                        logger.info(
                            f"  Round {entry['round']}: "
                            f"loss={metrics.get('loss', 'N/A'):.4f}, "
                            f"accuracy={metrics.get('accuracy', 'N/A'):.4f}, "
                            f"proximal_term={metrics.get('proximal_term', 'N/A'):.6f}"
                        )
            except Exception as e:
                logger.error(f"  Error: {e}")
            
            time.sleep(1)


def main():
    """Main example execution"""
    
    logger.info("\n" + "="*60)
    logger.info("FedProx Personalization Example")
    logger.info("="*60)
    logger.info("\nThis example demonstrates:")
    logger.info("1. Standard federated learning (without personalization)")
    logger.info("2. FedProx personalization with proximal regularization")
    logger.info("3. Performance comparison")
    logger.info("\nRequirements:")
    logger.info("  - Main server: http://localhost:5000")
    logger.info("  - Hospitals: http://localhost:5001-5003")
    logger.info("  - Environment: PERSONALIZATION_ENABLED=True")
    
    # Check servers are running
    demo = FedProxPersonalizationDemo()
    
    logger.info("\nChecking server connectivity...")
    try:
        response = requests.get(f"{demo.main_url}/health", timeout=5)
        logger.info(f"✓ Main server is ready")
    except Exception as e:
        logger.error(f"✗ Main server not available: {e}")
        logger.error("\nPlease ensure main server is running:")
        logger.error("  cd main_server && python app.py")
        return
    
    # Run comparison
    logger.info("\nStarting personalization demo...\n")
    demo.compare_with_and_without_personalization(num_rounds=5)
    
    logger.info("\n" + "="*60)
    logger.info("Example completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
