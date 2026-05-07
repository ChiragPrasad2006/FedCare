"""
Communication utilities for federated learning servers
"""
import json
import logging
import requests
import pickle
import base64
from typing import Dict, Any, Optional
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class ServerCommunicator:
    """Handles communication between servers"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
    
    def send_model_weights(self, url: str, weights: Any, hospital_id: str) -> bool:
        """Send model weights to a server"""
        try:
            # Serialize weights
            weights_bytes = pickle.dumps(weights)
            weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
            
            payload = {
                "hospital_id": hospital_id,
                "weights": weights_b64,
                "type": "model_weights"
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully sent weights to {url}")
                        return True
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Status {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Attempt {attempt + 1}: Timeout connecting to {url}")
                    if attempt < self.max_retries - 1:
                        asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            return False
            
        except Exception as e:
            logger.error(f"Error sending weights: {str(e)}")
            return False
    
    def receive_model_weights(self, weights_b64: str) -> Any:
        """Deserialize model weights"""
        try:
            weights_bytes = base64.b64decode(weights_b64)
            weights = pickle.loads(weights_bytes)
            return weights
        except Exception as e:
            logger.error(f"Error deserializing weights: {str(e)}")
            return None
    
    async def send_async(self, url: str, data: Dict) -> bool:
        """Asynchronously send data"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=self.timeout) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Async send error: {str(e)}")
            return False
    
    def send_training_update(self, url: str, hospital_id: str, 
                            local_weights: Any, metrics: Dict) -> bool:
        """Send training update from hospital to main server"""
        try:
            weights_bytes = pickle.dumps(local_weights)
            weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
            
            payload = {
                "hospital_id": hospital_id,
                "weights": weights_b64,
                "metrics": metrics,
                "type": "training_update"
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending training update: {str(e)}")
            return False
