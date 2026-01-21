"""
BILT Client - API wrapper for BILT Service

This client provides a simple interface to communicate with the BILT service.
It handles all HTTP communication and provides a clean API for the annotation app.
"""

import requests
from typing import Dict, Any, Optional


class BILTClient:
    """Client for communicating with BILT service via REST API"""
    
    def __init__(self, base_url: str = 'http://127.0.0.1:5001'):
        """
        Initialize BILT client
        
        Args:
            base_url: Base URL of the BILT service (default: http://127.0.0.1:5001)
        """
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({'Content-Type': 'application/json'})
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     timeout: int = 30) -> Dict[str, Any]:
        """
        Make HTTP request to BILT service
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request payload
            timeout: Request timeout in seconds
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                response = self._session.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = self._session.post(url, json=data, timeout=timeout)
            else:
                return {'error': f'Unsupported HTTP method: {method}'}
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            return {'error': 'Cannot connect to BILT service. Is it running?'}
        except requests.exceptions.Timeout:
            return {'error': 'Request to BILT service timed out'}
        except requests.exceptions.HTTPError as e:
            try:
                error_data = e.response.json()
                return error_data
            except:
                return {'error': f'HTTP error: {e.response.status_code}'}
        except Exception as e:
            return {'error': f'Unexpected error: {str(e)}'}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if BILT service is healthy
        
        Returns:
            Health status dictionary
        """
        return self._make_request('GET', '/health')
    
    def get_available_models(self, task_type: str = 'detect') -> Dict[str, Any]:
        """
        Get list of available BILT models
        
        Args:
            task_type: Type of task ('detect', 'segment', or 'obb')
            
        Returns:
            Dictionary containing list of models
        """
        return self._make_request('POST', '/models/available', 
                                 data={'task_type': task_type})
    
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start BILT training process
        
        Args:
            config: Training configuration dictionary containing:
                - model: Model name
                - task_type: Type of task
                - data_yaml: Path to data.yaml
                - project_path: Project directory path
                - epochs, batch, lr0, etc.: Training parameters
                
        Returns:
            Response dictionary with success/error status
        """
        return self._make_request('POST', '/train/start', data=config, timeout=60)
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status
        
        Returns:
            Status dictionary with 'active' boolean and status message
        """
        return self._make_request('GET', '/train/status')
    
    def start_autotrain(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start autotraining process (auto-label + train)
        
        Args:
            config: Autotrain configuration dictionary containing:
                - model_path: Path to model for auto-labeling
                - project_path: Project directory
                - data_yaml: Path to data.yaml
                - task_type: Type of task
                - conf_threshold: Confidence threshold
                - iou_threshold: IoU threshold
                - backup_enabled: Whether to backup data
                - epochs, batch, etc.: Training parameters
                
        Returns:
            Response dictionary with success/error status
        """
        return self._make_request('POST', '/autotrain/start', data=config, timeout=60)
    
    def get_autotrain_status(self) -> Dict[str, Any]:
        """
        Get current autotrain status
        
        Returns:
            Status dictionary with 'running' boolean
        """
        return self._make_request('GET', '/autotrain/status')
    
    def start_relabel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start relabeling process
        
        Args:
            config: Relabel configuration dictionary containing:
                - model_path: Path to model for predictions
                - project_path: Project directory
                - task_type: Type of task
                - target_split: 'train' or 'val'
                - conf_threshold: Confidence threshold
                - iou_threshold: IoU threshold
                - backup_enabled: Whether to backup labels
                - mode: 'all', 'labeled', or 'unlabeled'
                
        Returns:
            Response dictionary with success/error status and count
        """
        return self._make_request('POST', '/relabel/start', data=config, timeout=120)
    
    def is_service_available(self) -> bool:
        """
        Check if BILT service is available
        
        Returns:
            True if service is reachable and healthy, False otherwise
        """
        result = self.health_check()
        return 'status' in result and result['status'] == 'healthy'
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session"""
        self._session.close()


# Example usage and testing
if __name__ == '__main__':
    # Create client instance
    client = BILTClient()
    
    # Check if service is available
    print("Checking BILT service health...")
    health = client.health_check()
    print(f"Health check: {health}")
    
    if client.is_service_available():
        print("✓ BILT service is available")
        
        # Get available models
        print("\nGetting available models...")
        models = client.get_available_models('detect')
        print(f"Available models: {models}")
        
        # Get training status
        print("\nChecking training status...")
        status = client.get_training_status()
        print(f"Training status: {status}")
    else:
        print("✗ BILT service is not available")
        print("Please start yolo_service.py first")