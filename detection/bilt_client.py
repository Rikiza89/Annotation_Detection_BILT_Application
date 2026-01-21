"""
BILT Client - Client library for communicating with BILT Service
Used by detection_app.py to interact with the BILT service API
"""

import requests
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class BILTClient:
    """Client for BILT Service API"""
    
    def __init__(self, service_url: str = "http://127.0.0.1:5002"):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to service"""
        url = f"{self.service_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check if service is running"""
        return self._make_request('GET', '/health')
    
    # Camera methods
    def get_cameras(self) -> Dict[str, Any]:
        """Get available cameras"""
        return self._make_request('GET', '/api/cameras')
    
    def select_camera(self, camera_index: int) -> Dict[str, Any]:
        """Select camera by index"""
        return self._make_request('POST', '/api/camera/select', 
                                  json={'camera_index': camera_index})
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information"""
        return self._make_request('GET', '/api/camera/info')
    
    def set_camera_resolution(self, width: int, height: int) -> Dict[str, Any]:
        """Set camera resolution"""
        return self._make_request('POST', '/api/camera/resolution',
                                  json={'width': width, 'height': height})
    
    # Model methods
    def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        return self._make_request('GET', '/api/models')
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a YOLO model"""
        return self._make_request('POST', '/api/model/load',
                                  json={'model_name': model_name})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return self._make_request('GET', '/api/model/info')
    
    # Detection methods
    def get_detection_settings(self) -> Dict[str, Any]:
        """Get current detection settings"""
        return self._make_request('GET', '/api/detection/settings')
    
    def update_detection_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update detection settings"""
        return self._make_request('POST', '/api/detection/settings', json=settings)
    
    def start_detection(self) -> Dict[str, Any]:
        """Start detection"""
        return self._make_request('POST', '/api/detection/start')
    
    def stop_detection(self) -> Dict[str, Any]:
        """Stop detection"""
        return self._make_request('POST', '/api/detection/stop')
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return self._make_request('GET', '/api/detection/stats')
    
    def get_latest_frame(self) -> Optional[bytes]:
        """Get latest frame as JPEG bytes"""
        try:
            url = f"{self.service_url}/api/frame/latest"
            response = self.session.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    # Counter methods
    def get_counters(self) -> Dict[str, Any]:
        """Get object counters"""
        return self._make_request('GET', '/api/counters')
    
    def reset_counters(self) -> Dict[str, Any]:
        """Reset object counters"""
        return self._make_request('POST', '/api/counters/reset')
    
    # Chain detection methods
    def get_chain_status(self) -> Dict[str, Any]:
        """Get chain detection status"""
        return self._make_request('GET', '/api/chain/status')
    
    def chain_control(self, action: str) -> Dict[str, Any]:
        """Control chain detection (start/stop/reset)"""
        return self._make_request('POST', '/api/chain/control',
                                  json={'action': action})
    
    def get_chain_config(self) -> Dict[str, Any]:
        """Get chain configuration"""
        return self._make_request('GET', '/api/chain/config')
    
    def update_chain_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update chain configuration"""
        return self._make_request('POST', '/api/chain/config', json=config)
    
    def acknowledge_error(self) -> Dict[str, Any]:
        """Acknowledge error in chain detection"""
        return self._make_request('POST', '/api/chain/acknowledge_error')
    
    # Chain save/load methods
    def get_saved_chains(self) -> Dict[str, Any]:
        """Get list of saved chains"""
        return self._make_request('GET', '/api/chains/saved')
    
    def save_chain(self, chain_name: str, model_name: str) -> Dict[str, Any]:
        """Save current chain configuration"""
        return self._make_request('POST', '/api/chains/save',
                                  json={'chain_name': chain_name, 'model_name': model_name})
    
    def load_chain(self, chain_name: str) -> Dict[str, Any]:
        """Load chain configuration"""
        return self._make_request('POST', '/api/chains/load',
                                  json={'chain_name': chain_name})
    
    def delete_chain(self, chain_name: str) -> Dict[str, Any]:
        """Delete saved chain"""
        return self._make_request('POST', '/api/chains/delete',
                                  json={'chain_name': chain_name})
    
    # Project methods
    def get_projects(self) -> Dict[str, Any]:
        """Get available projects"""
        return self._make_request('GET', '/api/projects')
    
    def create_project(self, project_name: str, description: str = "", 
                      classes: List[str] = None) -> Dict[str, Any]:
        """Create new project"""
        return self._make_request('POST', '/api/projects/create',
                                  json={
                                      'project_name': project_name,
                                      'description': description,
                                      'classes': classes or []
                                  })


class BILTServiceManager:
    """High-level manager for BILT service operations"""
    
    def __init__(self, service_url: str = "http://127.0.0.1:5002"):
        self.client = BILTClient(service_url)
    
    def is_service_available(self) -> bool:
        """Check if BILT service is available"""
        result = self.client.health_check()
        return result.get('status') == 'ok'
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera"""
        result = self.client.select_camera(camera_index)
        return result.get('success', False)
    
    def initialize_model(self, model_name: str) -> tuple:
        """Initialize model and return (success, model_info or error)"""
        result = self.client.load_model(model_name)
        if result.get('success'):
            return True, result.get('model_info')
        return False, result.get('error')
    
    def start_detection_session(self) -> bool:
        """Start detection session"""
        result = self.client.start_detection()
        return result.get('success', False)
    
    def stop_detection_session(self) -> bool:
        """Stop detection session"""
        result = self.client.stop_detection()
        return result.get('success', False)
    
    def configure_detection(self, conf: float = None, iou: float = None,
                          max_det: int = None, classes: List[int] = None,
                          counter_mode: bool = None, save_images: bool = None) -> bool:
        """Configure detection parameters"""
        settings = {}
        if conf is not None:
            settings['conf'] = conf
        if iou is not None:
            settings['iou'] = iou
        if max_det is not None:
            settings['max_det'] = max_det
        if classes is not None:
            settings['classes'] = classes
        if counter_mode is not None:
            settings['counter_mode'] = counter_mode
        if save_images is not None:
            settings['save_images'] = save_images
        
        if settings:
            result = self.client.update_detection_settings(settings)
            return result.get('success', False)
        return True
    
    def setup_chain_detection(self, chain_steps: List[Dict], timeout: float = 5.0,
                             auto_advance: bool = True, pause_time: float = 10.0) -> bool:
        """Setup chain detection"""
        config = {
            'chain_steps': chain_steps,
            'chain_timeout': timeout,
            'chain_auto_advance': auto_advance,
            'chain_pause_time': pause_time
        }
        result = self.client.update_chain_config(config)
        return result.get('success', False)
    
    def start_chain(self) -> bool:
        """Start chain detection"""
        result = self.client.chain_control('start')
        return result.get('success', False)
    
    def stop_chain(self) -> bool:
        """Stop chain detection"""
        result = self.client.chain_control('stop')
        return result.get('success', False)
    
    def reset_chain(self) -> bool:
        """Reset chain detection"""
        result = self.client.chain_control('reset')
        return result.get('success', False)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get current detection statistics"""
        result = self.client.get_detection_stats()
        return result.get('stats', {})
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get object counter values"""
        result = self.client.get_counters()
        return result.get('counters', {})
    
    def clear_counters(self) -> bool:
        """Clear object counters"""
        result = self.client.reset_counters()
        return result.get('success', False)


# Convenience function for quick service check
def check_bilt_service(service_url: str = "http://127.0.0.1:5002") -> bool:
    """Quick check if BILT service is running"""
    try:
        client = BILTClient(service_url)
        result = client.health_check()
        return result.get('status') == 'ok'
    except Exception:
        return False


if __name__ == '__main__':
    # Test the client
    print("Testing BILT Client...")
    client = BILTClient()
    
    # Check service health
    health = client.health_check()
    print(f"Service health: {health}")
    
    if health.get('status') == 'ok':
        # Get cameras
        cameras = client.get_cameras()
        print(f"Available cameras: {cameras}")
        
        # Get models
        models = client.get_models()
        print(f"Available models: {models}")
        
        # Get detection settings
        settings = client.get_detection_settings()
        print(f"Detection settings: {settings}")
        
        print("\nBILT Client test completed!")
    else:
        print("BILT Service is not available. Make sure it's running on port 5002")