"""
BILT Service - Standalone service for BILT detection
Handles all BILT model operations, camera management, and detection processing
Run this service on port 5002: python yolo_service.py
"""

from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import os
import json
import threading
import time
import logging
from datetime import datetime
import sys
import traceback
from werkzeug.utils import secure_filename

# Add parent directory to path for config
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import BILT
from bilt import BILT
                    
from config import config, Config

def create_service_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    app.config['SERVICE_PORT'] = 5002
    
    Config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bilt_service.log'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    return app

app = create_service_app()
logger = logging.getLogger(__name__)

# Global state
current_model = None
model_info = {'name': None, 'classes': [], 'loaded': False}
camera_manager = None
detection_thread = None
detection_active = False
frame_lock = threading.Lock()
latest_frame = None
counter_triggered = {}

detection_settings = {
    'conf': app.config['DEFAULT_CONF_THRESHOLD'],
    'iou': app.config['DEFAULT_IOU_THRESHOLD'],
    'max_det': app.config['DEFAULT_MAX_DETECTIONS'],
    'classes': None,
    'counter_mode': False,
    'save_images': False,
    'dataset_capture': False,
    'project_folder': '',
    'chain_mode': False,
    'chain_steps': [],
    'chain_timeout': 5.0,
    'chain_auto_advance': True,
    'chain_pause_time': 10.0
}

object_counters = {}
detection_stats = {'total_detections': 0, 'fps': 0, 'last_detection_time': None}

# chain_state = {
#     'active': False, 'current_step': 0, 'step_start_time': None,
#     'completed_cycles': 0, 'failed_steps': 0, 'step_history': [],
#     'current_detections': {}, 'last_step_result': None,
#     'cycle_pause': False, 'cycle_pause_start': None
# }
chain_state = {
    'active': False, 
    'current_step': 0, 
    'step_start_time': None,
    'completed_cycles': 0, 
    'failed_steps': 0, 
    'step_history': [],
    'current_detections': {}, 
    'last_step_result': None,
    'cycle_pause': False, 
    'cycle_pause_start': None,
    'waiting_for_ack': False,
    'error_message': None,
    'wrong_object': None,
    'error_step': None
}
class RGBBalancer:
    def __init__(self):
        self.red_gain = 1.0
        self.green_gain = 1.0
        self.blue_gain = 1.0
    
    def set_gains(self, red, green, blue):
        self.red_gain = red / 128.0
        self.green_gain = green / 128.0
        self.blue_gain = blue / 128.0
    
    def apply(self, frame):
        if frame is None:
            return frame
        b, g, r = cv2.split(frame)
        r = np.clip(r * self.red_gain, 0, 255).astype(np.uint8)
        g = np.clip(g * self.green_gain, 0, 255).astype(np.uint8)
        b = np.clip(b * self.blue_gain, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

rgb_balancer = RGBBalancer()

# Import managers from separate file
from detection.bilt_managers import (
    EnhancedCameraManager, ModelManager, ChainDetectionManager,
    DetectionProcessor, ImageManager
)

camera_manager = EnhancedCameraManager()

def detection_loop():
    """Main detection loop using BILT"""
    global latest_frame, detection_active
    frame_time = 1.0 / app.config['FRAME_RATE_LIMIT']
    
    while True:
        if not detection_active:
            time.sleep(0.1)
            continue
        
        start_time = time.time()
        try:
            frame, fps = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            frame = rgb_balancer.apply(frame)
            
            if current_model and model_info['loaded']:
                # Convert frame to PIL Image for BILT
                from PIL import Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Run BILT detection
                bilt_detections = current_model.predict(
                    pil_image,
                    conf=detection_settings['conf'],
                    iou=detection_settings['iou']
                )
                
                detections, annotated_frame = DetectionProcessor.process_detections(
                    bilt_detections, frame, detection_settings, object_counters,
                    detection_stats, chain_state, counter_triggered, latest_frame, frame_lock
                )
                
                if detection_settings['save_images'] and detections and not detection_settings['chain_mode']:
                    ImageManager.save_detection_image(annotated_frame, detections, app.config)
                
                if detection_settings['dataset_capture'] and detections:
                    ImageManager.save_dataset_image(
                        frame, detections, detection_settings['project_folder'], app.config
                    )
            else:
                annotated_frame = frame
            
            with frame_lock:
                latest_frame = annotated_frame
                
        except Exception as e:
            logger.error(f"Detection loop error: {str(e)}")
            logger.error(traceback.format_exc())
        
        elapsed = time.time() - start_time
        time.sleep(max(0, frame_time - elapsed))

# API Endpoints
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'bilt_service', 'version': '1.0.0'})

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    try:
        cameras = camera_manager.get_available_cameras()
        return jsonify({'success': True, 'cameras': cameras})
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        models = ModelManager.get_available_models(app.config)
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/select', methods=['POST'])
def select_camera():
    try:
        data = request.json
        camera_index = data.get('camera_index', 0)
        success = camera_manager.initialize_camera(camera_index, app.config)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error selecting camera: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/info', methods=['GET'])
def get_camera_info():
    try:
        info = camera_manager.get_camera_info()
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/resolution', methods=['POST'])
def set_camera_resolution():
    try:
        data = request.json
        width = int(data.get('width', 1280))
        height = int(data.get('height', 960))
        app.config['DEFAULT_CAMERA_WIDTH'] = width
        app.config['DEFAULT_CAMERA_HEIGHT'] = height
        
        if camera_manager.cap:
            camera_manager.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera_manager.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        return jsonify({'success': True, 'width': width, 'height': height})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model/load', methods=['POST'])
def load_model():
    global current_model, model_info
    try:
        data = request.json
        model_name = data.get('model_name')
        success, result = ModelManager.load_model(model_name, app.config)
        
        if success:
            current_model = result['model']
            model_info = {
                'name': model_name,
                'classes': result['classes'],
                'loaded': True,
                'class_count': len(result['classes'])
            }
            return jsonify({'success': True, 'model_info': model_info})
        else:
            return jsonify({'success': False, 'error': result})
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    return jsonify({'success': True, 'model_info': model_info})

@app.route('/api/detection/settings', methods=['GET', 'POST'])
def detection_settings_endpoint():
    global detection_settings
    try:
        if request.method == 'POST':
            data = request.json
            detection_settings.update(data)
            logger.info(f"Updated detection settings: {data}")
            return jsonify({'success': True, 'settings': detection_settings})
        return jsonify({'success': True, 'settings': detection_settings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    global detection_active, detection_thread
    try:
        if not detection_active:
            detection_active = True
            if detection_thread is None or not detection_thread.is_alive():
                detection_thread = threading.Thread(target=detection_loop, daemon=True)
                detection_thread.start()
            logger.info("Detection started")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    global detection_active
    try:
        detection_active = False
        logger.info("Detection stopped")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detection/stats', methods=['GET'])
def get_detection_stats():
    return jsonify({'success': True, 'stats': detection_stats})

@app.route('/api/counters', methods=['GET'])
def get_counters():
    return jsonify({'success': True, 'counters': object_counters})

@app.route('/api/counters/reset', methods=['POST'])
def reset_counters():
    global object_counters, counter_triggered
    try:
        object_counters = {}
        counter_triggered = {}
        detection_stats['total_detections'] = 0
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chain/status', methods=['GET'])
def get_chain_status():
    try:
        status = ChainDetectionManager.get_chain_status(
            detection_settings, chain_state
        )
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chain/control', methods=['POST'])
def chain_control():
    global chain_state
    try:
        data = request.json
        action = data.get('action')
        
        if action == 'start':
            detection_settings['chain_mode'] = True
            ChainDetectionManager.initialize_chain(chain_state)
            return jsonify({'success': True})
        elif action == 'stop':
            detection_settings['chain_mode'] = False
            chain_state['active'] = False
            return jsonify({'success': True})
        elif action == 'reset':
            ChainDetectionManager.reset_chain(chain_state)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chain/config', methods=['GET', 'POST'])
def chain_config():
    try:
        if request.method == 'POST':
            data = request.json
            if 'chain_steps' in data:
                detection_settings['chain_steps'] = data['chain_steps']
            if 'chain_timeout' in data:
                detection_settings['chain_timeout'] = float(data['chain_timeout'])
            if 'chain_auto_advance' in data:
                detection_settings['chain_auto_advance'] = data['chain_auto_advance']
            if 'chain_pause_time' in data:
                detection_settings['chain_pause_time'] = float(data['chain_pause_time'])
            return jsonify({'success': True, 'config': {
                'chain_steps': detection_settings['chain_steps'],
                'chain_timeout': detection_settings['chain_timeout'],
                'chain_auto_advance': detection_settings['chain_auto_advance'],
                'chain_pause_time': detection_settings['chain_pause_time']
            }})
        
        return jsonify({'success': True, 'config': {
            'chain_steps': detection_settings['chain_steps'],
            'chain_timeout': detection_settings['chain_timeout'],
            'chain_auto_advance': detection_settings['chain_auto_advance'],
            'chain_pause_time': detection_settings['chain_pause_time']
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chain/acknowledge_error', methods=['POST'])
def acknowledge_error():
    global chain_state
    try:
        if ChainDetectionManager.acknowledge_error(chain_state):
            return jsonify({'success': True, 'message': 'Error acknowledged, restarting from error step'})
        return jsonify({'success': False, 'message': 'No error to acknowledge'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/frame/latest', methods=['GET'])
def get_latest_frame():
    """Get latest frame as JPEG"""
    try:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Apply sharpening
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel_sharp)
        
        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 95,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        else:
            return jsonify({'success': False, 'error': 'Failed to encode frame'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/projects', methods=['GET'])
def get_projects():
    try:
        projects = []
        if os.path.exists(app.config['PROJECTS_DIR']):
            for item in os.listdir(app.config['PROJECTS_DIR']):
                project_path = os.path.join(app.config['PROJECTS_DIR'], item)
                if os.path.isdir(project_path):
                    projects.append(item)
        return jsonify({'success': True, 'projects': projects})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/projects/create', methods=['POST'])
def create_project():
    try:
        data = request.json
        project_name = secure_filename(data.get('project_name', '').strip())
        
        if not project_name:
            return jsonify({'success': False, 'error': 'Project name required'})
        
        project_path = os.path.join(app.config['PROJECTS_DIR'], project_name)
        if os.path.exists(project_path):
            return jsonify({'success': False, 'error': 'Project already exists'})
        
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(project_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(project_path, 'labels'), exist_ok=True)
        
        project_info = {
            'name': project_name,
            'created': datetime.now().isoformat(),
            'description': data.get('description', ''),
            'classes': data.get('classes', [])
        }
        
        with open(os.path.join(project_path, 'project_info.json'), 'w') as f:
            json.dump(project_info, f, indent=2)
        
        return jsonify({'success': True, 'project': project_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chains/saved', methods=['GET'])
def get_saved_chains():
    try:
        chains_dir = os.path.join(app.config['BASE_DIR'], 'chains')
        chains = []
        
        if os.path.exists(chains_dir):
            for file in os.listdir(chains_dir):
                if file.endswith('.json'):
                    filepath = os.path.join(chains_dir, file)
                    try:
                        with open(filepath, 'r') as f:
                            chain_data = json.load(f)
                        chains.append({
                            'name': chain_data.get('name', file[:-5]),
                            'model_name': chain_data.get('model_name', 'Unknown'),
                            'steps': len(chain_data.get('chain_steps', [])),
                            'created': chain_data.get('created', ''),
                            'filename': file[:-5]
                        })
                    except Exception as e:
                        logger.error(f"Error reading chain file {file}: {str(e)}")
        
        return jsonify({'success': True, 'chains': chains})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chains/save', methods=['POST'])
def save_chain():
    try:
        data = request.json
        chain_name = secure_filename(data.get('chain_name', '').strip())
        model_name = data.get('model_name', '')
        
        if not chain_name:
            return jsonify({'success': False, 'error': 'Chain name required'})
        
        chains_dir = os.path.join(app.config['BASE_DIR'], 'chains')
        os.makedirs(chains_dir, exist_ok=True)
        
        chain_data = {
            'name': chain_name,
            'model_name': model_name,
            'created': datetime.now().isoformat(),
            'chain_steps': detection_settings['chain_steps'],
            'chain_timeout': detection_settings['chain_timeout'],
            'chain_auto_advance': detection_settings['chain_auto_advance'],
            'chain_pause_time': detection_settings['chain_pause_time']
        }
        
        filepath = os.path.join(chains_dir, f"{chain_name}.json")
        with open(filepath, 'w') as f:
            json.dump(chain_data, f, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chains/load', methods=['POST'])
def load_chain():
    try:
        data = request.json
        chain_name = data.get('chain_name', '')
        
        chains_dir = os.path.join(app.config['BASE_DIR'], 'chains')
        filepath = os.path.join(chains_dir, f"{secure_filename(chain_name)}.json")
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Chain not found'})
        
        with open(filepath, 'r') as f:
            chain_data = json.load(f)
        
        detection_settings['chain_steps'] = chain_data['chain_steps']
        detection_settings['chain_timeout'] = chain_data.get('chain_timeout', 5.0)
        detection_settings['chain_auto_advance'] = chain_data.get('chain_auto_advance', True)
        detection_settings['chain_pause_time'] = chain_data.get('chain_pause_time', 10.0)
        
        return jsonify({'success': True, 'chain_data': chain_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chains/delete', methods=['POST'])
def delete_chain():
    try:
        data = request.json
        chain_name = data.get('chain_name', '')
        
        chains_dir = os.path.join(app.config['BASE_DIR'], 'chains')
        filepath = os.path.join(chains_dir, f"{secure_filename(chain_name)}.json")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Chain not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

import atexit

def cleanup():
    global detection_active
    detection_active = False
    if camera_manager:
        camera_manager.release()
    logger.info("Service cleanup completed")

atexit.register(cleanup)

if __name__ == '__main__':
    logger.info("Starting BILT Service on port 5002")
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    app.run(host='127.0.0.1', port=5002, debug=False, threaded=True, use_reloader=False)