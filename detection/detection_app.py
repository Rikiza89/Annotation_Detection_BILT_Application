"""
Detection App - Web interface for BILT detection
Flask application that uses yolo_client.py to communicate with yolo_service.py
Run this on port 5001: python detection_app.py
"""

from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import logging
import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import config, Config
from detection.bilt_client import BILTClient, check_bilt_service

if getattr(sys, 'frozen', False):
    # Parent of the folder containing the executable
    BASE_DIR = os.path.dirname(os.path.dirname(sys.executable))

    template_folder = os.path.join(sys._MEIPASS, 'web_templates')
    static_folder = os.path.join(sys._MEIPASS, 'web_static')
else:
    # Parent of the folder containing this file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    template_folder = 'templates'
    static_folder = 'static'

os.chdir(BASE_DIR)

projects_dir = os.path.join(BASE_DIR, "projects") 
    
def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    Config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('detection_app.log'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    return app

app = create_app()
logger = logging.getLogger(__name__)

# Initialize BILT client
bilt_client = BILTClient("http://127.0.0.1:5002")

def generate_frames():
    """Generate video frames for streaming from BILT service"""
    import time
    
    while True:
        try:
            frame_bytes = bilt_client.get_latest_frame()
            
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send placeholder frame if service unavailable
                import cv2
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for YOLO Service...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            logger.error(f"Frame generation error: {str(e)}")
        
        time.sleep(0.033)  # ~30 FPS

# Web routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index_detection_segment.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Service health check
@app.route('/api/service/health')
def service_health():
    """Check if BILT service is available"""
    is_available = check_bilt_service()
    return jsonify({
        'success': True,
        'service_available': is_available
    })

# Camera endpoints
@app.route('/api/cameras')
def get_cameras():
    """Get available cameras"""
    try:
        result = bilt_client.get_cameras()
        if result.get('success'):
            camera_indices = [cam['index'] for cam in result.get('cameras', [])]
            return jsonify(camera_indices)
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        return jsonify([])

@app.route('/api/select_camera', methods=['POST'])
def select_camera():
    """Select camera"""
    try:
        data = request.json
        camera_index = data.get('camera_index', 0)
        result = bilt_client.select_camera(camera_index)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error selecting camera: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/set_image_size', methods=['POST'])
def set_image_size():
    """Set camera resolution"""
    try:
        data = request.get_json()
        width = int(data.get('width', 1280))
        height = int(data.get('height', 960))
        result = bilt_client.set_camera_resolution(width, height)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Model endpoints
@app.route('/api/models')
def get_models():
    """Get available models"""
    try:
        result = bilt_client.get_models()
        if result.get('success'):
            model_names = [m['name'] for m in result.get('models', [])]
            return jsonify(model_names)
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify([])

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load model"""
    try:
        data = request.json
        model_name = data.get('model_name')
        result = bilt_client.load_model(model_name)
        
        if result.get('success'):
            model_info = result.get('model_info', {})
            return jsonify({
                'success': True,
                'classes': model_info.get('classes', [])
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error')
            })
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Detection endpoints
@app.route('/api/detection_settings', methods=['GET', 'POST'])
def detection_settings():
    """Get or update detection settings"""
    try:
        if request.method == 'POST':
            data = request.json
            result = bilt_client.update_detection_settings(data)
            return jsonify(result)
        else:
            result = bilt_client.get_detection_settings()
            if result.get('success'):
                return jsonify(result.get('settings', {}))
            return jsonify({})
    except Exception as e:
        logger.error(f"Error handling detection settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection"""
    try:
        result = bilt_client.start_detection()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error starting detection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    try:
        result = bilt_client.stop_detection()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Counter endpoints
@app.route('/api/counters')
def get_counters():
    """Get object counters"""
    try:
        result = bilt_client.get_counters()
        if result.get('success'):
            return jsonify(result.get('counters', {}))
        return jsonify({})
    except Exception as e:
        logger.error(f"Error getting counters: {str(e)}")
        return jsonify({})

@app.route('/api/reset_counters', methods=['POST'])
def reset_counters():
    """Reset object counters"""
    try:
        result = bilt_client.reset_counters()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error resetting counters: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Chain detection endpoints
@app.route('/api/chain_status')
def get_chain_status():
    """Get chain detection status"""
    try:
        result = bilt_client.get_chain_status()
        if result.get('success'):
            return jsonify(result.get('status', {}))
        return jsonify({'active': False, 'error': result.get('error')})
    except Exception as e:
        logger.error(f"Error getting chain status: {str(e)}")
        return jsonify({'active': False, 'error': str(e)})

@app.route('/api/chain_control', methods=['POST'])
def chain_control():
    """Control chain detection"""
    try:
        data = request.json
        action = data.get('action')
        result = bilt_client.chain_control(action)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in chain control: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chain_config', methods=['GET', 'POST'])
def chain_config():
    """Get or update chain configuration"""
    try:
        if request.method == 'POST':
            data = request.json
            result = bilt_client.update_chain_config(data)
            return jsonify(result)
        else:
            result = bilt_client.get_chain_config()
            if result.get('success'):
                return jsonify(result.get('config', {}))
            return jsonify({})
    except Exception as e:
        logger.error(f"Error in chain config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/acknowledge_error', methods=['POST'])
def acknowledge_error():
    """Acknowledge error in chain detection"""
    try:
        result = bilt_client.acknowledge_error()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error acknowledging error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Chain save/load endpoints
@app.route('/api/saved_chains')
def get_saved_chains():
    """Get list of saved chains"""
    try:
        result = bilt_client.get_saved_chains()
        if result.get('success'):
            return jsonify(result.get('chains', []))
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting saved chains: {str(e)}")
        return jsonify([])

@app.route('/api/save_chain', methods=['POST'])
def save_chain():
    """Save chain configuration"""
    try:
        data = request.json
        chain_name = data.get('chain_name', '')
        model_name = data.get('model_name', '')
        result = bilt_client.save_chain(chain_name, model_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error saving chain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_chain', methods=['POST'])
def load_chain():
    """Load chain configuration"""
    try:
        data = request.json
        chain_name = data.get('chain_name', '')
        result = bilt_client.load_chain(chain_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading chain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_chain', methods=['POST'])
def delete_chain():
    """Delete saved chain"""
    try:
        data = request.json
        chain_name = data.get('chain_name', '')
        result = bilt_client.delete_chain(chain_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error deleting chain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Project endpoints
@app.route('/api/projects')
def get_projects():
    """Get available projects"""
    try:
        result = bilt_client.get_projects()
        if result.get('success'):
            return jsonify(result.get('projects', []))
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting projects: {str(e)}")
        return jsonify([])

@app.route('/api/create_project', methods=['POST'])
def create_project():
    """Create new project"""
    try:
        data = request.json
        project_name = data.get('project_name', '').strip()
        description = data.get('description', '')
        classes = data.get('classes', [])
        
        result = bilt_client.create_project(project_name, description, classes)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Detection Web Application on port 5001")

    # Check if BILT service is available
    if check_bilt_service():
        logger.info("BILT Service is available at http://127.0.0.1:5002")
    else:
        logger.warning("BILT Service is not available. Please start bilt_service.py first!")
        logger.warning("Run: python bilt_service.py")
    
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
        threaded=True,
        use_reloader=False
    )