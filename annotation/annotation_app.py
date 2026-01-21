import os
import json
import cv2
import base64
from datetime import datetime
import sys
import eventlet

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_socketio import SocketIO, emit

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from annotation.bilt_client import BILTClient

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

app = Flask(
    __name__,
    template_folder=template_folder,
    static_folder=static_folder
)
app.config['SECRET_KEY'] = 'your-secret-key'

socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*"
)

# Initialize BILT client (connects to BILT service)
bilt_client = BILTClient(base_url='http://127.0.0.1:5001')

current_project = None
camera_manager = None

class ProjectManager:
    """Manages BILT project structure without BILT-specific logic"""
    def __init__(self, project_name, project_path):
        self.name = project_name
        self.path = project_path
        self.classes_file = os.path.join(project_path, "classes.txt")
        self.data_yaml = os.path.join(project_path, "data.yaml")
 
    def get_task_type(self):
        """Get current task type (detect, segment, obb)"""
        config_file = os.path.join(self.path, "project_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('task_type', 'detect')
        return 'detect'

    def set_task_type(self, task_type):
        """Set task type"""
        if task_type not in ['detect', 'segment', 'obb']:
            return False
        config_file = os.path.join(self.path, "project_config.json")
        config = {'task_type': task_type}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        config['task_type'] = task_type
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True 
        
    def create_structure(self):
        """Create BILT project folder structure"""
        folders = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for folder in folders:
            os.makedirs(os.path.join(self.path, folder), exist_ok=True)
        
        if not os.path.exists(self.classes_file):
            with open(self.classes_file, 'w') as f:
                f.write("object\n")
        
        config_file = os.path.join(self.path, "project_config.json")
        if not os.path.exists(config_file):
            with open(config_file, 'w') as f:
                json.dump({'task_type': 'detect'}, f, indent=2)
        
        self.update_data_yaml()
    
    def update_data_yaml(self):
        """Update data.yaml file"""
        import yaml
        classes = self.get_classes()
        data = {
            'train': os.path.abspath(os.path.join(self.path, 'train/images')),
            'val': os.path.abspath(os.path.join(self.path, 'val/images')),
            'nc': len(classes),
            'names': classes
        }
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_classes(self):
        """Read classes from classes.txt"""
        if os.path.exists(self.classes_file):
            with open(self.classes_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []
    
    def save_classes(self, classes):
        """Save classes to classes.txt"""
        with open(self.classes_file, 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
        self.update_data_yaml()
    
    def get_images(self, split='train'):
        """Get list of images in train/val folder"""
        images_dir = os.path.join(self.path, f'{split}/images')
        if not os.path.exists(images_dir):
            return []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = []
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(valid_extensions):
                images.append({
                    'filename': img_file,
                    'path': os.path.join(images_dir, img_file),
                    'has_labels': self.has_labels(img_file, split)
                })
        return sorted(images, key=lambda x: x['filename'])
    
    def has_labels(self, image_filename, split='train'):
        """Check if image has corresponding label file"""
        label_name = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.path, f'{split}/labels', label_name)
        return os.path.exists(label_path)
    
    def get_labels(self, image_filename, split='train'):
        """Get BILT labels for an image (supports detect, segment, obb)"""
        label_name = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.path, f'{split}/labels', label_name)
        
        if not os.path.exists(label_path):
            return []
        
        task_type = self.get_task_type()
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                class_id = int(parts[0])
                
                if task_type == 'detect':
                    if len(parts) >= 5:
                        labels.append({
                            'class_id': class_id,
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4]),
                            'type': 'detect'
                        })
                
                elif task_type == 'segment':
                    if len(parts) >= 7:
                        points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                points.append({
                                    'x': float(parts[i]),
                                    'y': float(parts[i + 1])
                                })
                        if len(points) >= 3:
                            labels.append({
                                'class_id': class_id,
                                'points': points,
                                'type': 'segment'
                            })
                
                elif task_type == 'obb':
                    if len(parts) == 9:
                        points = []
                        for i in range(1, 9, 2):
                            points.append({
                                'x': float(parts[i]),
                                'y': float(parts[i + 1])
                            })
                        labels.append({
                            'class_id': class_id,
                            'points': points,
                            'type': 'obb'
                        })
        
        return labels
    
    def save_labels(self, image_filename, labels, split='train'):
        """Save BILT labels for an image (supports detect, segment, obb)"""
        label_name = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.path, f'{split}/labels', label_name)
        
        task_type = self.get_task_type()
        
        with open(label_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                
                if task_type == 'detect':
                    f.write(f"{class_id} {label['x_center']} {label['y_center']} {label['width']} {label['height']}\n")
                
                elif task_type == 'segment':
                    if 'points' in label and len(label['points']) >= 3:
                        points_str = ' '.join([f"{p['x']} {p['y']}" for p in label['points']])
                        f.write(f"{class_id} {points_str}\n")
                
                elif task_type == 'obb':
                    if 'points' in label and len(label['points']) == 4:
                        points_str = ' '.join([f"{p['x']} {p['y']}" for p in label['points']])
                        f.write(f"{class_id} {points_str}\n")


class CameraManager:
    def __init__(self):
        self.camera = None
        self.active = False
        self.color_mode = 'bgr'

    def _detect_color_issue(self, frame):
        """Detect if frame has color issues by analyzing a test frame"""
        if frame is None or len(frame.shape) != 3:
            return None
        
        b_avg = frame[:,:,0].mean()
        g_avg = frame[:,:,1].mean()
        r_avg = frame[:,:,2].mean()
        
        if b_avg > r_avg * 1.5 and b_avg > 100:
            return 'yuv2bgr'
        elif r_avg > b_avg * 1.5 and r_avg > 100:
            return 'rgb2bgr'
        
        return None

    def start_camera(self, camera_id=0, resolution='1080p'):
        """Start camera capture with specified resolution"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if self.camera.isOpened():
                resolution_map = {
                    '4k': (3840, 2160),
                    '1080p': (1920, 1080),
                    '720p': (1280, 720),
                    '480p': (640, 480),
                    '360p': (480, 360)
                }
                
                width, height = resolution_map.get(resolution, (1920, 1080))
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if resolution in ['4k', '1080p']:
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"Camera resolution set to: {actual_width}x{actual_height}")
                
                is_4k = (actual_width >= 3840 or actual_height >= 2160)
                is_high_res = (actual_width >= 1920 or actual_height >= 1080)
                
                if is_4k or is_high_res:
                    self.color_mode = 'bgr'
                else:
                    ret, test_frame = self.camera.read()
                    if ret:
                        color_issue = self._detect_color_issue(test_frame)
                        if color_issue:
                            self.color_mode = color_issue
                            print(f"Color correction needed: {color_issue}")
                        else:
                            self.color_mode = 'bgr'
                
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set default camera properties
                default_brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
                default_contrast = self.camera.get(cv2.CAP_PROP_CONTRAST)
                default_exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
                default_autowb = self.camera.get(cv2.CAP_PROP_AUTO_WB)
                default_saturation = self.camera.get(cv2.CAP_PROP_SATURATION)
                default_hue = self.camera.get(cv2.CAP_PROP_HUE)
                default_temp = self.camera.get(cv2.CAP_PROP_WB_TEMPERATURE)
                
                if default_autowb != -1:
                    self.camera.set(cv2.CAP_PROP_AUTO_WB, default_autowb)
                if default_exposure != -1:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, default_exposure)
                if default_brightness != -1:
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, default_brightness)
                if default_contrast != -1:
                    self.camera.set(cv2.CAP_PROP_CONTRAST, default_contrast)
                if default_saturation != -1:
                    self.camera.set(cv2.CAP_PROP_SATURATION, default_saturation)
                if default_hue != -1:
                    self.camera.set(cv2.CAP_PROP_HUE, default_hue)
                if default_temp != -1:
                    self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, default_temp)

                self.active = True
                return True
        except Exception as e:
            print(f"Error starting camera: {e}")
        return False
        
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.active = False
    
    def get_available_cameras(self):
        """Get list of available camera indices"""
        cameras = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras.append(i)
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
        return cameras
    
    def set_resolution(self, width, height):
        """Set camera resolution"""
        if self.camera and self.active:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_color_mode(self, mode):
        """Set color correction mode"""
        if mode in ['bgr', 'swap', 'rgb2bgr']:
            self.color_mode = mode
            return True
        return False
            
    def capture_frame(self):
        """Capture a single frame with color correction"""
        if self.camera and self.active:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                if self.color_mode == 'swap':
                    frame = frame[:, :, ::-1]
                elif self.color_mode == 'rgb2bgr':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
        return None

camera_manager = CameraManager()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/projects')
def projects():
    """List all projects"""
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir)
    
    project_list = []
    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)
        if os.path.isdir(project_path):
            project = ProjectManager(project_name, project_path)
            train_images = len(project.get_images('train'))
            val_images = len(project.get_images('val'))
            project_list.append({
                'name': project_name,
                'train_images': train_images,
                'val_images': val_images,
                'classes': len(project.get_classes())
            })
    
    return render_template('projects.html', projects=project_list)

@app.route('/create_project', methods=['POST'])
def create_project():
    """Create new BILT project"""
    project_name = request.json.get('name')
    if not project_name:
        return jsonify({'error': 'Project name required'}), 400
    
    project_path = os.path.join(projects_dir, project_name)
    if os.path.exists(project_path):
        return jsonify({'error': 'Project already exists'}), 400
    
    project = ProjectManager(project_name, project_path)
    project.create_structure()
    
    return jsonify({'success': True, 'message': 'Project created successfully'})

@app.route('/load_project/<project_name>')
def load_project(project_name):
    """Load existing project"""
    global current_project
    project_path = os.path.join(projects_dir, project_name)
    
    if not os.path.exists(project_path):
        return redirect(url_for('projects'))
    
    current_project = ProjectManager(project_name, project_path)
    return redirect(url_for('workspace'))

@app.route('/workspace')
def workspace():
    """Main workspace for annotation"""
    if not current_project:
        return redirect(url_for('projects'))
    
    return render_template('workspace_segmentation.html', project=current_project, current_project=current_project)

@app.route('/api/camera/available')
def get_available_cameras():
    """Get available cameras"""
    cameras = camera_manager.get_available_cameras()
    return jsonify({'cameras': cameras})

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera"""
    camera_id = request.json.get('camera_id', 0)
    resolution = request.json.get('resolution', '1080p')
    if camera_manager.start_camera(camera_id, resolution):
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to start camera'}), 400

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera"""
    camera_manager.stop_camera()
    return jsonify({'success': True})

@app.route('/api/camera/capture', methods=['POST'])
def capture_image():
    """Capture image from camera"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    frame = camera_manager.capture_frame()
    if frame is None:
        return jsonify({'error': 'Failed to capture image'}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"img_{timestamp}.jpg"
    
    split = request.json.get('split', 'train')
    image_path = os.path.join(current_project.path, f'{split}/images', filename)
    
    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 99])
    
    return jsonify({
        'success': True, 
        'filename': filename,
        'message': f'Image saved to {split} folder'
    })

@app.route('/api/project/images/<split>')
def get_project_images(split):
    """Get images from project"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    images = current_project.get_images(split)
    return jsonify({'images': images})

@app.route('/api/project/classes')
def get_project_classes():
    """Get project classes"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    classes = current_project.get_classes()
    return jsonify({'classes': classes})

@app.route('/api/project/classes', methods=['POST'])
def save_project_classes():
    """Save project classes"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    classes = request.json.get('classes', [])
    current_project.save_classes(classes)
    
    return jsonify({'success': True})

@app.route('/api/labels/<split>/<filename>')
def get_labels(split, filename):
    """Get labels for an image"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    labels = current_project.get_labels(filename, split)
    return jsonify({'labels': labels})

@app.route('/api/labels/<split>/<filename>', methods=['POST'])
def save_labels(split, filename):
    """Save labels for an image"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    labels = request.json.get('labels', [])
    current_project.save_labels(filename, labels, split)
    
    return jsonify({'success': True})

@app.route('/api/models/available')
def get_available_models():
    """Get locally available BILT models - delegates to BILT service"""
    task_type = current_project.get_task_type() if current_project else 'detect'
    result = bilt_client.get_available_models(task_type)
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start BILT training - delegates to BILT service"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    config = request.json
    config['data_yaml'] = current_project.data_yaml
    config['project_path'] = current_project.path
    config['task_type'] = current_project.get_task_type()

    result = bilt_client.start_training(config)
    
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@app.route('/api/training/config', methods=['GET', 'POST'])
def training_config():
    """Save/load training configuration"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    config_file = os.path.join(current_project.path, 'training_config.json')
    
    if request.method == 'POST':
        config = request.json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({'success': True})
    else:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            return jsonify(config)
        return jsonify({})

@app.route('/api/training/status')
def training_status():
    """Get training progress status - delegates to BILT service"""
    result = bilt_client.get_training_status()
    return jsonify(result)

@app.route('/images/<split>/<filename>')
def serve_image(split, filename):
    """Serve project images"""
    if not current_project:
        return "No project loaded", 404
    
    image_dir = os.path.join(current_project.path, f'{split}/images')
    return send_from_directory(image_dir, filename)

@socketio.on('start_camera_feed')
def handle_camera_feed(data):
    """Handle real-time camera feed"""
    camera_id = data.get('camera_id', 0)
    resolution = data.get('resolution', '1080p')
    
    if not camera_manager.start_camera(camera_id, resolution):
        emit('camera_error', {'error': 'Failed to start camera'})
        return
    
    def stream_frames():
        """Background task to stream frames"""
        print("Starting camera feed stream...")
        frame_count = 0
        while camera_manager.active:
            try:
                frame = camera_manager.capture_frame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    
                    max_dimension = 800
                    if width > height:
                        if width > max_dimension:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        else:
                            new_width = width
                            new_height = height
                    else:
                        if height > max_dimension:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))
                        else:
                            new_width = width
                            new_height = height
                    
                    display_frame = cv2.resize(frame, (new_width, new_height))
                    
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    _, buffer = cv2.imencode('.jpg', display_frame, encode_param)
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                    socketio.emit('camera_frame', {
                        'frame': frame_data,
                        'width': new_width,
                        'height': new_height
                    })
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"Streamed {frame_count} frames")
                    
                socketio.sleep(0.033)
            except Exception as e:
                print(f"Error streaming frame: {e}")
                break
        
        print("Camera feed stream stopped")
    
    socketio.start_background_task(stream_frames)
    emit('camera_started', {'message': 'Camera feed started'})

@socketio.on('stop_camera_feed')
def handle_stop_camera():
    """Stop camera feed"""
    print("Stopping camera feed...")
    camera_manager.stop_camera()
    emit('camera_stopped', {'message': 'Camera feed stopped'})

@app.route('/api/autotrain/check_model')
def check_autotrain_model():
    """Check for all best.pth models in project"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    models = []
    for root, dirs, files in os.walk(current_project.path):
        if 'best.pth' in files:
            model_path = os.path.join(root, 'best.pth')
            mtime = os.path.getmtime(model_path)
            created_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            
            models.append({
                'path': model_path,
                'relative_path': os.path.relpath(model_path, current_project.path),
                'created_date': created_date,
                'timestamp': mtime,
                'size_mb': round(file_size, 2)
            })
    
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({
        'has_model': len(models) > 0,
        'models': models
    })

@app.route('/api/autotrain/config', methods=['GET', 'POST'])
def autotrain_config():
    """Get/Save autotrain configuration"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    config_file = os.path.join(current_project.path, 'autotrain_config.json')
    
    if request.method == 'POST':
        config = request.json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({'success': True})
    else:
        default_config = {
            'model': 'best.pt',
            'epochs': 50,
            'imgsz': 640,
            'batch': 16,
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'patience': 50,
            'backup_enabled': True
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            return jsonify(config)
        return jsonify(default_config)

@app.route('/api/autotrain/start', methods=['POST'])
def start_autotrain():
    """Start autotraining process - delegates to BILT service"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    config = request.json
    config['project_path'] = current_project.path
    config['data_yaml'] = current_project.data_yaml
    config['task_type'] = current_project.get_task_type()
    
    result = bilt_client.start_autotrain(config)
    
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@app.route('/api/autotrain/status')
def autotrain_status():
    """Check if autotrain is running - delegates to BILT service"""
    result = bilt_client.get_autotrain_status()
    return jsonify(result)

@app.route('/api/relabel/models')
def get_relabel_models():
    """Get all available best.pt models from project"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    models = []
    for root, dirs, files in os.walk(current_project.path):
        if 'best.pth' in files:
            model_path = os.path.join(root, 'best.pth')
            mtime = os.path.getmtime(model_path)
            created_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            
            models.append({
                'path': model_path,
                'name': os.path.relpath(model_path, current_project.path),
                'created_date': created_date,
                'timestamp': mtime,
                'size_mb': round(file_size, 2),
                'source': 'project'
            })
    
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'models': models})

@app.route('/api/relabel/add_external_model', methods=['POST'])
def add_external_model():
    """Validate external model path"""
    model_path = request.json.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Invalid model path'}), 400
    
    if not model_path.endswith('.pth'):
        return jsonify({'error': 'Must be a .pth file'}), 400
    
    try:
        mtime = os.path.getmtime(model_path)
        created_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        
        return jsonify({
            'success': True,
            'model': {
                'path': model_path,
                'name': os.path.basename(model_path),
                'created_date': created_date,
                'size_mb': round(file_size, 2),
                'source': 'external'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/relabel/start', methods=['POST'])
def start_relabel():
    """Start relabeling process - delegates to BILT service"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    config = request.json
    config['project_path'] = current_project.path
    config['task_type'] = current_project.get_task_type()
    
    result = bilt_client.start_relabel(config)
    
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@app.route('/api/project/task_type', methods=['GET', 'POST'])
def project_task_type():
    """Get or set project task type"""
    if not current_project:
        return jsonify({'error': 'No project loaded'}), 400
    
    if request.method == 'POST':
        task_type = request.json.get('task_type')
        if current_project.set_task_type(task_type):
            return jsonify({'success': True, 'task_type': task_type})
        return jsonify({'error': 'Invalid task type'}), 400
    else:
        return jsonify({'task_type': current_project.get_task_type()})

if __name__ == '__main__':
    os.makedirs(projects_dir, exist_ok=True)

    HOST = '127.0.0.1'
    PORT = 5000
    
    print(f"Starting annotation app on http://{HOST}:{PORT}")
    print(f"BILT service should be running on http://127.0.0.1:5001")
    
    listener = eventlet.listen((HOST, PORT))
    eventlet.wsgi.server(listener, app)