#!/usr/bin/env python3
"""
BILT Service - AGPL-3.0 Licensed Component

This service handles all detection operations including:
- Model management and validation
- Training operations
- Auto-labeling and prediction
- Model inference

License: AGPL-3.0
"""

import os
import shutil
import threading
from datetime import datetime
from flask import Flask, request, jsonify
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bilt import BILT

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bilt-service-secret-key'

# Global state for training/autotrain processes
training_active = False
autotrain_active = False
training_thread = None
autotrain_thread = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class BILTModelManager:
    """Manages BILT model operations"""
    
    @staticmethod
    def get_available_models(task_type='detect'):
        """Get locally available BILT models + scratch option"""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        
        models = []

        models.append({
            'name': 'scratch',
            'compatible': True,
            'description': 'Train new model from scratch'
        })
        
        if os.path.exists(MODELS_DIR):
            for file in os.listdir(MODELS_DIR):
                if file.endswith('.pth'):
                    # BILT only supports detection (no segment/obb variants)
                    is_compatible = (task_type == 'detect')
                    
                    models.append({
                        'name': file,
                        'compatible': is_compatible
                    })
        
        return {'models': models}

    @staticmethod
    def validate_model(model_name, task_type):
        """Validate model - allow None, 'scratch', or existing .pth"""
        
        # BILT only supports detection
        if task_type != 'detect':
            return False, f'BILT only supports detection task (requested: {task_type})'
        
        # Accept None or 'scratch' as train-from-scratch
        if model_name is None or model_name == 'scratch':
            return True, 'scratch'
        
        # From here on, model_name is guaranteed to be a string
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            return False, f'Model {model_name} not found in models folder'
        
        return True, model_path

class BILTTrainer:
    """Handles BILT training operations"""
    
    @staticmethod
    def train_model(config):
        """Execute BILT training"""
        global training_active
        
        try:
            training_active = True
            
            model_name = config.get("model")
            task_type = config.get("task_type", "detect")
            
            # BILT only supports detection
            if task_type != 'detect':
                print(f"Error: BILT only supports detection, not {task_type}")
                return

            # Treat None and 'scratch' as train-from-scratch
            if model_name is None or model_name == 'scratch':
                print("Training new model from scratch")
                model = BILT()
            else:
                model_path = os.path.join(MODELS_DIR, model_name)

                if os.path.exists(model_path):
                    print(f"Loading pretrained model: {model_path}")
                    model = BILT(model_path)
                else:
                    print(f"Model {model_name} not found, training from scratch")
                    model = BILT()
            
            # Train with BILT
            train_params = {
                'dataset': os.path.dirname(config.get('data_yaml')),
                'epochs': int(config.get("epochs", 100)),
                'batch_size': int(config.get("batch", 16)),
                'img_size': int(config.get("imgsz", 640)),
                'learning_rate': float(config.get("lr0", 0.01)),
                'device': config.get("device", "cpu"),
                'save_dir': config.get('project_path'),
                'name': f'training_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
            
            results = model.train(**train_params)
            
            # Copy best.pth to models folder with custom name
            custom_model_name = config.get("custom_model_name")
            if custom_model_name:
                # Ensure it has .pth extension
                if not custom_model_name.endswith('.pth'):
                    custom_model_name += '.pth'
                    
                best_pth_path = os.path.join(
                    config.get('project_path'), 
                    train_params['name'], 
                    'weights', 
                    'best.pth'
                )
                if os.path.exists(best_pth_path):
                    custom_path = os.path.join(MODELS_DIR, custom_model_name)
                    shutil.copy2(best_pth_path, custom_path)
                    print(f"Saved best.pth as {custom_model_name} in models folder")
            
            print("Training completed successfully")
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            training_active = False


class BILTAutoLabeler:
    """Handles auto-labeling operations"""
    
    @staticmethod
    def auto_label_images(model_path, config):
        """Auto-detect and create labels for all images"""
        project_path = config.get('project_path')
        task_type = config.get('task_type', 'detect')
        
        # BILT only supports detection
        if task_type != 'detect':
            print(f"Error: BILT only supports detection, not {task_type}")
            return
        
        model = BILT(model_path)
        
        conf_threshold = float(config.get('conf_threshold', 0.25))
        iou_threshold = float(config.get('iou_threshold', 0.45))
        
        for split in ['train', 'val']:
            images_dir = os.path.join(project_path, f'{split}/images')
            labels_dir = os.path.join(project_path, f'{split}/labels')
            
            if not os.path.exists(images_dir):
                continue
            
            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(images_dir, img_file)
                
                # Get predictions from BILT
                detections = model.predict(
                    img_path,
                    conf=conf_threshold,
                    iou=iou_threshold
                )
                
                label_name = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                # Convert BILT detections to YOLO format
                from PIL import Image
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'w') as f:
                    for det in detections:
                        cls_id = det['class_id']
                        x1, y1, x2, y2 = det['bbox']
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
    
    @staticmethod
    def backup_project_data(project_path):
        """Backup images and labels before autotrain"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(project_path, f'backup_{timestamp}')
        
        for split in ['train', 'val']:
            src_images = os.path.join(project_path, f'{split}/images')
            src_labels = os.path.join(project_path, f'{split}/labels')
            
            if os.path.exists(src_images):
                dst_images = os.path.join(backup_dir, f'{split}/images')
                shutil.copytree(src_images, dst_images)
            
            if os.path.exists(src_labels):
                dst_labels = os.path.join(backup_dir, f'{split}/labels')
                shutil.copytree(src_labels, dst_labels)
        
        print(f"Backup created at: {backup_dir}")


# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'bilt-service',
        'version': '1.0.0'
    })

@app.route('/models/available', methods=['POST'])
def get_available_models():
    """Get available BILT models"""
    data = request.json or {}
    task_type = data.get('task_type', 'detect')
    
    result = BILTModelManager.get_available_models(task_type)
    return jsonify(result)

@app.route('/train/start', methods=['POST'])
def start_training():
    """Start BILT training"""
    global training_active, training_thread
    
    if training_active:
        return jsonify({'error': 'Training already in progress'}), 400
    
    config = request.json
    
    if not config:
        return jsonify({'error': 'No configuration provided'}), 400
    
    # Get model and task type
    model_name = config.get("model")
    task_type = config.get("task_type", "detect")
    
    valid, message = BILTModelManager.validate_model(model_name, task_type)
    if not valid:
        return jsonify({'error': message}), 400
    # Start training in background thread
    training_thread = threading.Thread(
        target=BILTTrainer.train_model,
        args=(config,),
        daemon=True
    )
    training_thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Training started with {task_type} model',
        'saved_as': config.get("custom_model_name") if config.get("custom_model_name") else None
    })

@app.route('/train/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    if training_active:
        return jsonify({
            'status': 'running',
            'message': 'Training in progress...',
            'active': True
        })
    else:
        return jsonify({
            'status': 'idle',
            'message': 'No training running',
            'active': False
        })

@app.route('/autotrain/start', methods=['POST'])
def start_autotrain():
    """Start autotraining process"""
    global autotrain_active, autotrain_thread
    
    if autotrain_active:
        return jsonify({'error': 'Autotrain already running'}), 400
    
    config = request.json
    
    if not config:
        return jsonify({'error': 'No configuration provided'}), 400
    
    model_path = config.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Invalid model path'}), 400
    
    def autotrain_worker():
        global autotrain_active
        try:
            autotrain_active = True
            
            # Backup if enabled
            if config.get('backup_enabled', True):
                BILTAutoLabeler.backup_project_data(config.get('project_path'))
            
            # Auto-label images
            BILTAutoLabeler.auto_label_images(model_path, config)
            
            # Train model
            model = BILT(model_path)
            model.train(
                dataset=os.path.dirname(config.get('data_yaml')),
                epochs=int(config.get('epochs', 50)),
                batch_size=int(config.get('batch', 16)),
                img_size=int(config.get('imgsz', 640)),
                learning_rate=float(config.get('lr0', 0.01)),
                device=config.get('device', 'cpu'),
                save_dir=config.get('project_path'),
                name=f'autotrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            print("Autotrain completed successfully")
            
        except Exception as e:
            print(f"Autotrain error: {e}")
        finally:
            autotrain_active = False
    
    autotrain_thread = threading.Thread(target=autotrain_worker, daemon=True)
    autotrain_thread.start()
    
    return jsonify({'success': True, 'message': 'Autotrain started'})

@app.route('/autotrain/status', methods=['GET'])
def get_autotrain_status():
    """Get autotrain status"""
    return jsonify({'running': autotrain_active})

@app.route('/relabel/start', methods=['POST'])
def start_relabel():
    """Start relabeling process"""
    config = request.json
    
    if not config:
        return jsonify({'error': 'No configuration provided'}), 400
    
    model_path = config.get('model_path')
    target_split = config.get('target_split', 'train')
    conf_threshold = float(config.get('conf_threshold', 0.25))
    iou_threshold = float(config.get('iou_threshold', 0.45))
    backup_enabled = config.get('backup_enabled', True)
    mode = config.get('mode', 'all')
    project_path = config.get('project_path')
    task_type = config.get('task_type', 'detect')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Invalid model path'}), 400
    
    # BILT only supports detection
    if task_type != 'detect':
        return jsonify({'error': 'BILT only supports detection task'}), 400
    
    try:
        # Backup if enabled
        if backup_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(project_path, f'relabel_backup_{timestamp}')
            
            src_labels = os.path.join(project_path, f'{target_split}/labels')
            if os.path.exists(src_labels):
                dst_labels = os.path.join(backup_dir, f'{target_split}/labels')
                shutil.copytree(src_labels, dst_labels)
        
        model = BILT(model_path)
        
        images_dir = os.path.join(project_path, f'{target_split}/images')
        labels_dir = os.path.join(project_path, f'{target_split}/labels')
        
        if not os.path.exists(images_dir):
            return jsonify({'error': f'No images found in {target_split}'}), 400
        
        relabeled_count = 0
        
        from PIL import Image
        
        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            label_name = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            has_labels = os.path.exists(label_path)
            
            # Filter based on mode
            if mode == 'labeled' and not has_labels:
                continue
            if mode == 'unlabeled' and has_labels:
                continue
            
            img_path = os.path.join(images_dir, img_file)
            
            # Get predictions from BILT
            detections = model.predict(
                img_path,
                conf=conf_threshold,
                iou=iou_threshold
            )
            
            # Get image dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Write YOLO format labels
            with open(label_path, 'w') as f:
                for det in detections:
                    cls_id = det['class_id']
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Convert to YOLO format
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            
            relabeled_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Relabeled {relabeled_count} images',
            'count': relabeled_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    HOST = '127.0.0.1'
    PORT = 5001
    
    print("=" * 60)
    print("BILT Service - AGPL-3.0 Licensed")
    print("=" * 60)
    print(f"Starting BILT service on http://{HOST}:{PORT}")
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 60)
    
    app.run(host=HOST, port=PORT, debug=False, threaded=True)