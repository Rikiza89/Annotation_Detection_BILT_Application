"""
BILT Managers - Core detection, camera, and model management classes
Used by yolo_service.py (now using BILT instead of Ultralytics)
"""

import cv2
import numpy as np
import os
import json
import logging
import time
from datetime import datetime
import sys

# Import BILT - Add parent directory to path first
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bilt import BILT

logger = logging.getLogger(__name__)

class EnhancedCameraManager:
    def __init__(self):
        self.cap = None
        self.lock = __import__('threading').Lock()
        self.current_index = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.color_mode = 'bgr'
        
    def get_available_cameras(self):
        """Get all available camera indices with detailed info"""
        cameras = []
        camera_names = {}
        
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command', 
                 "Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status | ConvertTo-Json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                devices = json.loads(result.stdout)
                if not isinstance(devices, list):
                    devices = [devices]
                for idx, dev in enumerate(devices):
                    if dev.get('Status') == 'OK':
                        camera_names[idx] = dev.get('FriendlyName', f'Camera {idx}')
        except:
            pass
        
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                try:
                    cameras.append({
                        'index': i,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'name': camera_names.get(i, f'Camera {i}')
                    })
                except:
                    cameras.append({
                        'index': i, 'width': 640, 'height': 480,
                        'fps': 30.0, 'name': f'Camera {i}'
                    })
                finally:
                    cap.release()
        
        return cameras
    
    def _detect_color_issue(self, frame):
        if frame is None or len(frame.shape) != 3:
            return None
        
        b_avg, g_avg, r_avg = frame[:,:,0].mean(), frame[:,:,1].mean(), frame[:,:,2].mean()
        
        if b_avg > r_avg * 1.5 and b_avg > 100:
            return 'yuv2bgr'
        elif r_avg > b_avg * 1.5 and r_avg > 100:
            return 'rgb2bgr'
        return None
    
    def initialize_camera(self, index, config):
        """Initialize camera"""
        with self.lock:
            try:
                if self.cap:
                    self.cap.release()
                
                backends = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation"),
                    (cv2.CAP_ANY, "Default")
                ]
                
                self.cap = None
                for backend, name in backends:
                    try:
                        test_cap = cv2.VideoCapture(index, backend)
                        if test_cap.isOpened():
                            self.cap = test_cap
                            logger.info(f"Using {name} backend")
                            break
                        else:
                            test_cap.release()
                    except:
                        continue
                
                if not self.cap or not self.cap.isOpened():
                    return False
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['DEFAULT_CAMERA_WIDTH'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['DEFAULT_CAMERA_HEIGHT'])
                self.cap.set(cv2.CAP_PROP_FPS, config['DEFAULT_CAMERA_FPS'])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config['FRAME_BUFFER_SIZE'])
                
                self.current_index = index
                self.color_mode = 'bgr'
                return True
                
            except Exception as e:
                logger.error(f"Error initializing camera: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False
    
    def get_frame(self):
        """Get frame with FPS calculation"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None, 0
            
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                if self.color_mode == 'yuv2bgr':
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
                    except:
                        pass
                elif self.color_mode == 'rgb2bgr':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                self.frame_count += 1
                current_time = time.time()
                fps = 0
                if current_time - self.last_fps_time >= 1.0:
                    fps = round(self.frame_count / (current_time - self.last_fps_time), 1)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                return frame, fps
            return None, 0
    
    def get_camera_info(self):
        """Get current camera information"""
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return None
            
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            return {
                'index': self.current_index,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'backend': self.cap.getBackendName(),
                'fourcc': fourcc_str,
                'color_mode': self.color_mode
            }

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.current_index = None
                self.color_mode = 'bgr'


class ModelManager:
    @staticmethod
    def get_available_models(config):
        """Get all available BILT models (.pth files)"""
        models = []
        
        if os.path.exists(config['MODELS_DIR']):
            for file in os.listdir(config['MODELS_DIR']):
                if file.endswith('.pth'):
                    file_path = os.path.join(config['MODELS_DIR'], file)
                    models.append({
                        'name': file,
                        'path': file_path,
                        'size': os.path.getsize(file_path),
                        'type': 'local'
                    })
        
        return models
    
    @staticmethod
    def load_model(model_name, config):
        """Load BILT model"""
        try:
            model_path = os.path.join(config['MODELS_DIR'], model_name)
            if not os.path.exists(model_path):
                return False, f"Model not found: {model_name}"
            
            logger.info(f"Loading BILT model: {model_path}")
            model = BILT(model_path)
            
            return True, {
                'model': model,
                'classes': model.names,
                'class_count': model.num_classes
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False, str(e)


class ChainDetectionManager:
    @staticmethod
    def initialize_chain(chain_state):
        """Initialize chain detection state"""
        chain_state.update({
            'active': True,
            'current_step': 0,
            'step_start_time': time.time(),
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
        })
    
    @staticmethod
    def process_chain_detection(detections, detection_settings, chain_state, latest_frame, frame_lock):
        """Process detections in chain mode"""
        if not detection_settings['chain_mode'] or not chain_state['active']:
            return detections, None
        
        if not detection_settings['chain_steps']:
            return detections, "No chain steps configured"
        
        current_time = time.time()
        step_detections = {}
        for d in detections:
            class_name = d['class_name']
            step_detections[class_name] = step_detections.get(class_name, 0) + 1
        
        # Check if waiting for acknowledgment
        if chain_state.get('waiting_for_ack', False):
            return detections, {
                'step': chain_state['error_step'],
                'step_name': chain_state.get('error_step_name', 'Error'),
                'detected': step_detections,
                'error': True,
                'error_message': chain_state.get('error_message', 'Wrong object detected'),
                'waiting_for_ack': True,
                'timestamp': current_time
            }
        
        # Check cycle pause
        if chain_state.get('cycle_pause', False):
            pause_elapsed = current_time - chain_state['cycle_pause_start']
            pause_remaining = detection_settings['chain_pause_time'] - pause_elapsed
            
            if pause_remaining <= 0:
                chain_state['cycle_pause'] = False
                chain_state['cycle_pause_start'] = None
                chain_state['current_step'] = 0
                chain_state['step_start_time'] = current_time
            else:
                return detections, {
                    'step': -1,
                    'step_name': 'Cycle Pause',
                    'detected': step_detections,
                    'remaining_pause': pause_remaining,
                    'timestamp': current_time
                }
        
        # Check cycle completion
        if chain_state['current_step'] >= len(detection_settings['chain_steps']):
            chain_state['current_step'] = 0
            chain_state['completed_cycles'] += 1
            chain_state['cycle_pause'] = True
            chain_state['cycle_pause_start'] = current_time
            return detections, {
                'step': -1,
                'step_name': 'Cycle Completed',
                'detected': step_detections,
                'remaining_pause': detection_settings['chain_pause_time'],
                'timestamp': current_time
            }
        
        current_step_config = detection_settings['chain_steps'][chain_state['current_step']]
        required_classes = set(current_step_config.get('classes', {}).keys())
        
        # Check for wrong objects (objects not in current step)
        for detected_class in step_detections.keys():
            if detected_class not in required_classes:
                # Wrong object detected - raise error
                chain_state['waiting_for_ack'] = True
                chain_state['error_step'] = chain_state['current_step']
                chain_state['error_step_name'] = current_step_config['name']
                chain_state['error_message'] = f"Wrong object '{detected_class}' detected in step '{current_step_config['name']}'"
                chain_state['wrong_object'] = detected_class
                chain_state['failed_steps'] += 1
                
                return detections, {
                    'step': chain_state['current_step'],
                    'step_name': current_step_config['name'],
                    'detected': step_detections,
                    'error': True,
                    'error_message': chain_state['error_message'],
                    'wrong_object': detected_class,
                    'waiting_for_ack': True,
                    'timestamp': current_time
                }
        
        # Check if correct objects detected with required counts
        step_completed = all(
            step_detections.get(cls, 0) >= cnt
            for cls, cnt in current_step_config.get('classes', {}).items()
        )
        
        if step_completed:
            # Correct detection - advance to next step
            chain_state['current_step'] += 1
            chain_state['step_start_time'] = current_time
            chain_state['last_step_result'] = 'success'
        
        return detections, {
            'step': chain_state['current_step'],
            'step_name': current_step_config['name'],
            'detected': step_detections,
            'required': current_step_config.get('classes', {}),
            'completed': step_completed,
            'timestamp': current_time
        }
    
    @staticmethod
    def check_for_skip(step_detections, current_step_index, detection_settings):
        """Check if any objects from future steps are detected"""
        total_steps = len(detection_settings['chain_steps'])
        if total_steps == 0:
            return False, None
        
        current_step_config = detection_settings['chain_steps'][current_step_index]
        current_classes = set(current_step_config.get('classes', {}).keys())
        
        # Check future steps
        for future_idx in range(current_step_index + 1, total_steps):
            future_config = detection_settings['chain_steps'][future_idx]
            for class_name, required_count in future_config.get('classes', {}).items():
                if class_name in current_classes:
                    continue
                if step_detections.get(class_name, 0) >= required_count:
                    return True, future_idx
        
        return False, None
    
    @staticmethod
    def reset_chain(chain_state):
        """Reset chain detection state"""
        chain_state.update({
            'current_step': 0,
            'step_start_time': time.time(),
            'step_history': [],
            'current_detections': {},
            'last_step_result': None
        })
    
    @staticmethod
    def get_chain_status(detection_settings, chain_state):
        """Get current chain status"""
        if not detection_settings['chain_mode']:
            return {'active': False}
        
        current_step_config = None
        if (detection_settings['chain_steps'] and 
            chain_state['current_step'] < len(detection_settings['chain_steps'])):
            current_step_config = detection_settings['chain_steps'][chain_state['current_step']]
        
        remaining_time = max(0, detection_settings['chain_timeout'] - 
                        (time.time() - chain_state['step_start_time']))
        
        pause_remaining = 0
        if chain_state['cycle_pause'] and chain_state['cycle_pause_start']:
            pause_remaining = max(0, detection_settings['chain_pause_time'] - 
                                (time.time() - chain_state['cycle_pause_start']))
        
        return {
            'active': chain_state['active'],
            'current_step': chain_state['current_step'],
            'total_steps': len(detection_settings['chain_steps']),
            'current_step_config': current_step_config,
            'completed_cycles': chain_state['completed_cycles'],
            'failed_steps': chain_state['failed_steps'],
            'current_detections': chain_state['current_detections'],
            'remaining_time': remaining_time,
            'last_result': chain_state['last_step_result'],
            'step_history': chain_state['step_history'][-10:],
            'cycle_pause': chain_state['cycle_pause'],
            'pause_remaining': pause_remaining,
            'waiting_for_ack': chain_state.get('waiting_for_ack', False),
            'error': chain_state.get('waiting_for_ack', False),
            'error_message': chain_state.get('error_message'),
            'wrong_object': chain_state.get('wrong_object'),
            'error_step': chain_state.get('error_step')
        }

    @staticmethod
    def acknowledge_error(chain_state):
        """Acknowledge error and reset to error step"""
        if chain_state.get('waiting_for_ack', False):
            # Clear all error flags
            chain_state['waiting_for_ack'] = False
            chain_state['error_message'] = None
            chain_state['wrong_object'] = None
            chain_state['last_step_result'] = None  # Important: clear the error result
            chain_state['step_start_time'] = time.time()
            # Stay at the same step (error_step)
            chain_state['current_step'] = chain_state.get('error_step', chain_state['current_step'])
            chain_state['error_step'] = None  # Clear error step
            return True
        return False

class DetectionProcessor:
    @staticmethod
    def process_detections(bilt_detections, frame, detection_settings, object_counters, 
                          detection_stats, chain_state, counter_triggered, latest_frame, frame_lock):
        """Process BILT detection results"""
        detections = []
        
        if not bilt_detections:
            return detections, frame
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        # Convert BILT detections to internal format
        for det in bilt_detections:
            if detection_settings['classes'] and det['class_id'] not in detection_settings['classes']:
                continue
            
            detections.append({
                'bbox': det['bbox'],
                'confidence': det['score'],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'type': 'detect'
            })
        
        # Count detections
        frame_detections = {}
        for det in detections:
            class_name = det['class_name']
            frame_detections[class_name] = frame_detections.get(class_name, 0) + 1
        
        # Process chain mode
        chain_result = None
        if detection_settings['chain_mode']:
            detections, chain_result = ChainDetectionManager.process_chain_detection(
                detections, detection_settings, chain_state, latest_frame, frame_lock
            )
        
        # Counter mode
        if detection_settings['counter_mode'] and not detection_settings['chain_mode']:
            for class_name, count in frame_detections.items():
                if class_name not in counter_triggered:
                    object_counters[class_name] = object_counters.get(class_name, 0) + count
                    counter_triggered[class_name] = True
        
        detection_stats['total_detections'] += len(detections)
        detection_stats['last_detection_time'] = datetime.now()
        
        annotated_frame = DetectionProcessor.draw_detections(
            frame.copy(), detections, chain_result, detection_settings, chain_state, detection_stats
        )
        
        return detections, annotated_frame
    
    @staticmethod
    def draw_detections(frame, detections, chain_result, detection_settings, chain_state, detection_stats):
        """Draw detections on frame"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            color = colors[det['class_id'] % len(colors)]
            thickness = 2
            
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw chain overlay
        if detection_settings['chain_mode']:
            DetectionProcessor.draw_chain_overlay(frame, chain_state, detection_settings)
        else:
            cv2.putText(frame, f"FPS: {detection_stats['fps']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    @staticmethod
    def draw_chain_overlay(frame, chain_state, detection_settings):
        """Draw chain mode overlay"""
        if not detection_settings['chain_steps']:
            return
        
        current_step = chain_state['current_step']
        total_steps = len(detection_settings['chain_steps'])
        
        # Draw progress bar
        bar_width, bar_height = 400, 30
        bar_x, bar_y = (frame.shape[1] - bar_width) // 2, 20
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        if chain_state.get('cycle_pause', False):
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 165, 255), -1)
            cv2.putText(frame, "CYCLE PAUSE", (bar_x + 120, bar_y + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            progress = current_step / total_steps if total_steps > 0 else 0
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, f"Step {current_step + 1}/{total_steps}", (bar_x + 10, bar_y + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class ImageManager:
    @staticmethod
    def save_detection_image(frame, detections, config, prefix="detection"):
        """Save detection image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(config['SAVED_IMAGES_DIR'], filename)
            
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, config['SAVE_IMAGE_QUALITY']])
            logger.info(f"Saved detection image: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return None
    
    @staticmethod
    def save_dataset_image(frame, detections, project_folder, config):
        """Save image for dataset"""
        if not project_folder:
            return None
        
        try:
            from werkzeug.utils import secure_filename
            project_path = os.path.join(config['PROJECTS_DIR'], secure_filename(project_folder))
            images_path = os.path.join(project_path, 'images')
            labels_path = os.path.join(project_path, 'labels')
            
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_filename = f"img_{timestamp}.jpg"
            label_filename = f"img_{timestamp}.txt"
            
            image_path = os.path.join(images_path, image_filename)
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config['SAVE_IMAGE_QUALITY']])
            
            label_path = os.path.join(labels_path, label_filename)
            h, w = frame.shape[:2]
            
            with open(label_path, 'w') as f:
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    f.write(f"{detection['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            logger.info(f"Saved dataset image: {image_filename} to project {project_folder}")
            return image_filename
            
        except Exception as e:
            logger.error(f"Error saving dataset image: {str(e)}")
            return None