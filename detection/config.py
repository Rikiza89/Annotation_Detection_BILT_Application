"""
Configuration file for BILT Detection Web Application
"""

import os
import sys

# Determine base directory for PyInstaller compatibility
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    BASE_DIR = current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    
    # Application Settings
    HOST = '127.0.0.1'
    PORT = 5003
    DEBUG = True
    
    # Directories - Use BASE_DIR for PyInstaller compatibility
    BASE_DIR = BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    SAVED_IMAGES_DIR = os.path.join(BASE_DIR, 'saved_images')
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    PROJECTS_DIR = os.path.join(BASE_DIR, 'projects')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    
    # Camera Settings
    DEFAULT_CAMERA_WIDTH = 1280
    DEFAULT_CAMERA_HEIGHT = 960
    DEFAULT_CAMERA_FPS = 30
    MAX_CAMERA_INDEX = 3
    
    # Detection Settings
    DEFAULT_CONF_THRESHOLD = 0.60
    DEFAULT_IOU_THRESHOLD = 0.10
    DEFAULT_MAX_DETECTIONS = 10
    
    # Performance Settings
    FRAME_RATE_LIMIT = 30
    FRAME_BUFFER_SIZE = 1
    
    # Image Settings
    SAVE_IMAGE_QUALITY = 99
    MAX_IMAGE_SIZE = (3840, 2160)
    
    # UI Settings
    WIDGET_UPDATE_INTERVAL = 2000
    
    # Model Settings
    SUPPORTED_MODEL_FORMATS = ['.pth', '.onnx', '.engine', '.torchscript']
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(BASE_DIR, 'bilt_detection.log')
    
    # Security Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        directories = [
            Config.MODELS_DIR,
            Config.SAVED_IMAGES_DIR,
            Config.DATASETS_DIR,
            Config.PROJECTS_DIR,
            Config.TEMPLATES_DIR,
            os.path.join(Config.BASE_DIR, 'chains')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    HOST = '127.0.0.1'
    
class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}