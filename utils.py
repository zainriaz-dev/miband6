#!/usr/bin/env python3
"""
Utility Functions Module
Provides file management, error handling, and helper functions
for the Zepp Life health metrics extractor.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import json
import csv
import tempfile
import shutil

# Configure logging
logger = logging.getLogger(__name__)

class FileManager:
    """File management utilities"""
    
    @staticmethod
    def ensure_directory(path: Path) -> bool:
        """Ensure directory exists, create if necessary"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Could not create directory {path}: {e}")
            return False
            
    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Optional[Path]:
        """Create backup of existing file"""
        try:
            if file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backup created: {backup_path}")
                return backup_path
        except Exception as e:
            logger.error(f"Could not create backup of {file_path}: {e}")
        return None
        
    @staticmethod
    def safe_write_csv(data: List[List[Any]], output_path: Path, headers: Optional[List[str]] = None) -> bool:
        """Safely write CSV data with backup and error handling"""
        try:
            # Create backup if file exists
            if output_path.exists():
                FileManager.backup_file(output_path)
                
            # Ensure parent directory exists
            FileManager.ensure_directory(output_path.parent)
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, 
                                           suffix='.csv', encoding='utf-8') as temp_file:
                writer = csv.writer(temp_file)
                
                if headers:
                    writer.writerow(headers)
                    
                writer.writerows(data)
                temp_path = Path(temp_file.name)
                
            # Move temporary file to final location
            shutil.move(temp_path, output_path)
            logger.info(f"CSV file written successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing CSV file {output_path}: {e}")
            # Clean up temporary file if it exists
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            return False
            
    @staticmethod
    def safe_write_json(data: Dict[str, Any], output_path: Path, indent: int = 2) -> bool:
        """Safely write JSON data with backup and error handling"""
        try:
            # Create backup if file exists
            if output_path.exists():
                FileManager.backup_file(output_path)
                
            # Ensure parent directory exists
            FileManager.ensure_directory(output_path.parent)
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           suffix='.json', encoding='utf-8') as temp_file:
                json.dump(data, temp_file, indent=indent, ensure_ascii=False)
                temp_path = Path(temp_file.name)
                
            # Move temporary file to final location
            shutil.move(temp_path, output_path)
            logger.info(f"JSON file written successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing JSON file {output_path}: {e}")
            # Clean up temporary file if it exists
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            return False
            
    @staticmethod
    def find_files_by_pattern(directory: Path, patterns: List[str], 
                            recursive: bool = True, max_files: int = 1000) -> List[Path]:
        """Find files matching patterns in directory"""
        found_files = []
        
        try:
            for pattern in patterns:
                if recursive:
                    matches = list(directory.rglob(pattern))
                else:
                    matches = list(directory.glob(pattern))
                    
                found_files.extend(matches)
                
                # Limit number of files to prevent memory issues
                if len(found_files) >= max_files:
                    logger.warning(f"File search limited to {max_files} files")
                    break
                    
        except Exception as e:
            logger.error(f"Error searching for files in {directory}: {e}")
            
        return sorted(list(set(found_files)))  # Remove duplicates and sort
        
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'extension': file_path.suffix.lower(),
                'is_image': file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'],
                'path': str(file_path.absolute())
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}

class ErrorHandler:
    """Comprehensive error handling utilities"""
    
    @staticmethod
    def handle_exception(func):
        """Decorator for handling exceptions with logging"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return None
        return wrapper
        
    @staticmethod
    def log_system_info():
        """Log system information for debugging"""
        try:
            import platform
            import psutil
            
            logger.info("=== SYSTEM INFORMATION ===")
            logger.info(f"Platform: {platform.platform()}")
            logger.info(f"Python: {sys.version}")
            logger.info(f"CPU: {platform.processor()}")
            logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            logger.info("=" * 28)
            
        except ImportError:
            logger.info("psutil not available, skipping detailed system info")
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            
    @staticmethod
    def validate_dependencies() -> Tuple[bool, List[str]]:
        """Validate that all required dependencies are available"""
        required_modules = [
            'cv2',
            'numpy',
            'PIL',
            'pytesseract'
        ]
        
        optional_modules = [
            'easyocr',
            'paddleocr',
            'torch'
        ]
        
        missing_required = []
        missing_optional = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_required.append(module)
                
        for module in optional_modules:
            try:
                __import__(module)
            except ImportError:
                missing_optional.append(module)
                
        if missing_required:
            logger.error(f"Missing required dependencies: {missing_required}")
            
        if missing_optional:
            logger.warning(f"Missing optional dependencies (reduced functionality): {missing_optional}")
            
        return len(missing_required) == 0, missing_required + missing_optional

class ConfigManager:
    """Configuration management utilities"""
    
    DEFAULT_CONFIG = {
        'ocr': {
            'engines': ['tesseract', 'easyocr'],
            'tesseract_config': '--oem 3 --psm 6',
            'preprocessing': {
                'scale_factor': 3,
                'enhance_contrast': True,
                'denoise': True
            }
        },
        'regions': {
            'auto_scaling': True,
            'validation': True
        },
        'output': {
            'format': 'csv',
            'append_mode': True,
            'backup_existing': True
        },
        'logging': {
            'level': 'INFO',
            'file_logging': False,
            'console_logging': True
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / 'config.json'
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
        
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    
                # Merge with defaults
                self._merge_config(self.config, user_config)
                logger.info(f"Configuration loaded from: {self.config_path}")
                return True
            else:
                logger.info("No configuration file found, using defaults")
                self.save_config()  # Create default config file
                return False
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
            
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            return FileManager.safe_write_json(self.config, self.config_path)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def _merge_config(self, default: Dict, user: Dict):
        """Recursively merge user config with defaults"""
        for key, value in user.items():
            if key in default:
                if isinstance(default[key], dict) and isinstance(value, dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
                
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'ocr.engines')"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
                
            return value
            
        except (KeyError, TypeError):
            return default
            
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config = self.config
            
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
                
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {e}")
            return False

class ProgressTracker:
    """Progress tracking for batch operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        self._display_progress()
        
    def _display_progress(self) -> None:
        """Display progress bar in console"""
        if self.total == 0:
            return
            
        percentage = (self.current / self.total) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "Unknown"
            
        print(f'\r{self.description}: |{bar}| {self.current}/{self.total} ({percentage:.1f}%) ETA: {eta_str}', 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup comprehensive logging configuration"""
    
    # Map string levels to logging constants
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            FileManager.ensure_directory(log_file.parent)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            logger.error(f"Could not setup file logging: {e}")
    
    root_logger.setLevel(log_level)
    
def get_temp_dir() -> Path:
    """Get temporary directory for processing"""
    temp_dir = Path(tempfile.gettempdir()) / "zepp_life_extractor"
    FileManager.ensure_directory(temp_dir)
    return temp_dir

def cleanup_temp_files(temp_dir: Optional[Path] = None) -> None:
    """Clean up temporary files"""
    if temp_dir is None:
        temp_dir = get_temp_dir()
        
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary files: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary files: {e}")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
        
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def validate_image_file(file_path: Path) -> Tuple[bool, str]:
    """Validate if file is a valid image with detailed error message"""
    try:
        if not file_path.exists():
            return False, "File does not exist"
            
        if not file_path.is_file():
            return False, "Path is not a file"
            
        # Check file extension
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        if file_path.suffix.lower() not in valid_extensions:
            return False, f"Invalid file extension. Supported: {valid_extensions}"
            
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"
            
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return False, f"File too large: {format_file_size(file_size)}"
            
        # Try to load with OpenCV
        import cv2
        image = cv2.imread(str(file_path))
        if image is None:
            return False, "Could not load image (corrupted or unsupported format)"
            
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            return False, f"Image too small: {width}x{height} (minimum 100x100)"
            
        return True, "Valid image file"
        
    except Exception as e:
        return False, f"Error validating image: {e}"
