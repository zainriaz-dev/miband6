"""
OCR Engines Module
Provides multiple OCR engine implementations with hardware optimization
for extracting text from Zepp Life health metric screenshots.
"""

import cv2
import numpy as np
import platform
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Container for OCR extraction results"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    
@dataclass
class HardwareConfig:
    """Hardware configuration for optimization"""
    has_cuda: bool = False
    has_opencl: bool = False
    cpu_cores: int = 1
    ram_gb: float = 0.0
    platform: str = "unknown"
    
class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware_config = hardware_config
        self.engine_name = "base"
        
    @abstractmethod
    def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text from image or image region"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if OCR engine is available"""
        pass

class TesseractOCR(OCREngine):
    """Tesseract OCR engine implementation"""
    
    def __init__(self, hardware_config: HardwareConfig):
        super().__init__(hardware_config)
        self.engine_name = "tesseract"
        self._configure_tesseract()
        
    def _configure_tesseract(self):
        """Configure Tesseract executable path"""
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            if self.hardware_config.platform == "Windows":
                tesseract_cmd = os.environ.get("TESSERACT_CMD")
                if tesseract_cmd and os.path.isfile(tesseract_cmd):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                else:
                    possible_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                    ]
                    for path in possible_paths:
                        if os.path.isfile(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            break
            else:
                pytesseract.pytesseract.tesseract_cmd = "tesseract"
                
        except ImportError:
            self.pytesseract = None
            logger.warning("Tesseract not available")
            
    def is_available(self) -> bool:
        """Check if Tesseract is available"""
        return self.pytesseract is not None
        
    def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using Tesseract"""
        if not self.is_available():
            return []
            
        try:
            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y:y+h, x:x+w]
            else:
                roi = image
                x, y = 0, 0
                
            # Configure Tesseract for numeric data
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.%kmKM'
            
            # Get detailed OCR data
            data = self.pytesseract.image_to_data(roi, config=custom_config, output_type=self.pytesseract.Output.DICT)
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Confidence threshold
                    bbox = (
                        data['left'][i] + x,
                        data['top'][i] + y,
                        data['width'][i],
                        data['height'][i]
                    )
                    results.append(OCRResult(text, data['conf'][i] / 100.0, bbox))
                    
            return results
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return []

class EasyOCR(OCREngine):
    """EasyOCR engine implementation"""
    
    def __init__(self, hardware_config: HardwareConfig):
        super().__init__(hardware_config)
        self.engine_name = "easyocr"
        self.reader = None
        self._initialize_reader()
        
    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        try:
            import easyocr
            gpu = self.hardware_config.has_cuda
            self.reader = easyocr.Reader(['en'], gpu=gpu)
            logger.info(f"EasyOCR initialized with GPU: {gpu}")
        except ImportError:
            logger.warning("EasyOCR not available")
        except Exception as e:
            logger.error(f"EasyOCR initialization error: {e}")
            
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        return self.reader is not None
        
    def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        if not self.is_available():
            return []
            
        try:
            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y:y+h, x:x+w]
            else:
                roi = image
                x, y = 0, 0
                
            # Perform OCR
            results_raw = self.reader.readtext(roi, allowlist='0123456789.%kmKM')
            
            results = []
            for detection in results_raw:
                bbox_coords, text, confidence = detection
                if confidence > 0.3:  # Confidence threshold
                    # Convert bbox to x, y, w, h format
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    bbox = (
                        int(min(x_coords)) + x,
                        int(min(y_coords)) + y,
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    )
                    results.append(OCRResult(text.strip(), confidence, bbox))
                    
            return results
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []

class PaddleOCR(OCREngine):
    """PaddleOCR engine implementation"""
    
    def __init__(self, hardware_config: HardwareConfig):
        super().__init__(hardware_config)
        self.engine_name = "paddleocr"
        self.ocr = None
        self._initialize_paddle()
        
    def _initialize_paddle(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR as PaddleOCRLib
            use_gpu = self.hardware_config.has_cuda
            self.ocr = PaddleOCRLib(use_angle_cls=True, lang='en', use_gpu=use_gpu)
            logger.info(f"PaddleOCR initialized with GPU: {use_gpu}")
        except ImportError:
            logger.warning("PaddleOCR not available")
        except Exception as e:
            logger.error(f"PaddleOCR initialization error: {e}")
            
    def is_available(self) -> bool:
        """Check if PaddleOCR is available"""
        return self.ocr is not None
        
    def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using PaddleOCR"""
        if not self.is_available():
            return []
            
        try:
            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y:y+h, x:x+w]
            else:
                roi = image
                x, y = 0, 0
                
            # Perform OCR
            results_raw = self.ocr.ocr(roi, cls=True)
            
            results = []
            if results_raw and results_raw[0]:
                for detection in results_raw[0]:
                    bbox_coords, (text, confidence) = detection
                    if confidence > 0.3:  # Confidence threshold
                        # Convert bbox to x, y, w, h format
                        x_coords = [point[0] for point in bbox_coords]
                        y_coords = [point[1] for point in bbox_coords]
                        bbox = (
                            int(min(x_coords)) + x,
                            int(min(y_coords)) + y,
                            int(max(x_coords) - min(x_coords)),
                            int(max(y_coords) - min(y_coords))
                        )
                        results.append(OCRResult(text.strip(), confidence, bbox))
                        
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return []

class MultiEngineOCR:
    """Multi-engine OCR with automatic fallback and hardware optimization"""
    
    def __init__(self):
        self.hardware_config = self._detect_hardware()
        self.engines = self._initialize_engines()
        logger.info(f"Initialized {len(self.engines)} OCR engines")
        
    def _detect_hardware(self) -> HardwareConfig:
        """Detect hardware capabilities"""
        config = HardwareConfig()
        config.platform = platform.system()
        config.cpu_cores = os.cpu_count() or 1
        
        try:
            import psutil
            config.ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            config.ram_gb = 4.0  # Default estimate
            
        # Check CUDA availability
        try:
            import torch
            config.has_cuda = torch.cuda.is_available()
        except ImportError:
            config.has_cuda = False
            
        # Check OpenCL
        try:
            config.has_opencl = cv2.ocl.haveOpenCL()
        except:
            config.has_opencl = False
            
        logger.info(f"Hardware: {config.platform}, {config.cpu_cores} cores, "
                   f"{config.ram_gb:.1f}GB RAM, CUDA: {config.has_cuda}, OpenCL: {config.has_opencl}")
        
        return config
        
    def _initialize_engines(self) -> List[OCREngine]:
        """Initialize available OCR engines"""
        engines = []
        
        # Try EasyOCR first (usually most reliable)
        easy_ocr = EasyOCR(self.hardware_config)
        if easy_ocr.is_available():
            engines.append(easy_ocr)
            
        # Try Tesseract
        tesseract = TesseractOCR(self.hardware_config)
        if tesseract.is_available():
            engines.append(tesseract)
            
        # Try PaddleOCR
        paddle_ocr = PaddleOCR(self.hardware_config)
        if paddle_ocr.is_available():
            engines.append(paddle_ocr)
            
        return engines
        
    def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using best available engine"""
        for engine in self.engines:
            try:
                results = engine.extract_text(image, region)
                if results:
                    logger.debug(f"Successfully extracted text using {engine.engine_name}")
                    return results
            except Exception as e:
                logger.warning(f"Engine {engine.engine_name} failed: {e}")
                continue
                
        logger.warning("All OCR engines failed")
        return []
        
    def extract_text_consensus(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using multiple engines and find consensus"""
        all_results = {}
        
        for engine in self.engines:
            try:
                results = engine.extract_text(image, region)
                all_results[engine.engine_name] = results
            except Exception as e:
                logger.warning(f"Engine {engine.engine_name} failed: {e}")
                
        # Simple consensus: use results from the engine with highest average confidence
        best_engine = None
        best_confidence = 0
        
        for engine_name, results in all_results.items():
            if results:
                avg_confidence = sum(r.confidence for r in results) / len(results)
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_engine = engine_name
                    
        return all_results.get(best_engine, [])
