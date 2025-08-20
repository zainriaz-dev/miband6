"""
Computer Vision Utilities Module
Provides image preprocessing, enhancement, and region detection functions
for Zepp Life health metric screenshots.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Region:
    """Represents a region of interest in an image"""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    metric_type: str
    expected_format: str  # e.g., "number", "number_with_unit", "percentage"

class ImagePreprocessor:
    """Image preprocessing utilities for OCR optimization"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        if use_gpu and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            logger.info("OpenCL acceleration enabled")
        else:
            cv2.ocl.setUseOpenCL(False)
            
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], 
                              [-1, 9,-1], 
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return image
            
    def preprocess_for_ocr(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Preprocess image region for optimal OCR"""
        try:
            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y:y+h, x:x+w].copy()
            else:
                roi = image.copy()
                
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
                
            # Enhance contrast
            enhanced = self.enhance_image(gray)
            
            # Scale up for better OCR (OCR works better on larger text)
            scale_factor = 3
            height, width = enhanced.shape
            scaled = cv2.resize(enhanced, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            
            # Apply binary threshold
            _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Optional: Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"OCR preprocessing error: {e}")
            return image
            
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions using MSER or contour detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter regions by size (likely to contain text)
                if 10 <= w <= 200 and 10 <= h <= 100:
                    aspect_ratio = w / h
                    if 0.1 <= aspect_ratio <= 10:  # Reasonable aspect ratios for text
                        regions.append((x, y, w, h))
                        
            return regions
            
        except Exception as e:
            logger.error(f"Text region detection error: {e}")
            return []

class ZeppLifeRegionDetector:
    """Specific region detector for Zepp Life app screenshots"""
    
    def __init__(self):
        # Define enhanced regions for Zepp Life interface based on user's annotated image
        # Improved coordinates with better spacing and positioning
        self.standard_regions = {
            'heart_rate': Region(
                name='heart_rate',
                bbox=(60, 190, 150, 80),  # Green box - Heart Rate: 86 BPM
                metric_type='heart_rate',
                expected_format='number_with_unit'
            ),
            'stress': Region(
                name='stress', 
                bbox=(90, 370, 120, 80),  # Blue box - Stress: 40
                metric_type='stress',
                expected_format='number'
            ),
            'steps': Region(
                name='steps',
                bbox=(45, 680, 175, 90),  # Red box - Steps: 0
                metric_type='steps',
                expected_format='number'
            ),
            'distance': Region(
                name='distance',
                bbox=(335, 585, 135, 80),  # Cyan box - Distance: 0.0 km
                metric_type='distance',
                expected_format='number_with_unit'
            ),
            'calories': Region(
                name='calories',
                bbox=(270, 680, 175, 60),  # Magenta box - Calories: 0 kcal
                metric_type='calories',
                expected_format='number_with_unit'
            ),
            'blood_oxygen': Region(
                name='blood_oxygen',
                bbox=(245, 895, 125, 80),  # Yellow box - Blood Oxygen: 97%
                metric_type='blood_oxygen',
                expected_format='percentage'
            )
        }
        
    def detect_regions(self, image: np.ndarray) -> Dict[str, Region]:
        """Detect regions in Zepp Life screenshot"""
        height, width = image.shape[:2]
        
        # For images close to our standard coordinates (622x1280), use direct coordinates
        # These coordinates are specifically for 622x1280 images based on user's annotations
        scaled_regions = {}
        
        for name, region in self.standard_regions.items():
            # Use the coordinates directly for now since they're based on 622x1280 images
            x, y, w, h = region.bbox
            
            # Only apply minimal scaling if image dimensions are very different
            if abs(width - 622) > 100 or abs(height - 1280) > 200:
                # Apply proportional scaling only if significantly different
                scale_x = width / 622
                scale_y = height / 1280
                
                scaled_bbox = (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                )
            else:
                # Use coordinates as-is for similar sized images
                scaled_bbox = (x, y, w, h)
            
            # Ensure bbox is within image bounds
            scaled_bbox = self._clip_bbox(scaled_bbox, width, height)
            
            scaled_regions[name] = Region(
                name=region.name,
                bbox=scaled_bbox,
                metric_type=region.metric_type,
                expected_format=region.expected_format
            )
            
        return scaled_regions
        
    def _clip_bbox(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Clip bounding box to image boundaries"""
        x, y, w, h = bbox
        
        # Clip to image boundaries
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        return (x, y, w, h)
        
    def adaptive_region_detection(self, image: np.ndarray) -> Dict[str, Region]:
        """Adaptively detect regions using computer vision"""
        try:
            # Start with standard regions
            regions = self.detect_regions(image)
            
            # Use template matching or feature detection to refine regions
            # This is a simplified implementation
            refined_regions = {}
            
            for name, region in regions.items():
                # Always include all regions for now to ensure proper extraction
                refined_regions[name] = region
                        
            return refined_regions
            
        except Exception as e:
            logger.error(f"Adaptive region detection error: {e}")
            return self.detect_regions(image)
            
    def _is_valid_metric_region(self, roi: np.ndarray, expected_format: str) -> bool:
        """Check if ROI likely contains valid metric data"""
        try:
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
                
            # Check for text-like features
            # Calculate edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Check variance (text regions usually have higher variance)
            variance = np.var(gray)
            
            # Simple heuristics
            return edge_density > 0.01 and variance > 100
            
        except Exception as e:
            logger.warning(f"Region validation error: {e}")
            return True
            
    def _search_nearby_region(self, image: np.ndarray, original_region: Region) -> Optional[Region]:
        """Search for better region near the original"""
        try:
            x, y, w, h = original_region.bbox
            
            # Search in expanded area
            search_x = max(0, x - w//2)
            search_y = max(0, y - h//2)
            search_w = min(image.shape[1] - search_x, w * 2)
            search_h = min(image.shape[0] - search_y, h * 2)
            
            search_roi = image[search_y:search_y+search_h, search_x:search_x+search_w]
            
            # Use template matching or other CV techniques here
            # For now, return None (use original region)
            return None
            
        except Exception as e:
            logger.warning(f"Nearby region search error: {e}")
            return None

def draw_debug_regions(image: np.ndarray, regions: Dict[str, Region], output_path: str = None) -> np.ndarray:
    """Draw enhanced region visualization with proper colors and spacing"""
    debug_image = image.copy()
    
    # Create a semi-transparent overlay for better visualization
    overlay = image.copy()
    
    # Enhanced colors matching your image requirements
    colors = {
        'heart_rate': (0, 255, 0),      # Bright Green 
        'stress': (255, 100, 0),        # Blue
        'steps': (0, 50, 255),          # Red
        'distance': (255, 255, 0),      # Cyan
        'calories': (255, 0, 255),      # Magenta  
        'blood_oxygen': (0, 255, 255)   # Yellow
    }
    
    # Labels for display
    labels = {
        'heart_rate': 'heart_rate',
        'stress': 'stress', 
        'steps': 'steps',
        'distance': 'distance',
        'calories': 'calories',
        'blood_oxygen': 'blood_oxygen'
    }
    
    for name, region in regions.items():
        x, y, w, h = region.bbox
        color = colors.get(name, (128, 128, 128))
        label = labels.get(name, name)
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
        # Draw border rectangle
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 3)
        
        # Add label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        label_x = x
        label_y = y - 15
        
        # Ensure label is within image bounds
        if label_y < 20:
            label_y = y + h + 25
            
        # Draw label background
        cv2.rectangle(debug_image, 
                     (label_x - 2, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 2, label_y + 5), 
                     color, -1)
        
        # Draw label text
        cv2.putText(debug_image, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend the overlay with the original image for semi-transparency
    alpha = 0.2  # Transparency factor
    debug_image = cv2.addWeighted(overlay, alpha, debug_image, 1 - alpha, 0)
    
    # Add final border rectangles for clarity
    for name, region in regions.items():
        x, y, w, h = region.bbox
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
    
    if output_path:
        cv2.imwrite(output_path, debug_image)
        
    return debug_image

def validate_image(image_path: str) -> bool:
    """Validate if image is suitable for processing"""
    try:
        # Check if file exists and is readable
        image = cv2.imread(image_path)
        if image is None:
            return False
            
        # Check image dimensions
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            return False
            
        # Check if image is not completely black or white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val < 10 or mean_val > 245:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False
