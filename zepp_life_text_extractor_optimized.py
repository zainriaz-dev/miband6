#!/usr/bin/env python3
"""
Zepp Life Text Extractor - Optimized Edition
Advanced OCR-based health metrics extraction with performance optimization and batch processing.

Features:
- Multi-engine OCR support with automatic selection
- Hardware optimization (CPU/GPU detection and acceleration)
- Batch processing with parallel execution
- Advanced image preprocessing and enhancement
- Smart metric extraction with validation
- Clean CSV output with comprehensive logging
- Cross-platform compatibility
- Performance monitoring and statistics
- Automatic error recovery and retry mechanisms
- Continuous monitoring mode
- Multi-format output (CSV, JSON, XML)

Usage:
    # Single image
    python zepp_life_text_extractor_optimized.py image.png
    
    # Batch processing
    python zepp_life_text_extractor_optimized.py --batch screenshots/ --output results.csv
    
    # Continuous monitoring
    python zepp_life_text_extractor_optimized.py --monitor screenshots/ --interval 30
    
    # Performance analysis
    python zepp_life_text_extractor_optimized.py image.png --performance --benchmark
"""

import cv2
import numpy as np
import argparse
import logging
import csv
import json
import xml.etree.ElementTree as ET
import re
import time
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import statistics
import psutil
import platform

# Import our modules
from ocr_engines import MultiEngineOCR, OCRResult
from cv_utils import ImagePreprocessor, ZeppLifeRegionDetector, draw_debug_regions, validate_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance monitoring data"""
    processing_time: float = 0.0
    ocr_time: float = 0.0
    preprocessing_time: float = 0.0
    regions_detected: int = 0
    metrics_extracted: int = 0
    ocr_engine_used: str = ""
    image_size: Tuple[int, int] = (0, 0)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    hardware_acceleration: bool = False
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class HealthMetrics:
    """Enhanced container for extracted health metrics"""
    date: str = ""
    time: str = ""
    heart_rate: int = 0
    stress: int = 0
    steps: int = 0
    distance: float = 0.0
    calories: int = 0
    blood_oxygen: int = 0
    
    # Additional metadata
    image_source: str = ""
    extraction_confidence: float = 0.0
    processing_time: float = 0.0
    ocr_engine: str = ""
    performance_metrics: Optional[PerformanceMetrics] = field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert performance_metrics to dict if present
        if self.performance_metrics:
            data['performance_metrics'] = self.performance_metrics.to_dict()
        return data
        
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row"""
        return [
            self.date,
            self.time,
            str(self.heart_rate),
            str(self.stress),
            str(self.steps),
            str(self.distance),
            str(self.calories),
            str(self.blood_oxygen)
        ]
        
    def to_extended_csv_row(self) -> List[str]:
        """Convert to extended CSV row with metadata"""
        base_row = self.to_csv_row()
        extended_data = [
            self.image_source,
            f"{self.extraction_confidence:.2f}",
            f"{self.processing_time:.3f}",
            self.ocr_engine
        ]
        return base_row + extended_data

class AdvancedTextCleaner:
    """Enhanced utilities for cleaning and parsing OCR text"""
    
    @staticmethod
    def clean_numeric_text(text: str) -> str:
        """Advanced text cleaning for numeric extraction"""
        if not text:
            return ""
            
        # Remove common OCR errors
        corrections = {
            'O': '0', 'o': '0', 'l': '1', 'I': '1', '|': '1',
            'S': '5', 's': '5', 'B': '8', 'G': '6', 'q': '9',
            'Z': '2', 'z': '2', 'T': '7', 'A': '4'
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Keep only digits, decimal points, and specific units
        cleaned = re.sub(r'[^0-9.%kmKMBPM]', '', text)
        return cleaned.strip()
        
    @staticmethod
    def extract_number(text: str, allow_decimal: bool = True) -> Optional[float]:
        """Extract number with enhanced pattern matching"""
        cleaned = AdvancedTextCleaner.clean_numeric_text(text)
        
        # Multiple pattern attempts
        patterns = [
            r'(\d+\.\d+)' if allow_decimal else r'(\d+)',  # Decimal or integer
            r'(\d+)',  # Integer fallback
            r'(\d+)(?:\s*[a-zA-Z]*)',  # Number followed by letters
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
        
    @staticmethod
    def extract_percentage(text: str) -> Optional[int]:
        """Extract percentage with validation"""
        # Look for percentage patterns
        patterns = [
            r'(\d+)\s*%',
            r'(\d+)(?=\s*percent)',
            r'(\d+)(?=\s*pct)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = int(match.group(1))
                    if 0 <= value <= 100:
                        return value
                except ValueError:
                    continue
        
        # Fallback to general number extraction
        number = AdvancedTextCleaner.extract_number(text, allow_decimal=False)
        if number and 0 <= number <= 100:
            return int(number)
        return None
        
    @staticmethod
    def extract_distance(text: str) -> Optional[float]:
        """Extract distance with unit conversion and validation"""
        # Look for distance patterns
        patterns = [
            r'(\d+\.\d+)\s*km',
            r'(\d+\.\d+)\s*kilometers?',
            r'(\d+)\s*km',
            r'(\d+)\s*meters?',
            r'(\d+\.\d+)\s*m(?!\w)',  # meters but not part of another word
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert meters to kilometers if needed
                    if 'm' in pattern and 'km' not in pattern:
                        value = value / 1000
                    if 0 <= value <= 1000:  # Reasonable range
                        return round(value, 3)
                except ValueError:
                    continue
        
        # Fallback to general number extraction
        number = AdvancedTextCleaner.extract_number(text)
        if number and 0 <= number <= 1000:
            return round(number, 3)
        return None

class OptimizedZeppLifeExtractor:
    """Optimized extractor with advanced features and performance monitoring"""
    
    def __init__(self, debug: bool = False, visualize: bool = False, 
                 performance_monitoring: bool = False, max_workers: int = None):
        self.debug = debug
        self.visualize = visualize
        self.performance_monitoring = performance_monitoring
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        
        # Performance tracking
        self.performance_stats = []
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        # Initialize components
        logger.info("Initializing optimized Zepp Life extractor...")
        self.ocr_engine = MultiEngineOCR()
        self.preprocessor = ImagePreprocessor(use_gpu=self.ocr_engine.hardware_config.has_opencl)
        self.region_detector = ZeppLifeRegionDetector()
        self.text_cleaner = AdvancedTextCleaner()
        
        # System information
        self.system_info = self._get_system_info()
        
        logger.info(f"Optimizer initialized with {self.max_workers} workers")
        logger.info(f"Hardware: {self.system_info['platform']} - {self.system_info['cpu_cores']} cores")
        logger.info(f"Memory: {self.system_info['total_memory_gb']:.1f}GB")
        logger.info(f"OCR acceleration: {self.ocr_engine.hardware_config.has_opencl}")
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': platform.python_version(),
        }
        
    def extract_metrics(self, image_path: str) -> Optional[HealthMetrics]:
        """Extract health metrics with performance monitoring"""
        start_time = time.time()
        metrics = HealthMetrics()
        perf_metrics = PerformanceMetrics()
        
        try:
            # Validate image
            if not validate_image(image_path):
                logger.error(f"Invalid image: {image_path}")
                return None
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
                
            perf_metrics.image_size = (image.shape[1], image.shape[0])
            perf_metrics.timestamp = datetime.now().isoformat()
            
            logger.info(f"Processing image: {Path(image_path).name}")
            logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
            
            # Performance monitoring
            preprocess_start = time.time()
            
            # Detect regions with enhanced algorithm
            regions = self.region_detector.adaptive_region_detection(image)
            perf_metrics.regions_detected = len(regions)
            
            preprocess_end = time.time()
            perf_metrics.preprocessing_time = preprocess_end - preprocess_start
            
            logger.info(f"Detected {len(regions)} regions")
            
            # Draw debug visualization if requested
            if self.visualize:
                debug_image = draw_debug_regions(image, regions)
                debug_path = Path(image_path).with_suffix('.debug.png')
                cv2.imwrite(str(debug_path), debug_image)
                logger.info(f"Debug visualization saved: {debug_path}")
            
            # Extract text from each region
            now = datetime.now()
            metrics.date = now.strftime("%Y-%m-%d")
            metrics.time = now.strftime("%H:%M:%S")
            metrics.image_source = str(Path(image_path).name)
            
            ocr_start = time.time()
            confidence_scores = []
            extracted_count = 0
            
            for region_name, region in regions.items():
                try:
                    # Extract region image
                    x, y, w, h = region.bbox
                    
                    # Enhanced preprocessing for this region
                    processed_roi = self.preprocessor.preprocess_for_ocr(image, region.bbox)
                    
                    # Save debug ROI if requested
                    if self.debug:
                        roi_path = Path(image_path).parent / f"debug_{region_name}_roi.png"
                        cv2.imwrite(str(roi_path), processed_roi)
                    
                    # Extract text using OCR with retry mechanism
                    ocr_results = self._extract_with_retry(processed_roi, max_retries=2)
                    
                    if ocr_results:
                        # Combine all text from region
                        region_text = " ".join([result.text for result in ocr_results])
                        avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
                        confidence_scores.append(avg_confidence)
                        
                        logger.debug(f"{region_name}: '{region_text}' (confidence: {avg_confidence:.2f})")
                        
                        # Parse based on metric type with enhanced algorithms
                        value = self._parse_metric_value_enhanced(region_text, region.metric_type)
                        if value is not None:
                            setattr(metrics, region_name, value)
                            extracted_count += 1
                            logger.info(f"Extracted {region_name}: {value}")
                        else:
                            logger.warning(f"Could not parse {region_name} from: '{region_text}'")
                    else:
                        logger.warning(f"No OCR results for {region_name}")
                        
                except Exception as e:
                    logger.error(f"Error processing region {region_name}: {e}")
                    continue
            
            ocr_end = time.time()
            perf_metrics.ocr_time = ocr_end - ocr_start
            perf_metrics.metrics_extracted = extracted_count
            perf_metrics.ocr_engine_used = self.ocr_engine.engines[0].engine_name if self.ocr_engine.engines else "none"
            
            # Calculate overall confidence
            if confidence_scores:
                metrics.extraction_confidence = statistics.mean(confidence_scores)
            
            # Complete performance metrics
            end_time = time.time()
            total_time = end_time - start_time
            perf_metrics.processing_time = total_time
            perf_metrics.cpu_usage = psutil.cpu_percent()
            perf_metrics.memory_usage = psutil.virtual_memory().percent
            perf_metrics.hardware_acceleration = self.ocr_engine.hardware_config.has_opencl
            
            metrics.processing_time = total_time
            metrics.ocr_engine = perf_metrics.ocr_engine_used
            metrics.performance_metrics = perf_metrics
            
            self.total_processed += 1
            
            if self.performance_monitoring:
                logger.info(f"Performance: {total_time:.3f}s total, {perf_metrics.ocr_time:.3f}s OCR, "
                           f"{extracted_count}/{len(regions)} metrics extracted")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            self.total_errors += 1
            return None
            
    def _extract_with_retry(self, processed_roi: np.ndarray, max_retries: int = 2) -> List[OCRResult]:
        """Extract text with retry mechanism for better reliability"""
        for attempt in range(max_retries + 1):
            try:
                ocr_results = self.ocr_engine.extract_text(processed_roi)
                if ocr_results:
                    return ocr_results
                    
                # If no results, try with different preprocessing for retry
                if attempt < max_retries:
                    # Apply different enhancement for retry
                    enhanced_roi = self._apply_alternative_preprocessing(processed_roi)
                    processed_roi = enhanced_roi
                    logger.debug(f"Retrying OCR with alternative preprocessing (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.warning(f"OCR attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    break
                    
        return []
        
    def _apply_alternative_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply alternative preprocessing for retry attempts"""
        try:
            # Different enhancement approach
            enhanced = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Different threshold method
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            
            # Different morphological operation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return processed
        except Exception as e:
            logger.warning(f"Alternative preprocessing failed: {e}")
            return image
            
    def _parse_metric_value_enhanced(self, text: str, metric_type: str) -> Optional[Any]:
        """Enhanced metric parsing with better accuracy"""
        try:
            if metric_type == 'heart_rate':
                # Enhanced heart rate extraction
                number = self.text_cleaner.extract_number(text, allow_decimal=False)
                if number and 30 <= number <= 220:  # Reasonable heart rate range
                    return int(number)
                    
            elif metric_type == 'stress':
                # Enhanced stress level extraction
                number = self.text_cleaner.extract_number(text, allow_decimal=False)
                if number and 0 <= number <= 100:
                    return int(number)
                    
            elif metric_type == 'steps':
                # Enhanced steps extraction
                number = self.text_cleaner.extract_number(text, allow_decimal=False)
                if number and 0 <= number <= 100000:  # Reasonable steps range
                    return int(number)
                    
            elif metric_type == 'distance':
                # Enhanced distance extraction
                distance = self.text_cleaner.extract_distance(text)
                if distance and 0 <= distance <= 100:  # Reasonable distance range
                    return round(distance, 2)
                    
            elif metric_type == 'calories':
                # Enhanced calories extraction
                number = self.text_cleaner.extract_number(text, allow_decimal=False)
                if number and 0 <= number <= 10000:  # Reasonable calories range
                    return int(number)
                    
            elif metric_type == 'blood_oxygen':
                # Enhanced blood oxygen extraction
                percentage = self.text_cleaner.extract_percentage(text)
                if percentage and 70 <= percentage <= 100:  # Reasonable SpO2 range
                    return percentage
                    
            return None
            
        except Exception as e:
            logger.error(f"Error parsing {metric_type} from '{text}': {e}")
            return None
            
    def batch_process(self, input_dir: str, output_path: str, 
                     file_patterns: List[str] = None) -> Dict[str, Any]:
        """Process multiple images in batch with parallel execution"""
        if file_patterns is None:
            file_patterns = ["*.png", "*.jpg", "*.jpeg"]
            
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return {"success": False, "error": "Directory not found"}
            
        # Find all matching files
        image_files = []
        for pattern in file_patterns:
            image_files.extend(input_path.glob(pattern))
            
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {"success": False, "error": "No image files found"}
            
        logger.info(f"Found {len(image_files)} images for batch processing")
        
        # Process files in parallel
        start_time = time.time()
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.extract_metrics, str(img_path)): img_path 
                for img_path in image_files
            }
            
            # Collect results
            for future in futures:
                img_path = futures[future]
                try:
                    metrics = future.result(timeout=30)  # 30 second timeout per image
                    if metrics:
                        all_metrics.append(metrics)
                        logger.info(f"Processed: {img_path.name}")
                    else:
                        logger.warning(f"Failed to extract metrics from: {img_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    self.total_errors += 1
                    
        processing_time = time.time() - start_time
        
        # Save results
        if all_metrics:
            success = self.save_batch_results(all_metrics, output_path)
            if success:
                logger.info(f"Batch processing completed: {len(all_metrics)} files processed in {processing_time:.2f}s")
                return {
                    "success": True,
                    "processed_count": len(all_metrics),
                    "total_files": len(image_files),
                    "processing_time": processing_time,
                    "output_file": output_path
                }
        
        return {"success": False, "error": "No metrics extracted"}
        
    def save_batch_results(self, metrics_list: List[HealthMetrics], output_path: str) -> bool:
        """Save batch processing results to file"""
        try:
            output_file = Path(output_path)
            
            if output_file.suffix.lower() == '.json':
                return self._save_to_json(metrics_list, output_path)
            elif output_file.suffix.lower() == '.xml':
                return self._save_to_xml(metrics_list, output_path)
            else:
                return self._save_to_csv(metrics_list, output_path)
                
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
            return False
            
    def _save_to_csv(self, metrics_list: List[HealthMetrics], output_path: str) -> bool:
        """Save to CSV with extended metadata"""
        try:
            headers = ['Date', 'Time', 'Heart rate', 'Stress', 'Steps', 'Distance', 'Calories', 'Blood Oxygen']
            extended_headers = headers + ['Image Source', 'Confidence', 'Processing Time', 'OCR Engine']
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(extended_headers)
                
                for metrics in metrics_list:
                    writer.writerow(metrics.to_extended_csv_row())
                    
            logger.info(f"CSV results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
            
    def _save_to_json(self, metrics_list: List[HealthMetrics], output_path: str) -> bool:
        """Save to JSON format"""
        try:
            data = {
                "extraction_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_records": len(metrics_list),
                    "system_info": self.system_info,
                    "processing_stats": {
                        "total_processed": self.total_processed,
                        "total_errors": self.total_errors,
                        "success_rate": (self.total_processed / (self.total_processed + self.total_errors)) * 100 if (self.total_processed + self.total_errors) > 0 else 0
                    }
                },
                "health_metrics": [metrics.to_dict() for metrics in metrics_list]
            }
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False)
                
            logger.info(f"JSON results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return False
            
    def _save_to_xml(self, metrics_list: List[HealthMetrics], output_path: str) -> bool:
        """Save to XML format"""
        try:
            root = ET.Element("ZeppLifeData")
            
            # Metadata
            metadata = ET.SubElement(root, "Metadata")
            ET.SubElement(metadata, "Timestamp").text = datetime.now().isoformat()
            ET.SubElement(metadata, "TotalRecords").text = str(len(metrics_list))
            ET.SubElement(metadata, "TotalProcessed").text = str(self.total_processed)
            ET.SubElement(metadata, "TotalErrors").text = str(self.total_errors)
            
            # Health metrics
            metrics_element = ET.SubElement(root, "HealthMetrics")
            
            for metrics in metrics_list:
                record = ET.SubElement(metrics_element, "Record")
                
                for key, value in metrics.to_dict().items():
                    if key != 'performance_metrics':  # Skip complex nested objects
                        ET.SubElement(record, key.replace('_', '').title()).text = str(value)
                        
            # Write to file
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"XML results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to XML: {e}")
            return False
            
    def continuous_monitor(self, watch_dir: str, output_path: str, 
                          interval: int = 30, max_files_per_batch: int = 10) -> None:
        """Continuously monitor directory for new images and process them"""
        watch_path = Path(watch_dir)
        if not watch_path.exists():
            logger.error(f"Watch directory not found: {watch_dir}")
            return
            
        logger.info(f"Starting continuous monitoring of: {watch_dir}")
        logger.info(f"Check interval: {interval} seconds")
        logger.info(f"Output file: {output_path}")
        logger.info("Press Ctrl+C to stop monitoring")
        
        processed_files = set()
        
        try:
            while True:
                try:
                    # Find new image files
                    current_files = set()
                    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
                        current_files.update(watch_path.glob(pattern))
                        
                    new_files = current_files - processed_files
                    
                    if new_files:
                        new_files_list = list(new_files)[:max_files_per_batch]
                        logger.info(f"Found {len(new_files_list)} new files")
                        
                        # Process new files
                        new_metrics = []
                        for file_path in new_files_list:
                            metrics = self.extract_metrics(str(file_path))
                            if metrics:
                                new_metrics.append(metrics)
                                processed_files.add(file_path)
                                logger.info(f"Processed: {file_path.name}")
                            else:
                                logger.warning(f"Failed to process: {file_path.name}")
                                
                        # Append to output file
                        if new_metrics:
                            self._append_to_output(new_metrics, output_path)
                            logger.info(f"Added {len(new_metrics)} records to {output_path}")
                            
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error during monitoring cycle: {e}")
                    time.sleep(interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in continuous monitoring: {e}")
            
        logger.info("Continuous monitoring ended")
        
    def _append_to_output(self, metrics_list: List[HealthMetrics], output_path: str) -> None:
        """Append new metrics to existing output file"""
        try:
            output_file = Path(output_path)
            
            # Check if file exists to determine if headers are needed
            file_exists = output_file.exists()
            
            with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers if new file
                if not file_exists:
                    headers = ['Date', 'Time', 'Heart rate', 'Stress', 'Steps', 'Distance', 'Calories', 'Blood Oxygen']
                    extended_headers = headers + ['Image Source', 'Confidence', 'Processing Time', 'OCR Engine']
                    writer.writerow(extended_headers)
                    
                # Write metrics
                for metrics in metrics_list:
                    writer.writerow(metrics.to_extended_csv_row())
                    
        except Exception as e:
            logger.error(f"Error appending to output file: {e}")
            
    def generate_performance_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_stats:
            logger.warning("No performance data available")
            return {}
            
        # Calculate statistics
        execution_times = [stat['execution_time'] for stat in self.performance_stats if stat['success']]
        cpu_usage = [stat['cpu_usage_avg'] for stat in self.performance_stats]
        memory_usage = [stat['memory_usage_avg'] for stat in self.performance_stats]
        
        report = {
            "performance_summary": {
                "total_operations": len(self.performance_stats),
                "successful_operations": len(execution_times),
                "failed_operations": len(self.performance_stats) - len(execution_times),
                "success_rate": (len(execution_times) / len(self.performance_stats)) * 100 if self.performance_stats else 0,
                "total_processing_time": sum(execution_times),
                "average_processing_time": statistics.mean(execution_times) if execution_times else 0,
                "median_processing_time": statistics.median(execution_times) if execution_times else 0,
                "min_processing_time": min(execution_times) if execution_times else 0,
                "max_processing_time": max(execution_times) if execution_times else 0,
            },
            "resource_usage": {
                "average_cpu_usage": statistics.mean(cpu_usage) if cpu_usage else 0,
                "peak_cpu_usage": max(cpu_usage) if cpu_usage else 0,
                "average_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
                "peak_memory_usage": max(memory_usage) if memory_usage else 0,
            },
            "system_info": self.system_info,
            "configuration": {
                "max_workers": self.max_workers,
                "ocr_engines_available": len(self.ocr_engine.engines),
                "hardware_acceleration": self.ocr_engine.hardware_config.has_opencl,
                "debug_mode": self.debug,
                "visualization_enabled": self.visualize,
            },
            "detailed_stats": self.performance_stats[-100:] if len(self.performance_stats) > 100 else self.performance_stats  # Last 100 entries
        }
        
        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Performance report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error saving performance report: {e}")
                
        return report

def main():
    """Enhanced main entry point with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description='Optimized Zepp Life health metrics extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image processing
  %(prog)s image.png
  
  # Batch processing
  %(prog)s --batch screenshots/ --output results.csv
  
  # Continuous monitoring
  %(prog)s --monitor screenshots/ --interval 30 --output live_results.csv
  
  # Advanced options
  %(prog)s image.png --debug --visualize --performance --format json
        """
    )
    
    # Input options
    parser.add_argument('input', nargs='?', help='Input image file path')
    parser.add_argument('--batch', help='Batch process directory')
    parser.add_argument('--monitor', help='Continuously monitor directory')
    
    # Output options
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['csv', 'json', 'xml'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--append', action='store_true', default=True,
                       help='Append to existing file (default: True)')
    
    # Processing options
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum files per monitoring batch (default: 10)')
    
    # Debug and analysis options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (save intermediate images)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of detected regions')
    parser.add_argument('--performance', action='store_true',
                       help='Enable performance monitoring')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Report options
    parser.add_argument('--performance-report', help='Generate performance report to file')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input options
    if not args.input and not args.batch and not args.monitor:
        parser.error("Must specify input image, --batch directory, or --monitor directory")
    
    # Create optimized extractor
    extractor = OptimizedZeppLifeExtractor(
        debug=args.debug,
        visualize=args.visualize,
        performance_monitoring=args.performance,
        max_workers=args.workers
    )
    
    try:
        if args.monitor:
            # Continuous monitoring mode
            output_path = args.output or f"zepp_life_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            extractor.continuous_monitor(args.monitor, output_path, args.interval, args.max_files)
            
        elif args.batch:
            # Batch processing mode
            output_path = args.output or f"zepp_life_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
            result = extractor.batch_process(args.batch, output_path)
            
            if result["success"]:
                print(f"\n✅ Batch processing completed!")
                print(f"📊 Processed: {result['processed_count']}/{result['total_files']} files")
                print(f"⏱️  Time: {result['processing_time']:.2f} seconds")
                print(f"💾 Output: {result['output_file']}")
            else:
                print(f"\n❌ Batch processing failed: {result.get('error', 'Unknown error')}")
                return 1
                
        elif args.input:
            # Single image processing
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return 1
                
            # Regular single image processing
            logger.info("Starting metric extraction...")
            metrics = extractor.extract_metrics(str(input_path))
            
            if metrics is None:
                logger.error("Failed to extract metrics")
                return 1
            
            # Display results
            print("\n=== Extracted Health Metrics ===")
            print(f"📅 Date: {metrics.date}")
            print(f"🕐 Time: {metrics.time}")
            print(f"❤️  Heart Rate: {metrics.heart_rate} BPM")
            print(f"😰 Stress: {metrics.stress}")
            print(f"👣 Steps: {metrics.steps}")
            print(f"📏 Distance: {metrics.distance} km")
            print(f"🔥 Calories: {metrics.calories} kcal")
            print(f"🩸 Blood Oxygen: {metrics.blood_oxygen}%")
            
            if args.performance and metrics.performance_metrics:
                print(f"\n📊 Performance:")
                print(f"   Processing: {metrics.performance_metrics.processing_time:.3f}s")
                print(f"   OCR: {metrics.performance_metrics.ocr_time:.3f}s")
                print(f"   Confidence: {metrics.extraction_confidence:.2f}")
                print(f"   Engine: {metrics.ocr_engine}")
            
            print("=" * 35)
            
            # Save to file
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.parent / f'zepp_life_metrics.{args.format}'
            
            if args.format == 'json':
                success = extractor._save_to_json([metrics], str(output_path))
            elif args.format == 'xml':
                success = extractor._save_to_xml([metrics], str(output_path))
            else:
                success = extractor._save_to_csv([metrics], str(output_path))
            
            if success:
                print(f"💾 Results saved to: {output_path}")
            else:
                logger.error("Failed to save results")
                return 1
        
        # Generate performance report if requested
        if args.performance_report:
            report = extractor.generate_performance_report(args.performance_report)
            if report:
                print(f"📊 Performance report saved to: {args.performance_report}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
