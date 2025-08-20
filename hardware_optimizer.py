#!/usr/bin/env python3
"""
Cross-Platform Hardware Optimizer for Zepp Life Text Extractor
Automatically detects and optimizes for available CPU/GPU resources
"""

import os
import sys
import platform
import subprocess
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import with fallback handling
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


@dataclass
class HardwareInfo:
    """Hardware information container"""
    platform_name: str
    cpu_count: int
    cpu_physical_cores: int
    total_memory_gb: float
    has_gpu: bool
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_mb: List[int]
    cuda_available: bool
    opencl_available: bool
    optimal_threads: int
    recommended_engine: str


class HardwareOptimizer:
    """Cross-platform hardware detection and optimization"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.hardware_info = None
        self._detect_hardware()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for hardware optimizer"""
        logger = logging.getLogger('HardwareOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_hardware(self) -> None:
        """Detect system hardware capabilities"""
        try:
            # Basic system info
            platform_name = platform.system()
            cpu_count = os.cpu_count() or 1
            
            # Get physical CPU cores
            cpu_physical_cores = cpu_count
            if HAS_PSUTIL:
                cpu_physical_cores = psutil.cpu_count(logical=False) or cpu_count
            
            # Memory info
            total_memory_gb = self._get_memory_info()
            
            # GPU detection
            has_gpu, gpu_count, gpu_names, gpu_memory_mb = self._detect_gpu()
            
            # CUDA and OpenCL detection
            cuda_available = self._detect_cuda()
            opencl_available = self._detect_opencl()
            
            # Calculate optimal settings
            optimal_threads = self._calculate_optimal_threads(cpu_physical_cores)
            recommended_engine = self._recommend_ocr_engine(
                has_gpu, cuda_available, total_memory_gb
            )
            
            self.hardware_info = HardwareInfo(
                platform_name=platform_name,
                cpu_count=cpu_count,
                cpu_physical_cores=cpu_physical_cores,
                total_memory_gb=total_memory_gb,
                has_gpu=has_gpu,
                gpu_count=gpu_count,
                gpu_names=gpu_names,
                gpu_memory_mb=gpu_memory_mb,
                cuda_available=cuda_available,
                opencl_available=opencl_available,
                optimal_threads=optimal_threads,
                recommended_engine=recommended_engine
            )
            
            self.logger.info("Hardware detection completed successfully")
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            # Fallback to basic configuration
            self._create_fallback_config()
    
    def _get_memory_info(self) -> float:
        """Get total system memory in GB"""
        try:
            if HAS_PSUTIL:
                return round(psutil.virtual_memory().total / (1024**3), 2)
            else:
                # Fallback methods for different platforms
                system = platform.system().lower()
                
                if system == "windows":
                    return self._get_windows_memory()
                elif system == "linux":
                    return self._get_linux_memory()
                elif system == "darwin":  # macOS
                    return self._get_macos_memory()
                else:
                    return 8.0  # Default fallback
        except:
            return 8.0  # Default fallback
    
    def _get_windows_memory(self) -> float:
        """Get Windows memory using wmic"""
        try:
            result = subprocess.run(
                ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and line.strip() != 'TotalPhysicalMemory':
                        memory_bytes = int(line.strip())
                        return round(memory_bytes / (1024**3), 2)
        except:
            pass
        return 8.0
    
    def _get_linux_memory(self) -> float:
        """Get Linux memory from /proc/meminfo"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_kb = int(line.split()[1])
                        return round(memory_kb / (1024**2), 2)
        except:
            pass
        return 8.0
    
    def _get_macos_memory(self) -> float:
        """Get macOS memory using sysctl"""
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                memory_bytes = int(result.stdout.strip())
                return round(memory_bytes / (1024**3), 2)
        except:
            pass
        return 8.0
    
    def _detect_gpu(self) -> Tuple[bool, int, List[str], List[int]]:
        """Detect GPU presence and capabilities"""
        has_gpu = False
        gpu_count = 0
        gpu_names = []
        gpu_memory_mb = []
        
        try:
            # Try GPUtil first (most reliable)
            if HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                if gpus:
                    has_gpu = True
                    gpu_count = len(gpus)
                    gpu_names = [gpu.name for gpu in gpus]
                    gpu_memory_mb = [int(gpu.memoryTotal) for gpu in gpus]
                    return has_gpu, gpu_count, gpu_names, gpu_memory_mb
            
            # Fallback: Platform-specific GPU detection
            system = platform.system().lower()
            
            if system == "windows":
                has_gpu, gpu_count, gpu_names, gpu_memory_mb = self._detect_windows_gpu()
            elif system == "linux":
                has_gpu, gpu_count, gpu_names, gpu_memory_mb = self._detect_linux_gpu()
            elif system == "darwin":
                has_gpu, gpu_count, gpu_names, gpu_memory_mb = self._detect_macos_gpu()
                
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
        
        return has_gpu, gpu_count, gpu_names, gpu_memory_mb
    
    def _detect_windows_gpu(self) -> Tuple[bool, int, List[str], List[int]]:
        """Detect GPU on Windows using wmic"""
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, text=True, timeout=15
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_names = []
                for line in lines:
                    line = line.strip()
                    if line and line != 'Name' and 'Microsoft' not in line:
                        gpu_names.append(line)
                
                if gpu_names:
                    return True, len(gpu_names), gpu_names, [0] * len(gpu_names)
        except:
            pass
        
        return False, 0, [], []
    
    def _detect_linux_gpu(self) -> Tuple[bool, int, List[str], List[int]]:
        """Detect GPU on Linux using lspci"""
        try:
            result = subprocess.run(
                ['lspci', '|', 'grep', '-i', 'vga'],
                shell=True, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                gpu_names = []
                for line in lines:
                    if 'VGA' in line or 'Display' in line:
                        # Extract GPU name
                        parts = line.split(': ')
                        if len(parts) > 1:
                            gpu_names.append(parts[1])
                
                if gpu_names:
                    return True, len(gpu_names), gpu_names, [0] * len(gpu_names)
        except:
            pass
        
        return False, 0, [], []
    
    def _detect_macos_gpu(self) -> Tuple[bool, int, List[str], List[int]]:
        """Detect GPU on macOS using system_profiler"""
        try:
            result = subprocess.run([
                'system_profiler', 'SPDisplaysDataType'
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                gpu_names = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('Chipset Model:'):
                        gpu_name = line.split(':', 1)[1].strip()
                        gpu_names.append(gpu_name)
                
                if gpu_names:
                    return True, len(gpu_names), gpu_names, [0] * len(gpu_names)
        except:
            pass
        
        return False, 0, [], []
    
    def _detect_cuda(self) -> bool:
        """Detect CUDA availability"""
        try:
            # Check if OpenCV has CUDA support
            if HAS_OPENCV:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                if cuda_devices > 0:
                    return True
            
            # Alternative: Check for nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi'], 
                    capture_output=True, timeout=5, 
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
        except Exception as e:
            self.logger.debug(f"CUDA detection error: {e}")
        
        return False
    
    def _detect_opencl(self) -> bool:
        """Detect OpenCL availability"""
        try:
            if HAS_OPENCV:
                # Check if OpenCV has OpenCL support
                return cv2.ocl.haveOpenCL()
        except:
            pass
        
        return False
    
    def _calculate_optimal_threads(self, physical_cores: int) -> int:
        """Calculate optimal thread count for processing"""
        # Use 75% of physical cores for optimal performance
        # but ensure at least 1 and at most 8 threads
        optimal = max(1, min(8, int(physical_cores * 0.75)))
        return optimal
    
    def _recommend_ocr_engine(self, has_gpu: bool, cuda_available: bool, memory_gb: float) -> str:
        """Recommend the best OCR engine based on hardware"""
        try:
            # Check if advanced OCR engines are available
            import easyocr
            if has_gpu and cuda_available and memory_gb >= 4:
                return "easyocr_gpu"
            elif memory_gb >= 2:
                return "easyocr_cpu"
        except ImportError:
            pass
        
        try:
            import paddleocr
            if has_gpu and memory_gb >= 3:
                return "paddleocr_gpu"
            elif memory_gb >= 2:
                return "paddleocr_cpu"
        except ImportError:
            pass
        
        # Default to Tesseract
        return "tesseract"
    
    def _create_fallback_config(self) -> None:
        """Create fallback configuration when detection fails"""
        self.hardware_info = HardwareInfo(
            platform_name=platform.system(),
            cpu_count=os.cpu_count() or 1,
            cpu_physical_cores=os.cpu_count() or 1,
            total_memory_gb=8.0,
            has_gpu=False,
            gpu_count=0,
            gpu_names=[],
            gpu_memory_mb=[],
            cuda_available=False,
            opencl_available=False,
            optimal_threads=2,
            recommended_engine="tesseract"
        )
    
    def get_opencv_settings(self) -> Dict[str, Any]:
        """Get optimized OpenCV settings based on hardware"""
        if not self.hardware_info:
            return {"use_gpu": False, "num_threads": 2}
        
        settings = {
            "use_gpu": False,
            "num_threads": self.hardware_info.optimal_threads,
            "use_opencl": False
        }
        
        try:
            if HAS_OPENCV:
                # Enable GPU acceleration if available
                if self.hardware_info.cuda_available and self.hardware_info.has_gpu:
                    settings["use_gpu"] = True
                    settings["backend"] = "cuda"
                elif self.hardware_info.opencl_available:
                    settings["use_opencl"] = True
                    cv2.ocl.setUseOpenCL(True)
                
                # Set optimal thread count for OpenCV
                cv2.setNumThreads(self.hardware_info.optimal_threads)
                
        except Exception as e:
            self.logger.warning(f"Failed to configure OpenCV: {e}")
        
        return settings
    
    def get_ocr_settings(self) -> Dict[str, Any]:
        """Get optimized OCR settings based on hardware"""
        if not self.hardware_info:
            return {"engine": "tesseract", "threads": 2}
        
        settings = {
            "engine": self.hardware_info.recommended_engine,
            "threads": self.hardware_info.optimal_threads,
            "gpu_enabled": False
        }
        
        # Configure based on recommended engine
        if "gpu" in self.hardware_info.recommended_engine and self.hardware_info.has_gpu:
            settings["gpu_enabled"] = True
        
        return settings
    
    def optimize_for_performance(self) -> None:
        """Apply performance optimizations based on hardware"""
        if not self.hardware_info:
            return
        
        try:
            # Set environment variables for performance
            if self.hardware_info.optimal_threads > 1:
                os.environ["OMP_NUM_THREADS"] = str(self.hardware_info.optimal_threads)
                os.environ["MKL_NUM_THREADS"] = str(self.hardware_info.optimal_threads)
                os.environ["NUMEXPR_NUM_THREADS"] = str(self.hardware_info.optimal_threads)
            
            # Configure OpenCV
            if HAS_OPENCV:
                cv2.setNumThreads(self.hardware_info.optimal_threads)
                
                # Enable optimizations
                cv2.setUseOptimized(True)
                
                # Configure OpenCL if available
                if self.hardware_info.opencl_available:
                    cv2.ocl.setUseOpenCL(True)
            
            self.logger.info(f"Performance optimizations applied for {self.hardware_info.platform_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply performance optimizations: {e}")
    
    def print_hardware_info(self) -> None:
        """Print detailed hardware information"""
        if not self.hardware_info:
            print("❌ Hardware detection failed")
            return
        
        print("\n" + "="*60)
        print("🖥️  SYSTEM HARDWARE INFORMATION")
        print("="*60)
        
        # System Info
        print(f"📋 Platform: {self.hardware_info.platform_name}")
        print(f"⚙️  CPU Cores: {self.hardware_info.cpu_count} logical, {self.hardware_info.cpu_physical_cores} physical")
        print(f"💾 Memory: {self.hardware_info.total_memory_gb} GB")
        print(f"🧵 Optimal Threads: {self.hardware_info.optimal_threads}")
        
        # GPU Info
        if self.hardware_info.has_gpu:
            print(f"\n🎮 GPU Information:")
            print(f"   Count: {self.hardware_info.gpu_count}")
            for i, gpu_name in enumerate(self.hardware_info.gpu_names):
                memory = f"{self.hardware_info.gpu_memory_mb[i]}MB" if i < len(self.hardware_info.gpu_memory_mb) and self.hardware_info.gpu_memory_mb[i] > 0 else "Unknown"
                print(f"   GPU {i+1}: {gpu_name} ({memory})")
        else:
            print(f"\n🎮 GPU: None detected")
        
        # Acceleration Support
        print(f"\n🚀 Acceleration Support:")
        print(f"   CUDA: {'✅ Available' if self.hardware_info.cuda_available else '❌ Not available'}")
        print(f"   OpenCL: {'✅ Available' if self.hardware_info.opencl_available else '❌ Not available'}")
        
        # Recommendations
        print(f"\n💡 Optimizations:")
        print(f"   Recommended OCR Engine: {self.hardware_info.recommended_engine}")
        print(f"   GPU Acceleration: {'✅ Enabled' if (self.hardware_info.has_gpu and ('gpu' in self.hardware_info.recommended_engine or self.hardware_info.cuda_available)) else '❌ CPU only'}")
        
        print("="*60 + "\n")
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get complete processing configuration"""
        if not self.hardware_info:
            return {
                "opencv_settings": {"use_gpu": False, "num_threads": 2},
                "ocr_settings": {"engine": "tesseract", "threads": 2, "gpu_enabled": False},
                "hardware_info": None
            }
        
        return {
            "opencv_settings": self.get_opencv_settings(),
            "ocr_settings": self.get_ocr_settings(),
            "hardware_info": self.hardware_info
        }


# Convenience function for easy integration
def detect_and_optimize() -> HardwareOptimizer:
    """
    Detect hardware and return optimizer instance
    
    Returns:
        HardwareOptimizer: Configured optimizer instance
    """
    optimizer = HardwareOptimizer()
    optimizer.optimize_for_performance()
    return optimizer


if __name__ == "__main__":
    # Test the hardware optimizer
    print("🔍 Detecting system hardware...")
    optimizer = detect_and_optimize()
    optimizer.print_hardware_info()
    
    # Show configuration
    config = optimizer.get_processing_config()
    print("\n📋 Processing Configuration:")
    print(f"   OpenCV: {config['opencv_settings']}")
    print(f"   OCR: {config['ocr_settings']}")
