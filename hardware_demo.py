#!/usr/bin/env python3
"""
Hardware Optimization Demo for Zepp Life Text Extractor
Demonstrates the automatic hardware detection and optimization capabilities
"""

import time
import os
from pathlib import Path

def print_banner():
    """Print the demo banner"""
    print("="*80)
    print("🚀 ZEPP LIFE TEXT EXTRACTOR - HARDWARE OPTIMIZATION DEMO")
    print("="*80)
    print()

def demo_hardware_detection():
    """Demonstrate hardware detection capabilities"""
    print("🔍 STEP 1: Hardware Detection")
    print("-" * 40)
    
    try:
        from hardware_optimizer import detect_and_optimize
        
        print("⏳ Detecting system hardware capabilities...")
        optimizer = detect_and_optimize()
        
        # Print detailed hardware info
        optimizer.print_hardware_info()
        
        # Show configuration
        config = optimizer.get_processing_config()
        print("📋 Optimized Configuration:")
        print(f"   🖼️  OpenCV Settings: {config['opencv_settings']}")
        print(f"   🔤 OCR Settings: {config['ocr_settings']}")
        print()
        
        return optimizer
        
    except ImportError:
        print("❌ Hardware optimizer not available")
        return None

def demo_extractor_comparison():
    """Compare original vs optimized extractor initialization"""
    print("⚖️  STEP 2: Extractor Comparison")
    print("-" * 40)
    
    # Test original extractor
    print("📊 Testing Original Extractor...")
    try:
        from zepp_life_text_extractor import ZeppLifeTextExtractor
        
        start_time = time.time()
        original_extractor = ZeppLifeTextExtractor()
        original_init_time = time.time() - start_time
        
        print(f"   ⏱️  Initialization time: {original_init_time:.2f}s")
        print(f"   🔧 OCR Engine: {getattr(original_extractor, 'selected_engine', 'tesseract')}")
        print(f"   🎯 Hardware optimization: Basic")
        
    except ImportError:
        print("   ❌ Original extractor not available")
        original_init_time = 0
    
    print()
    
    # Test optimized extractor
    print("🚀 Testing Hardware-Optimized Extractor...")
    try:
        from zepp_life_text_extractor_optimized import HardwareOptimizedZeppExtractor
        
        start_time = time.time()
        optimized_extractor = HardwareOptimizedZeppExtractor()
        optimized_init_time = time.time() - start_time
        
        print(f"   ⏱️  Initialization time: {optimized_init_time:.2f}s")
        print(f"   🔧 OCR Engine: {optimized_extractor.selected_engine}")
        print(f"   🎯 Hardware optimization: {optimized_extractor.performance_stats['hardware_acceleration']}")
        
        # Show performance comparison
        if original_init_time > 0:
            improvement = ((original_init_time - optimized_init_time) / original_init_time * 100)
            print(f"   📈 Initialization improvement: {improvement:+.1f}%")
        
        return optimized_extractor
        
    except ImportError:
        print("   ❌ Optimized extractor not available")
        return None

def demo_available_features(extractor):
    """Show available features based on system capabilities"""
    print("🎛️  STEP 3: Available Features")
    print("-" * 40)
    
    if not extractor:
        print("❌ No extractor available for feature demo")
        return
    
    print("✅ Core Features:")
    print("   📷 Cross-platform screenshot processing")
    print("   🔤 Multi-engine OCR support")
    print("   📊 CSV data export")
    print("   🔄 Batch processing")
    print()
    
    print("🚀 Hardware Optimizations:")
    
    if hasattr(extractor, 'hardware_optimizer') and extractor.hardware_optimizer:
        hw_info = extractor.hardware_optimizer.hardware_info
        
        # CPU optimizations
        print(f"   ⚙️  Multi-threading: {hw_info.optimal_threads} threads")
        
        # GPU optimizations
        if hw_info.has_gpu:
            print("   🎮 GPU acceleration: Available")
            if hw_info.cuda_available:
                print("   🔥 CUDA support: Enabled")
        else:
            print("   🎮 GPU acceleration: CPU-only mode")
            
        # OpenCL optimizations
        if hw_info.opencl_available:
            print("   ⚡ OpenCL acceleration: Enabled")
            
        # OCR engine optimization
        print(f"   🔤 Optimized OCR engine: {hw_info.recommended_engine}")
        
        # Memory optimization
        print(f"   💾 Memory-aware processing: {hw_info.total_memory_gb:.1f}GB available")
        
    else:
        print("   ❌ Hardware optimization not available")
    
    print()

def demo_usage_examples():
    """Show usage examples"""
    print("📚 STEP 4: Usage Examples")
    print("-" * 40)
    
    print("🖥️  Command Line Usage:")
    print("   # Basic processing")
    print("   python zepp_life_text_extractor_optimized.py")
    print()
    print("   # Continuous monitoring")
    print("   python zepp_life_text_extractor_optimized.py --monitor")
    print()
    print("   # Custom settings")
    print("   python zepp_life_text_extractor_optimized.py --no-delete --output my_data.csv")
    print()
    print("   # Disable optimizations (compatibility mode)")
    print("   python zepp_life_text_extractor_optimized.py --no-optimization")
    print()
    
    print("🐍 Python API Usage:")
    print("   from zepp_life_text_extractor_optimized import HardwareOptimizedZeppExtractor")
    print("   ")
    print("   # Initialize with automatic optimization")
    print("   extractor = HardwareOptimizedZeppExtractor()")
    print("   ")
    print("   # Process all screenshots")
    print("   extractor.process_all_screenshots()")
    print("   ")
    print("   # Continuous monitoring")
    print("   extractor.continuous_monitoring(interval=30)")
    print()

def demo_performance_tips():
    """Show performance optimization tips"""
    print("💡 STEP 5: Performance Optimization Tips")
    print("-" * 40)
    
    print("🚀 For CPU-only systems:")
    print("   • EasyOCR CPU mode provides better accuracy than Tesseract")
    print("   • OpenCL acceleration improves image preprocessing speed")
    print("   • Multi-threading optimizes based on your CPU cores")
    print()
    
    print("🎮 For systems with dedicated GPU:")
    print("   • Install: pip install easyocr (for GPU-accelerated OCR)")
    print("   • CUDA support enables GPU image processing")
    print("   • Significantly faster processing for large batches")
    print()
    
    print("💾 Memory optimization:")
    print("   • System automatically adjusts based on available RAM")
    print("   • Large memory systems can use more advanced OCR engines")
    print("   • Batch processing optimized for your system capacity")
    print()
    
    print("📊 Cross-platform considerations:")
    print("   • Windows: Automatic Tesseract detection in Program Files")
    print("   • Linux: Uses system PATH for OCR binaries")
    print("   • macOS: Homebrew integration for easy setup")
    print()

def main():
    """Run the complete hardware optimization demo"""
    print_banner()
    
    # Step 1: Hardware Detection
    optimizer = demo_hardware_detection()
    
    # Step 2: Extractor Comparison
    extractor = demo_extractor_comparison()
    
    # Step 3: Available Features
    demo_available_features(extractor)
    
    # Step 4: Usage Examples
    demo_usage_examples()
    
    # Step 5: Performance Tips
    demo_performance_tips()
    
    # Final message
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print("Your Zepp Life Text Extractor is now hardware-optimized and ready to use!")
    print("Run 'python zepp_life_text_extractor_optimized.py' to start processing.")
    print("="*80)

if __name__ == "__main__":
    main()
