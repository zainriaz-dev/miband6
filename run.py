#!/usr/bin/env python3
"""
Zepp Life Health Metrics Extractor - Cross-Platform Enhanced Startup Script
Comprehensive launcher with automatic screenshot discovery, batch processing,
and continuous monitoring features. Works on Windows, macOS, and Linux.
"""

import sys
import os
import time
import shutil
import platform
import subprocess
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =============================================================================
# CROSS-PLATFORM SETUP
# =============================================================================

# Platform detection
OS_NAME = platform.system().lower()
IS_WINDOWS = OS_NAME == 'windows'
IS_MACOS = OS_NAME == 'darwin'
IS_LINUX = OS_NAME == 'linux'

def get_tesseract_command():
    """Find Tesseract executable across platforms"""
    
    # Check environment variable first
    tesseract_cmd = os.environ.get('TESSERACT_CMD')
    if tesseract_cmd and shutil.which(tesseract_cmd):
        return tesseract_cmd
    
    if IS_WINDOWS:
        # Windows: check common installation paths
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            f"C:\\Users\\{os.getenv('USERNAME', '')}\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
                
        # Try PATH
        if shutil.which('tesseract.exe'):
            return 'tesseract.exe'
            
    elif IS_MACOS:
        # macOS: try brew path first, then standard locations
        try:
            result = subprocess.run(['brew', '--prefix', 'tesseract'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brew_path = result.stdout.strip() + '/bin/tesseract'
                if os.path.isfile(brew_path):
                    return brew_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Standard macOS paths
        mac_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract'
        ]
        
        for path in mac_paths:
            if os.path.isfile(path):
                return path
                
    # Linux and fallback: use which command
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return tesseract_path
        
    return None

def check_dependencies():
    """Check that all required dependencies are available"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    
    # Check Tesseract
    tesseract_cmd = get_tesseract_command()
    if not tesseract_cmd:
        if IS_WINDOWS:
            issues.append("Tesseract OCR not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif IS_MACOS:
            issues.append("Tesseract OCR not found. Install with: brew install tesseract")
        else:
            issues.append("Tesseract OCR not found. Install with: sudo apt install tesseract-ocr")
    else:
        # Set tesseract path for pytesseract
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        except ImportError:
            pass
    
    # Check required Python packages
    required_packages = [
        ('pytesseract', 'pytesseract'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('watchdog', 'watchdog')
    ]
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            issues.append(f"Missing package: {package_name}. Install with: pip install {package_name}")
    
    return issues

def safe_file_move(src_path: Path, dst_path: Path) -> bool:
    """Safely move file with cross-platform error handling"""
    try:
        src_path.rename(dst_path)
        return True
    except (PermissionError, OSError) as e:
        # Windows file locking or other OS issues - try copy + delete
        try:
            import shutil
            shutil.copy2(src_path, dst_path)
            src_path.unlink()
            return True
        except Exception:
            print(f"⚠️  Could not move {src_path.name}: {e}")
            return False
    except Exception as e:
        print(f"⚠️  Could not move {src_path.name}: {e}")
        return False

def setup_project_structure():
    """Create necessary project directories"""
    current_dir = Path.cwd()
    
    # Create project directories
    directories = [
        'screenshots',
        'processed_screenshots', 
        'output',
        'debug'
    ]
    
    created = []
    for directory in directories:
        dir_path = current_dir / directory
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            created.append(directory)
    
    if created:
        print(f"📁 Created directories: {', '.join(created)}")
    
    return current_dir

def find_project_screenshots(project_dir: Path) -> List[Path]:
    """Find all screenshot images in project directory (safe search)"""
    screenshot_dirs = [
        project_dir / 'screenshots',
        project_dir,  # Current directory
    ]
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
    found_images = []
    
    for search_dir in screenshot_dirs:
        if search_dir.exists() and search_dir.is_dir():
            for ext in image_extensions:
                # Only search in immediate directory, not recursively
                matches = list(search_dir.glob(ext))
                found_images.extend(matches)
    
    # Remove duplicates and sort
    unique_images = sorted(list(set(found_images)))
    
    # Filter out debug and processed images
    filtered_images = []
    for img in unique_images:
        # Skip debug images, processed images, and any images in debug folder
        if not any(keyword in img.name.lower() for keyword in ['debug', 'processed', '_roi']):
            # Also skip if the image is in a debug folder
            if 'debug' not in str(img.parent).lower():
                filtered_images.append(img)
    
    return filtered_images

def display_banner():
    """Display application banner"""
    print("\n" + "="*70)
    print("    🚀 ZEPP LIFE HEALTH METRICS EXTRACTOR v2.0")
    print("    Enhanced OCR with auto-visualization and cross-platform support")
    print("    ✨ Now with automatic region detection and visual debugging")
    print("="*70)
    print()

def display_quick_menu():
    """Display main menu"""
    print("🚀 ZEPP LIFE METRICS EXTRACTOR")
    print("1. 🔄 Auto-Process Screenshots (All-in-One Mode)")
    print("2. 📊 View Results")
    print("3. ❌ Exit")
    print("-" * 50)

def get_choice(prompt: str = "Enter your choice (1-3): ") -> str:
    """Get user input safely"""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
        sys.exit(0)

def display_processing_modes():
    """Display processing mode options"""
    print("\n🔄 AUTO-PROCESS MODES:")
    print("1. 📷 Process Existing Screenshots (One-time batch)")
    print("2. 👀 Continuous Monitoring (Watch for new screenshots)")
    print("3. 🔄 Complete Mode (Process existing + continuous monitoring)")
    print("4. ⬅️ Back to main menu")
    print("-" * 50)

def get_deletion_preference():
    """Get user preference for screenshot deletion after processing"""
    print("\n🗑️ SCREENSHOT MANAGEMENT:")
    print("What should happen to screenshots after processing?")
    print("1. 📁 Move to 'processed_screenshots' folder (Recommended)")
    print("2. 🗑️ Delete screenshots completely")
    print("3. 📋 Keep screenshots in original location")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            return "move"
        elif choice == "2":
            print("⚠️ Warning: Screenshots will be permanently deleted after processing!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                return "delete"
            else:
                continue
        elif choice == "3":
            return "keep"
        else:
            print("❌ Invalid choice. Please select 1-3.")

def handle_screenshot_file(img_path: Path, extractor, output_file: Path, deletion_mode: str, project_dir: Path) -> bool:
    """Handle processing and file management for a single screenshot"""
    try:
        metrics = extractor.extract_metrics(str(img_path))
        if metrics:
            # Save to CSV
            extractor.save_to_csv(metrics, str(output_file), append=True)
            
            # Handle file based on deletion preference
            if deletion_mode == "move":
                processed_dir = project_dir / "processed_screenshots"
                processed_dir.mkdir(exist_ok=True)
                processed_path = processed_dir / img_path.name
                return safe_file_move(img_path, processed_path)
            elif deletion_mode == "delete":
                img_path.unlink()
                return True
            elif deletion_mode == "keep":
                return True  # Do nothing, keep file in place
            
        return metrics is not None
        
    except Exception as e:
        print(f"❌ Error processing {img_path.name}: {e}")
        return False

def process_existing_screenshots(project_dir: Path, deletion_mode: str) -> bool:
    """Process all existing screenshots in project directory"""
    try:
        # Find screenshots
        screenshots = find_project_screenshots(project_dir)
        
        if not screenshots:
            print("❌ No existing screenshots found.")
            print("💡 Place your Zepp Life screenshots in the 'screenshots' folder or current directory.")
            return False
        
        print(f"✅ Found {len(screenshots)} existing screenshots:")
        for i, img in enumerate(screenshots[:5], 1):
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"  {i}. {img.name} ({size_mb:.1f} MB)")
        
        if len(screenshots) > 5:
            print(f"  ... and {len(screenshots) - 5} more")
        
        # Confirm processing
        print(f"\n❓ Process all {len(screenshots)} existing screenshots? (Y/n): ", end="")
        confirm = input().strip().lower()
        
        if confirm and confirm[0] == 'n':
            print("❌ Processing cancelled.")
            return False
        
        # Initialize extractor
        print("\n⚙️ Initializing OCR engines...")
        from zepp_life_text_extractor import ZeppLifeExtractor
        extractor = ZeppLifeExtractor(debug=False, visualize=True)
        
        output_file = project_dir / "zepp_life_metrics.csv"
        
        # Process images
        successful = 0
        failed = 0
        results = []
        
        print(f"\n🔄 Processing {len(screenshots)} existing screenshots...")
        print("=" * 70)
        
        for i, img_path in enumerate(screenshots, 1):
            print(f"[{i:3}/{len(screenshots)}] {img_path.name[:45]:45} ", end="")
            
            if handle_screenshot_file(img_path, extractor, output_file, deletion_mode, project_dir):
                # Try to get metrics for display
                try:
                    metrics = extractor.extract_metrics(str(img_path)) if deletion_mode != "delete" else None
                    if metrics:
                        results.append(metrics)
                except:
                    pass
                print("✅")
                successful += 1
            else:
                print("❌ (failed)")
                failed += 1
        
        print("=" * 70)
        print(f"🎉 Existing screenshots processing completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            print(f"📊 Results saved to: {output_file}")
            
            # Show recent results
            if results:
                print("\n📈 Sample extracted data:")
                for i, metrics in enumerate(results[-3:], 1):
                    print(f"  {i}. HR:{metrics.heart_rate} Stress:{metrics.stress} Steps:{metrics.steps} Distance:{metrics.distance}km")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error in processing existing screenshots: {e}")
        return False

class EnhancedScreenshotWatcher(FileSystemEventHandler):
    """Enhanced monitor for new screenshots with deletion management"""
    
    def __init__(self, extractor, output_file: Path, deletion_mode: str, project_dir: Path):
        self.extractor = extractor
        self.output_file = output_file
        self.deletion_mode = deletion_mode
        self.project_dir = project_dir
        self.processed_files = set()
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image file
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            # Wait a moment for file to be fully written
            time.sleep(2)
            
            if file_path in self.processed_files:
                return
                
            self.processed_files.add(file_path)
            self.process_new_screenshot(file_path)
    
    def process_new_screenshot(self, file_path: Path):
        """Process a newly detected screenshot"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] 📷 New screenshot detected: {file_path.name}")
            
            if handle_screenshot_file(file_path, self.extractor, self.output_file, self.deletion_mode, self.project_dir):
                try:
                    # Get metrics for display if file still exists
                    if self.deletion_mode != "delete" and file_path.exists():
                        metrics = self.extractor.extract_metrics(str(file_path))
                        if metrics:
                            print(f"[{timestamp}] ✅ Processed: HR={metrics.heart_rate}, Steps={metrics.steps}, Stress={metrics.stress}")
                    else:
                        print(f"[{timestamp}] ✅ Processed and {'deleted' if self.deletion_mode == 'delete' else 'moved'}")
                except:
                    print(f"[{timestamp}] ✅ Processed successfully")
                
                # File management message
                if self.deletion_mode == "move":
                    print(f"[{timestamp}] 📁 Moved to processed folder")
                elif self.deletion_mode == "delete":
                    print(f"[{timestamp}] 🗑️ Screenshot deleted after processing")
                elif self.deletion_mode == "keep":
                    print(f"[{timestamp}] 📋 Screenshot kept in original location")
            else:
                print(f"[{timestamp}] ❌ Could not extract metrics")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

def start_continuous_monitoring(project_dir: Path, deletion_mode: str) -> bool:
    """Start continuous monitoring for new screenshots"""
    try:
        print("\n👀 CONTINUOUS MONITORING MODE")
        print("This will monitor the 'screenshots' folder for new images and process them automatically.")
        
        screenshots_dir = project_dir / "screenshots"
        output_file = project_dir / "zepp_life_metrics.csv"
        
        print(f"📁 Monitoring: {screenshots_dir}")
        print(f"📊 Output: {output_file}")
        
        # File management info
        if deletion_mode == "move":
            print(f"📁 Screenshots will be moved to: {project_dir / 'processed_screenshots'}")
        elif deletion_mode == "delete":
            print(f"🗑️ Screenshots will be deleted after processing")
        elif deletion_mode == "keep":
            print(f"📋 Screenshots will remain in original location")
        
        print("\n⚙️ Initializing OCR engines...")
        
        from zepp_life_text_extractor import ZeppLifeExtractor
        extractor = ZeppLifeExtractor(debug=False, visualize=True)
        
        # Setup file watcher
        event_handler = EnhancedScreenshotWatcher(extractor, output_file, deletion_mode, project_dir)
        observer = Observer()
        observer.schedule(event_handler, str(screenshots_dir), recursive=False)
        
        observer.start()
        
        print("🚀 Monitoring started! Place screenshots in the 'screenshots' folder.")
        print("Press Ctrl+C to stop monitoring.\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping monitoring...")
            observer.stop()
            
        observer.join()
        print("✅ Monitoring stopped.")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        return False

def auto_process_screenshots():
    """Comprehensive auto-processing with multiple modes and deletion options"""
    try:
        print("\n🔍 Setting up auto-processing environment...")
        project_dir = setup_project_structure()
        
        # Show processing mode options
        while True:
            display_processing_modes()
            mode_choice = input("Enter your choice (1-4): ").strip()
            
            if mode_choice == "4":
                return False  # Back to main menu
            
            if mode_choice not in ["1", "2", "3"]:
                print("❌ Invalid choice. Please select 1-4.")
                continue
                
            # Get deletion preference
            deletion_mode = get_deletion_preference()
            
            if mode_choice == "1":  # Process existing screenshots
                return process_existing_screenshots(project_dir, deletion_mode)
                
            elif mode_choice == "2":  # Continuous monitoring only
                return start_continuous_monitoring(project_dir, deletion_mode)
                
            elif mode_choice == "3":  # Complete mode - both existing and continuous
                print("\n🔄 COMPLETE MODE: Processing existing screenshots first, then starting continuous monitoring...")
                
                # First process existing screenshots
                existing_success = process_existing_screenshots(project_dir, deletion_mode)
                
                if existing_success:
                    print("\n✅ Existing screenshots processed successfully!")
                else:
                    print("\n⚠️ No existing screenshots were processed, but continuing with monitoring...")
                
                # Ask if user wants to continue with monitoring
                print("\n❓ Continue with continuous monitoring? (Y/n): ", end="")
                continue_choice = input().strip().lower()
                
                if continue_choice and continue_choice[0] == 'n':
                    print("✅ Complete processing finished.")
                    return existing_success
                
                # Start continuous monitoring
                monitoring_success = start_continuous_monitoring(project_dir, deletion_mode)
                return existing_success or monitoring_success
            
            break
        
    except Exception as e:
        print(f"❌ Error in auto-processing: {e}")
        return False

def batch_process_existing():
    """Batch process existing images without moving them"""
    try:
        print("\n📁 BATCH PROCESSING MODE")
        project_dir = setup_project_structure()
        
        # Find screenshots
        screenshots = find_project_screenshots(project_dir)
        
        if not screenshots:
            print("❌ No screenshots found.")
            return False
        
        print(f"Found {len(screenshots)} images for processing.")
        
        # Initialize extractor
        from zepp_life_text_extractor import ZeppLifeExtractor
        extractor = ZeppLifeExtractor(debug=False, visualize=True)
        
        successful = 0
        for i, img_path in enumerate(screenshots, 1):
            print(f"Processing {i}/{len(screenshots)}: {img_path.name}")
            
            try:
                metrics = extractor.extract_metrics(str(img_path))
                if metrics:
                    output_file = project_dir / "zepp_life_metrics.csv"
                    extractor.save_to_csv(metrics, str(output_file), append=True)
                    successful += 1
                    print(f"  ✅ Extracted: HR={metrics.heart_rate}, Steps={metrics.steps}")
                else:
                    print("  ❌ No metrics extracted")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\n🎉 Batch processing completed: {successful}/{len(screenshots)} successful")
        return successful > 0
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

class ScreenshotWatcher(FileSystemEventHandler):
    """Monitor for new screenshots and process them automatically"""
    
    def __init__(self, extractor, output_file: Path):
        self.extractor = extractor
        self.output_file = output_file
        self.processed_files = set()
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image file
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            # Wait a moment for file to be fully written
            time.sleep(2)
            
            if file_path in self.processed_files:
                return
                
            self.processed_files.add(file_path)
            self.process_new_screenshot(file_path)
    
    def process_new_screenshot(self, file_path: Path):
        """Process a newly detected screenshot"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] 📷 New screenshot detected: {file_path.name}")
            
            metrics = self.extractor.extract_metrics(str(file_path))
            if metrics:
                self.extractor.save_to_csv(metrics, str(self.output_file), append=True)
                print(f"[{timestamp}] ✅ Processed: HR={metrics.heart_rate}, Steps={metrics.steps}, Stress={metrics.stress}")
                
                # Move to processed folder
                processed_dir = file_path.parent / "processed_screenshots"
                processed_dir.mkdir(exist_ok=True)
                processed_path = processed_dir / file_path.name
                
                if safe_file_move(file_path, processed_path):
                    print(f"[{timestamp}] 📁 Moved to processed folder")
                    
            else:
                print(f"[{timestamp}] ❌ Could not extract metrics")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

def continuous_monitoring():
    """Monitor for new screenshots and process them continuously"""
    try:
        print("\n👀 CONTINUOUS MONITORING MODE")
        print("This will monitor the 'screenshots' folder for new images and process them automatically.")
        
        project_dir = setup_project_structure()
        screenshots_dir = project_dir / "screenshots"
        output_file = project_dir / "zepp_life_metrics.csv"
        
        print(f"📁 Monitoring: {screenshots_dir}")
        print(f"📊 Output: {output_file}")
        print("\n⚙️ Initializing OCR engines...")
        
        from zepp_life_text_extractor import ZeppLifeExtractor
        extractor = ZeppLifeExtractor(debug=False, visualize=False)
        
        # Setup file watcher
        event_handler = ScreenshotWatcher(extractor, output_file)
        observer = Observer()
        observer.schedule(event_handler, str(screenshots_dir), recursive=False)
        
        observer.start()
        
        print("🚀 Monitoring started! Place screenshots in the 'screenshots' folder.")
        print("Press Ctrl+C to stop monitoring.\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            observer.stop()
            
        observer.join()
        print("✅ Monitoring stopped.")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        return False

def view_results():
    """Display recent results from CSV"""
    try:
        project_dir = Path.cwd()
        csv_file = project_dir / "zepp_life_metrics.csv"
        
        if not csv_file.exists():
            print("❌ No results file found. Process some screenshots first.")
            return
        
        print("\n📊 RECENT RESULTS")
        print("=" * 50)
        
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("❌ No data in results file.")
            return
        
        # Show last 10 entries
        recent = df.tail(10)
        
        print(f"📈 Showing last {len(recent)} entries from {len(df)} total:")
        print()
        
        for i, (_, row) in enumerate(recent.iterrows(), 1):
            print(f"{i:2}. {row['Date']} {row['Time']}")
            print(f"    💓 HR: {row['Heart rate']:3} BPM   😰 Stress: {row['Stress']:3}")
            print(f"    👟 Steps: {row['Steps']:6}        🔋 Calories: {row['Calories']:4} kcal")
            print(f"    📏 Distance: {row['Distance']:5.2f} km   🫁 O2: {row['Blood Oxygen']:3}%")
            print()
        
        print(f"📋 Complete data saved in: {csv_file}")
        
    except ImportError:
        print("❌ pandas not installed. Install with: pip install pandas")
    except Exception as e:
        print(f"❌ Error viewing results: {e}")

def main():
    """Enhanced main entry point with quick start options"""
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Check system dependencies
        print("🔍 Checking dependencies...")
        dependency_issues = check_dependencies()
        
        if dependency_issues:
            print("\n❌ DEPENDENCY ISSUES FOUND:")
            for issue in dependency_issues:
                print(f"   • {issue}")
            print("\n💡 Please resolve these issues before continuing.")
            return 1
        
        print("✅ All dependencies available!\n")
        
        # Check required files
        required_files = ['zepp_life_text_extractor.py', 'cv_utils.py']
        missing_files = [f for f in required_files if not (current_dir / f).exists()]
        
        if missing_files:
            print("❌ Error: Missing required files!")
            print(f"Missing: {', '.join(missing_files)}")
            return 1
        
        # Setup project structure
        setup_project_structure()
        
        display_banner()
        
        while True:
            display_quick_menu()
            choice = get_choice()
            
            if choice == "1":
                auto_process_screenshots()
                
            elif choice == "2":
                view_results()
                
            elif choice == "3":
                print("\n👋 Thank you for using Zepp Life Health Metrics Extractor!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-3.")
            
            # Pause before showing menu again
            if choice in ["1", "2"]:
                input("\nPress Enter to continue...")
        
        return 0
        
    except ImportError as e:
        print("❌ Error: Missing dependencies!")
        print(f"Import error: {e}")
        print("\n💡 Try installing dependencies:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
