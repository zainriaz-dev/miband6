#!/usr/bin/env python3
"""
Zepp Life Health Metrics Extractor - Enhanced Startup Script
Comprehensive launcher with automatic screenshot discovery, batch processing,
and continuous monitoring features.
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
        if not any(keyword in img.name.lower() for keyword in ['debug', 'processed', '_roi']):
            filtered_images.append(img)
    
    return filtered_images

def display_banner():
    """Display application banner"""
    print("\n" + "="*70)
    print("    🚀 ZEPP LIFE HEALTH METRICS EXTRACTOR v2.0")
    print("    Enhanced OCR-based health data extraction with automation")
    print("="*70)
    print()

def display_quick_menu():
    """Display quick start menu"""
    print("🚀 QUICK START OPTIONS:")
    print("1. 🔄 Auto-Process All Screenshots (Recommended)")
    print("2. 📁 Batch Process Existing Images")
    print("3. 👀 Continuous Screenshot Monitoring")
    print("4. 🎛️  Advanced Interactive Menu")
    print("5. 📊 View Results")
    print("6. ❌ Exit")
    print("-" * 50)

def get_choice(prompt: str = "Enter your choice (1-6): ") -> str:
    """Get user input safely"""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
        sys.exit(0)

def auto_process_screenshots():
    """Automatically find and process all screenshots in project"""
    try:
        print("\n🔍 Searching for screenshots in project directory...")
        project_dir = setup_project_structure()
        
        # Find screenshots
        screenshots = find_project_screenshots(project_dir)
        
        if not screenshots:
            print("❌ No screenshots found in project directory.")
            print("💡 Place your Zepp Life screenshots in the 'screenshots' folder or current directory.")
            return False
        
        print(f"✅ Found {len(screenshots)} screenshots:")
        for i, img in enumerate(screenshots[:5], 1):
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"  {i}. {img.name} ({size_mb:.1f} MB)")
        
        if len(screenshots) > 5:
            print(f"  ... and {len(screenshots) - 5} more")
        
        # Confirm processing
        print(f"\n❓ Process all {len(screenshots)} screenshots? (Y/n): ", end="")
        confirm = input().strip().lower()
        
        if confirm and confirm[0] == 'n':
            print("❌ Processing cancelled.")
            return False
        
        # Initialize extractor
        print("\n⚙️ Initializing OCR engines...")
        from zepp_life_text_extractor import ZeppLifeExtractor
        extractor = ZeppLifeExtractor(debug=False, visualize=False)
        
        # Process images
        successful = 0
        failed = 0
        results = []
        
        print(f"\n🔄 Processing {len(screenshots)} screenshots...")
        print("=" * 60)
        
        for i, img_path in enumerate(screenshots, 1):
            print(f"[{i:3}/{len(screenshots)}] {img_path.name[:45]:45} ", end="")
            
            try:
                metrics = extractor.extract_metrics(str(img_path))
                if metrics:
                    # Save to CSV
                    output_file = project_dir / "zepp_life_metrics.csv"
                    extractor.save_to_csv(metrics, str(output_file), append=True)
                    
                    # Move to processed folder
                    processed_dir = project_dir / "processed_screenshots"
                    processed_path = processed_dir / img_path.name
                    
                    try:
                        img_path.rename(processed_path)
                    except Exception:
                        pass  # If move fails, continue anyway
                    
                    results.append(metrics)
                    print("✅")
                    successful += 1
                else:
                    print("❌ (no metrics)")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ (error: {str(e)[:20]})")
                failed += 1
        
        print("=" * 60)
        print(f"🎉 Processing completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            output_file = project_dir / "zepp_life_metrics.csv"
            print(f"📊 Results saved to: {output_file}")
            
            # Show recent results
            if results:
                print("\n📈 Sample extracted data:")
                for i, metrics in enumerate(results[-3:], 1):
                    print(f"  {i}. HR:{metrics.heart_rate} Stress:{metrics.stress} Steps:{metrics.steps} Distance:{metrics.distance}km")
        
        return successful > 0
        
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
        extractor = ZeppLifeExtractor(debug=False, visualize=False)
        
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
                
                try:
                    file_path.rename(processed_path)
                    print(f"[{timestamp}] 📁 Moved to processed folder")
                except Exception:
                    pass
                    
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
                batch_process_existing()
                
            elif choice == "3":
                continuous_monitoring()
                
            elif choice == "4":
                # Launch advanced interactive menu
                try:
                    from main import main as cli_main
                    cli_main()
                except ImportError:
                    print("❌ Advanced menu not available.")
                    
            elif choice == "5":
                view_results()
                
            elif choice == "6":
                print("\n👋 Thank you for using Zepp Life Health Metrics Extractor!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")
            
            # Pause before showing menu again
            if choice in ["1", "2", "3", "5"]:
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
