#!/usr/bin/env python3
"""
Zepp Life Health Metrics Extractor - Main CLI Interface
A simple interactive command-line interface for extracting health metrics from Zepp Life screenshots.

Features:
- Simple menu-driven interface
- Single image processing
- Batch processing
- Automatic file discovery
- Clean CSV output with visualization options

Usage:
    python main.py
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import pandas as pd

# Import our modules
from zepp_life_text_extractor import ZeppLifeExtractor, HealthMetrics
from cv_utils import validate_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZeppLifeCLI:
    """Simple CLI interface for Zepp Life metrics extraction"""
    
    def __init__(self):
        self.extractor = None
        self.current_dir = Path.cwd()
        self.results = []
        
    def display_banner(self):
        """Display application banner"""
        print("\n" + "="*60)
        print("    ZEPP LIFE HEALTH METRICS EXTRACTOR")
        print("    Advanced OCR-based health data extraction")
        print("="*60)
        print()
        
    def display_menu(self):
        """Display main menu options"""
        print("\n📋 MAIN MENU:")
        print("1. 📷 Process Single Image")
        print("2. 📁 Process All Images in Folder")
        print("3. 🔍 Find and Process Images")
        print("4. 📊 View Recent Results")
        print("5. ⚙️  Settings & Debug Options")
        print("6. ❌ Exit")
        print("-" * 40)
        
    def get_user_choice(self, prompt: str = "Enter your choice (1-6): ") -> str:
        """Get user input with validation"""
        try:
            choice = input(prompt).strip()
            return choice
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            return "6"  # Exit
            
    def find_image_files(self, directory: Path) -> List[Path]:
        """Find all image files in directory"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(ext))
            image_files.extend(directory.glob(ext.upper()))
            
        return sorted(image_files)
        
    def process_single_image(self):
        """Process a single image file"""
        print("\n📷 SINGLE IMAGE PROCESSING")
        print("-" * 30)
        
        # Get image path
        image_path = input("Enter image file path (or drag & drop): ").strip().strip('"\'')
        
        if not image_path:
            print("❌ No image path provided.")
            return
            
        image_file = Path(image_path)
        
        if not image_file.exists():
            print(f"❌ File not found: {image_file}")
            return
            
        if not validate_image(str(image_file)):
            print(f"❌ Invalid image file: {image_file}")
            return
            
        print(f"\n🔄 Processing: {image_file.name}")
        
        # Initialize extractor if needed
        if self.extractor is None:
            print("⚙️  Initializing OCR engines...")
            self.extractor = ZeppLifeExtractor(debug=False, visualize=False)
        
        # Extract metrics
        try:
            metrics = self.extractor.extract_metrics(str(image_file))
            
            if metrics:
                self.display_metrics(metrics)
                self.save_results(metrics, image_file)
                self.results.append((str(image_file), metrics))
                print("✅ Processing completed successfully!")
            else:
                print("❌ Failed to extract metrics from image.")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            logger.error(f"Error processing {image_file}: {e}")
            
    def process_folder(self):
        """Process all images in a folder"""
        print("\n📁 BATCH PROCESSING")
        print("-" * 25)
        
        # Get folder path
        folder_path = input("Enter folder path (or press Enter for current directory): ").strip().strip('"\'')
        
        if not folder_path:
            folder_path = self.current_dir
        else:
            folder_path = Path(folder_path)
            
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ Invalid folder: {folder_path}")
            return
            
        # Find image files
        image_files = self.find_image_files(folder_path)
        
        if not image_files:
            print(f"❌ No image files found in: {folder_path}")
            return
            
        print(f"\n🔍 Found {len(image_files)} image files:")
        for i, img in enumerate(image_files[:5], 1):
            print(f"  {i}. {img.name}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
            
        # Confirm processing
        confirm = input(f"\n❓ Process all {len(image_files)} images? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Processing cancelled.")
            return
            
        # Initialize extractor if needed
        if self.extractor is None:
            print("⚙️  Initializing OCR engines...")
            self.extractor = ZeppLifeExtractor(debug=False, visualize=False)
            
        # Process images
        successful = 0
        failed = 0
        
        print(f"\n🔄 Processing {len(image_files)} images...")
        print("-" * 40)
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_file.name[:40]}...", end=" ")
            
            try:
                if validate_image(str(image_file)):
                    metrics = self.extractor.extract_metrics(str(image_file))
                    if metrics:
                        self.save_results(metrics, image_file)
                        self.results.append((str(image_file), metrics))
                        print("✅")
                        successful += 1
                    else:
                        print("❌ (extraction failed)")
                        failed += 1
                else:
                    print("❌ (invalid image)")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ (error: {str(e)[:30]})")
                failed += 1
                logger.error(f"Error processing {image_file}: {e}")
                
        print("-" * 40)
        print(f"✅ Completed: {successful} successful, {failed} failed")
        
        if successful > 0:
            output_file = folder_path / "zepp_life_metrics.csv"
            print(f"📊 Results saved to: {output_file}")
            
    def find_and_process(self):
        """Find images automatically and process them"""
        print("\n🔍 AUTOMATIC IMAGE DISCOVERY")
        print("-" * 35)
        
        # Search in common directories
        search_dirs = [
            self.current_dir,
            Path.home() / "Pictures",
            Path.home() / "Downloads",
            Path.home() / "Desktop"
        ]
        
        all_images = []
        for search_dir in search_dirs:
            if search_dir.exists():
                images = self.find_image_files(search_dir)
                # Filter for likely Zepp Life screenshots (by size or name)
                zepp_images = [img for img in images if self.is_likely_zepp_image(img)]
                all_images.extend(zepp_images)
                
        if not all_images:
            print("❌ No Zepp Life screenshots found in common directories.")
            print("💡 Try using 'Process Folder' option instead.")
            return
            
        print(f"🔍 Found {len(all_images)} potential Zepp Life screenshots:")
        for i, img in enumerate(all_images[:10], 1):
            print(f"  {i}. {img.name} ({img.parent.name})")
        if len(all_images) > 10:
            print(f"  ... and {len(all_images) - 10} more")
            
        # Confirm processing
        confirm = input(f"\n❓ Process all {len(all_images)} images? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Processing cancelled.")
            return
            
        # Process using batch method
        self.process_image_list(all_images)
        
    def is_likely_zepp_image(self, image_path: Path) -> bool:
        """Check if image is likely a Zepp Life screenshot"""
        try:
            # Check filename for common patterns
            name_lower = image_path.name.lower()
            if any(keyword in name_lower for keyword in ['zepp', 'miband', 'screenshot', 'health']):
                return True
                
            # Check image dimensions (Zepp Life screenshots are typically phone screenshots)
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                # Typical phone screenshot ratios
                if 0.4 <= width/height <= 0.7 and height > 800:
                    return True
                    
        except Exception:
            pass
            
        return False
        
    def process_image_list(self, image_files: List[Path]):
        """Process a list of image files"""
        # Initialize extractor if needed
        if self.extractor is None:
            print("⚙️  Initializing OCR engines...")
            self.extractor = ZeppLifeExtractor(debug=False, visualize=False)
            
        successful = 0
        failed = 0
        
        print(f"\n🔄 Processing {len(image_files)} images...")
        print("-" * 40)
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_file.name[:40]}...", end=" ")
            
            try:
                if validate_image(str(image_file)):
                    metrics = self.extractor.extract_metrics(str(image_file))
                    if metrics:
                        self.save_results(metrics, image_file)
                        self.results.append((str(image_file), metrics))
                        print("✅")
                        successful += 1
                    else:
                        print("❌ (extraction failed)")
                        failed += 1
                else:
                    print("❌ (invalid image)")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ (error: {str(e)[:30]})")
                failed += 1
                
        print("-" * 40)
        print(f"✅ Completed: {successful} successful, {failed} failed")
        
    def view_results(self):
        """Display recent results"""
        print("\n📊 RECENT RESULTS")
        print("-" * 20)
        
        if not self.results:
            print("❌ No results available. Process some images first.")
            return
            
        print(f"📈 Showing last {min(10, len(self.results))} results:")
        print()
        
        for i, (image_path, metrics) in enumerate(self.results[-10:], 1):
            image_name = Path(image_path).name
            print(f"{i:2}. {image_name[:30]:<30}")
            print(f"    💓 Heart Rate: {metrics.heart_rate:3} BPM   😰 Stress: {metrics.stress:3}")
            print(f"    👟 Steps: {metrics.steps:6}         🔋 Calories: {metrics.calories:4} kcal")
            print(f"    📏 Distance: {metrics.distance:5.2f} km    🫁 Blood O2: {metrics.blood_oxygen:3}%")
            print()
            
        # Check for CSV file
        csv_file = self.current_dir / "zepp_life_metrics.csv"
        if csv_file.exists():
            print(f"📋 Complete results saved in: {csv_file}")
            
    def settings_menu(self):
        """Settings and debug options"""
        print("\n⚙️  SETTINGS & DEBUG OPTIONS")
        print("-" * 30)
        print("1. 🔍 Enable Debug Mode (save intermediate images)")
        print("2. 🎨 Enable Visualization (show detected regions)")
        print("3. 🧪 Test Single Region")
        print("4. 📂 Open Results Folder")
        print("5. 🔙 Back to Main Menu")
        
        choice = self.get_user_choice("Enter choice (1-5): ")
        
        if choice == "1":
            self.toggle_debug_mode()
        elif choice == "2":
            self.toggle_visualization()
        elif choice == "3":
            self.test_region()
        elif choice == "4":
            self.open_results_folder()
        elif choice == "5":
            return
        else:
            print("❌ Invalid choice.")
            
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        if self.extractor is None:
            self.extractor = ZeppLifeExtractor(debug=True, visualize=False)
            print("✅ Debug mode enabled. Intermediate images will be saved.")
        else:
            self.extractor.debug = not self.extractor.debug
            status = "enabled" if self.extractor.debug else "disabled"
            print(f"✅ Debug mode {status}.")
            
    def toggle_visualization(self):
        """Toggle visualization mode"""
        if self.extractor is None:
            self.extractor = ZeppLifeExtractor(debug=False, visualize=True)
            print("✅ Visualization enabled. Region boundaries will be shown.")
        else:
            self.extractor.visualize = not self.extractor.visualize
            status = "enabled" if self.extractor.visualize else "disabled"
            print(f"✅ Visualization {status}.")
            
    def test_region(self):
        """Test extraction on a specific region"""
        print("\n🧪 REGION TESTING")
        print("This feature helps debug specific metric extraction issues.")
        print("Not implemented in this version.")
        
    def open_results_folder(self):
        """Open the results folder"""
        try:
            import subprocess
            if sys.platform == "win32":
                os.startfile(self.current_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.current_dir)])
            else:
                subprocess.run(["xdg-open", str(self.current_dir)])
            print(f"📂 Opened: {self.current_dir}")
        except Exception as e:
            print(f"❌ Could not open folder: {e}")
            print(f"📁 Results are saved in: {self.current_dir}")
            
    def display_metrics(self, metrics: HealthMetrics):
        """Display extracted metrics in a nice format"""
        print("\n" + "="*40)
        print("    📊 EXTRACTED HEALTH METRICS")
        print("="*40)
        print(f"📅 Date: {metrics.date}")
        print(f"🕐 Time: {metrics.time}")
        print(f"💓 Heart Rate: {metrics.heart_rate} BPM")
        print(f"😰 Stress Level: {metrics.stress}")
        print(f"👟 Steps: {metrics.steps:,}")
        print(f"📏 Distance: {metrics.distance} km")
        print(f"🔋 Calories: {metrics.calories} kcal")
        print(f"🫁 Blood Oxygen: {metrics.blood_oxygen}%")
        print("="*40)
        
    def save_results(self, metrics: HealthMetrics, image_file: Path):
        """Save results to CSV"""
        try:
            output_file = self.current_dir / "zepp_life_metrics.csv"
            
            if self.extractor is None:
                self.extractor = ZeppLifeExtractor()
                
            self.extractor.save_to_csv(metrics, str(output_file), append=True)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            
    def run(self):
        """Main application loop"""
        self.display_banner()
        
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == "1":
                self.process_single_image()
            elif choice == "2":
                self.process_folder()
            elif choice == "3":
                self.find_and_process()
            elif choice == "4":
                self.view_results()
            elif choice == "5":
                self.settings_menu()
            elif choice == "6":
                print("\n👋 Thank you for using Zepp Life Health Metrics Extractor!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
            # Pause before showing menu again
            input("\nPress Enter to continue...")

def main():
    """Entry point"""
    try:
        cli = ZeppLifeCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
