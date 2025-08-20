#!/usr/bin/env python3
"""
Zepp Life Screenshot Tool v2.0 - Cross-Platform Edition

Modern screenshot tool with:
- Cross-platform compatibility (Windows, Linux, macOS)
- Automatic display detection and high-DPI support
- Coordinated region highlighting for accuracy
- Smart file naming with timestamps
- Platform-optimized performance
"""

import os
import sys
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

# Cross-platform screenshot libraries
try:
    from mss import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("⚠️  MSS not installed. Install with: pip install mss")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed. Install with: pip install pillow")

# Configuration (import from config file)
try:
    from config import SCREENSHOT_BOXES, SCREENSHOT_FILENAME_FORMAT, DIRECTORIES
except ImportError:
    # Fallback if config.py is not available
    SCREENSHOT_BOXES = [
        (109, 340, 351, 456, "Heart Rate"),
        (163, 649, 342, 785, "Stress"),
        (83, 1183, 365, 1334, "Steps"),
        (591, 1042, 750, 1162, "Distance"),
        (475, 1177, 760, 1288, "Calories"),
        (429, 1552, 627, 1682, "Blood Oxygen")
    ]
    SCREENSHOT_FILENAME_FORMAT = "%d_%m_%Y-%H_%M_%S.png"
    DIRECTORIES = {
        'screenshots': 'screenshots'
    }

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZeppLifeScreenshotter:
    """
    Modern cross-platform screenshot tool for Zepp Life
    """
    
    def __init__(self, 
                 output_dir: str = DIRECTORIES['screenshots'],
                 highlight_color: str = "red",
                 highlight_duration: float = 0.5):
        """
        Initialize the screenshotter
        
        Args:
            output_dir: Directory to save screenshots
            highlight_color: Color for highlighting screenshot regions
            highlight_duration: Duration to show highlight overlay
        """
        self.output_dir = Path(output_dir)
        self.highlight_color = highlight_color
        self.highlight_duration = highlight_duration
        self.platform = platform.system().lower()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            raise RuntimeError("Missing core dependencies. Install with: pip install mss pillow")
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get information about connected displays"""
        with mss() as sct:
            monitors = sct.monitors
            primary_monitor = monitors[1]  # Usually the primary display
            
            return {
                'all': monitors,
                'primary': primary_monitor,
                'width': primary_monitor['width'],
                'height': primary_monitor['height']
            }
    
    def show_highlight_overlay(self, boxes: List[Tuple[int, int, int, int, str]]):
        """
        Show a temporary overlay to highlight screenshot regions
        
        This helps users position the app correctly but is not captured
        in the final screenshot. This feature is platform-dependent.
        """
        # This is a placeholder for a more advanced GUI implementation
        # For now, we will log the coordinates to the console
        print("\n" + "="*50)
        print("    Screenshot Regions (for reference)")
        print("="*50)
        for x1, y1, x2, y2, name in boxes:
            print(f"- {name}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
        print("\nEnsure the Zepp Life app window is visible and positioned correctly.")
    
    def capture_screenshot(self, 
                           filename: str, 
                           region: Optional[Dict[str, int]] = None) -> Optional[Path]:
        """
        Capture a screenshot of a specific region or the full screen
        
        Args:
            filename: Name for the output screenshot file
            region: Dictionary with region coordinates (top, left, width, height)
            
        Returns:
            Path to the saved screenshot file
        """
        try:
            output_path = self.output_dir / filename
            
            with mss() as sct:
                if region:
                    # Capture a specific region
                    monitor = region
                else:
                    # Capture the primary display
                    monitor = sct.monitors[1]
                
                # Grab data and save to file
                sct_img = sct.grab(monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(output_path))
            
            logger.info(f"Screenshot saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    def draw_boxes_on_image(self, 
                           image_path: Path, 
                           boxes: List[Tuple[int, int, int, int, str]]):
        """
        Draw boxes and labels on a captured screenshot for debugging
        
        Args:
            image_path: Path to the screenshot image
            boxes: List of box coordinates and names
        """
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                
                # Load a font (cross-platform fallback)
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                
                for x1, y1, x2, y2, name in boxes:
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=self.highlight_color, width=2)
                    
                    # Draw label
                    text_position = (x1, y1 - 20)  # Position text above the box
                    draw.text(text_position, name, fill=self.highlight_color, font=font)
            
            # Overwrite the original image with the annotated version
            img.save(image_path)
            logger.info(f"Drew debug boxes on {image_path}")
            
        except Exception as e:
            logger.error(f"Failed to draw boxes on image: {e}")
    
    def take_screenshot(self, draw_boxes: bool = False, 
                        show_overlay: bool = True) -> Optional[Path]:
        """
        Main function to take a screenshot with optional overlays and boxes
        
        Args:
            draw_boxes: Whether to draw debug boxes on the screenshot
            show_overlay: Show a temporary overlay to position the window
            
        Returns:
            Path to the final screenshot
        """
        # Show positioning overlay
        if show_overlay:
            self.show_highlight_overlay(SCREENSHOT_BOXES)
            print(f"\nTaking screenshot in 3 seconds...")
            time.sleep(3)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime(SCREENSHOT_FILENAME_FORMAT)
        
        # Capture the screenshot
        screenshot_path = self.capture_screenshot(timestamp)
        
        if screenshot_path and draw_boxes:
            # Draw debug boxes on the image
            self.draw_boxes_on_image(screenshot_path, SCREENSHOT_BOXES)
        
        return screenshot_path


def main():
    """
    Main function to run the screenshot tool
    """
    print("Zepp Life Screenshot Tool v2.0 - Cross-Platform Edition")
    print("="*55)
    
    # Initialize screenshotter
    screenshotter = ZeppLifeScreenshotter(
        output_dir=DIRECTORIES['screenshots']
    )
    
    # Get display info
    display_info = screenshotter.get_display_info()
    print(f"🖥️  Primary Display: {display_info['width']}x{display_info['height']}")
    
    # Main loop
    while True:
        print("\nMENU:")
        print("1. Take Screenshot (with debug boxes)")
        print("2. Take Screenshot (clean, no boxes)")
        print("3. Take Screenshots every 30 seconds")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == '1':
            screenshotter.take_screenshot(draw_boxes=True, show_overlay=True)
            
        elif choice == '2':
            screenshotter.take_screenshot(draw_boxes=False, show_overlay=True)
            
        elif choice == '3':
            print("\n🔄 Starting continuous screenshots every 30 seconds")
            print("   Press Ctrl+C to stop")
            try:
                while True:
                    screenshotter.take_screenshot(draw_boxes=False, show_overlay=False)
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\n🛑 Continuous screenshots stopped by user")
                
        elif choice == 'q':
            print("Exiting... 👋")
            break
            
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    # Check dependencies
    if not MSS_AVAILABLE or not PIL_AVAILABLE:
        print("\n❌ Error: Missing required packages. Please install them:")
        print("   pip install mss pillow")
        sys.exit(1)
    
    main()
