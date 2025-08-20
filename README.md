# Zepp Life Health Data Extractor v2.0 - Cross-Platform Edition

A modern, cross-platform Python automation tool for extracting health metrics from the Zepp Life app using enhanced OCR technology. Works seamlessly on Windows, Linux, and macOS with improved accuracy and platform-specific optimizations.

## 🌟 Key Features

✅ **Cross-Platform Compatibility** - Works on Windows, Linux, macOS  
✅ **Enhanced OCR Accuracy** - Fixed issues: Heart rate (997→97), Steps (2 15→215)  
✅ **Multi-Engine OCR Support** - Tesseract, EasyOCR, PaddleOCR  
✅ **Advanced Image Processing** - CLAHE, denoising, morphological operations  
✅ **Smart Error Correction** - Automatic OCR mistake fixes  
✅ **Real-time Monitoring** - Continuous screenshot processing  
✅ **Platform-Optimized** - Automatic OS detection and configuration  

## 📁 Project Structure

```
zepp-life-extractor/
├── zepp_life_screenshot.py     # Cross-platform screenshot capture
├── zepp_life_text_extractor.py # Enhanced OCR text extraction
├── config.py                   # Configuration settings
├── requirements.txt            # Essential dependencies
├── zepp_life_data.csv         # Extracted health data
├── screenshots/                # Raw screenshots
├── processed_screenshots/      # Processed screenshots
└── README.md                  # This documentation
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR**:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt install tesseract-ocr`
   - **macOS**: `brew install tesseract`

3. **Take Screenshots**:
   ```bash
   python zepp_life_screenshot.py
   ```

4. **Extract Health Data**:
   ```bash
   python zepp_life_text_extractor.py
   ```

## 🔧 Core Components

### 1. Screenshot Tool (`zepp_life_screenshot.py`)
- **Cross-platform screenshot capture** using MSS library
- **Coordinate-based region highlighting** for accuracy
- **Platform-specific optimizations** (Windows/Linux/macOS)
- **Visual debugging** with box overlays
- **Smart file naming** with timestamps

### 2. Text Extractor (`zepp_life_text_extractor.py`)
- **Enhanced cross-platform OCR** with multi-engine support
- **Advanced image preprocessing** for better accuracy
- **Smart numeric parsing** with error correction
- **Health metrics extraction**: Heart Rate, Stress, Steps, Distance, Calories, Blood Oxygen
- **CSV export** with comprehensive data tracking
- **Real-time monitoring** with automatic processing

### 3. Configuration (`config.py`)
- **Centralized settings** for coordinates and OCR parameters
- **Platform-specific configurations** 
- **Health metric validation ranges**
- **OCR error correction mappings**

## 📋 Requirements

### Essential Dependencies
```bash
pip install -r requirements.txt
```

- **pytesseract** - Primary OCR engine
- **Pillow** - Image processing
- **opencv-python** - Advanced image processing
- **numpy** - Numerical operations
- **mss** - Cross-platform screenshot capture

### Platform-Specific OCR Installation
- **Windows**: Download [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Optional Enhanced OCR (for improved accuracy)
- **easyocr**: `pip install easyocr`
- **paddleocr**: `pip install paddleocr`

## 📊 Project Status

✅ **Fully Functional** - Successfully tested and operational  
✅ **Cross-Platform** - Works on Windows, Linux, macOS  
✅ **Enhanced Accuracy** - Fixed OCR parsing issues  
✅ **Real-time Processing** - Continuous monitoring capabilities  

### Sample Data Output
```csv
Date,Time,Heart rate,Stress,Steps,Distance,Calories,Blood Oxygen
2025-07-27,20:36:41,99,62,215,0.13,7,99
2025-07-27,20:36:54,97,62,215,0.13,7,99
```

## Quick Start

1. **Prerequisites**: Ensure you have Python 3.6+, ADB, and Tesseract OCR installed
2. **Connect Device**: Enable USB debugging on your Android device and connect via USB
3. **Install Dependencies**: `pip install pytesseract pillow opencv-python`
4. **Run Screenshot Tool**: `python zepp_life_screenshot.py`
5. **Extract Data**: `python zepp_life_text_extractor.py`
6. **View Results**: Open `zepp_life_data.csv` to see extracted health metrics

For detailed setup instructions, see the [Installation](#installation) section below.

## Features

### Screenshot Tool (`zepp_life_screenshot.py`)
- **Coordinate-based Capture**: Takes screenshots with precise coordinate marking
- **Visual Box Overlay**: Shows exactly which areas will be extracted
- **Black Masking**: Blacks out all areas except the 5 health metric boxes
- **Box Numbering**: Labels each extraction area (box1-box5) for identification
- **Timestamp Naming**: Saves screenshots with date/time stamps

### Text Extractor (`zepp_life_text_extractor.py`)
- **Coordinate-based OCR**: Extracts text from 6 specific coordinate regions
- **Health Metrics**: Captures Heart Rate, Stress, Steps, Distance, Calories, and Blood Oxygen
- **CSV Export**: Saves data to timestamped CSV format
- **Batch Processing**: Processes multiple screenshots automatically
- **Data Validation**: Cleans and validates extracted numeric values
- **File Management**: Moves processed screenshots to separate folder

### Cross-platform Support
- **Ubuntu Compatibility**: Optimized for Ubuntu 22.04 LTS systems
- **Bash Integration**: Uses Linux-native command execution

## Requirements

### System Requirements
- Ubuntu 22.04 LTS or later
- Python 3.8 or higher
- Android device with USB debugging enabled
- USB cable for device connection

### Software Dependencies
- **ADB (Android Debug Bridge)** - For device communication
- **Python 3.6+** with standard libraries
- **Tesseract OCR** - For text extraction from images
- **Python packages**:
  - `pytesseract` - OCR wrapper
  - `Pillow (PIL)` - Image processing
  - `opencv-python` - Advanced image processing (optional)

### Android App Requirements
- Zepp Life app installed on your Android device
- Package name: `com.xiaomi.hm.health`
- USB debugging enabled in Developer Options

## Installation

### 1. Update System and Install Python

```bash
# Update package list
sudo apt update

# Install Python 3 and pip (usually pre-installed on Ubuntu 22.04)
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### 2. Install ADB (Android Debug Bridge)

#### Option A: Install via APT (Recommended)
```bash
# Install ADB from Ubuntu repositories
sudo apt install adb

# Verify installation
adb version
```

#### Option B: Install Android SDK Platform Tools
```bash
# Download and extract platform tools
wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip
unzip platform-tools-latest-linux.zip

# Move to /opt and add to PATH
sudo mv platform-tools /opt/
echo 'export PATH="/opt/platform-tools:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
adb version
```

### 3. Install Tesseract OCR (for text extraction)

```bash
# Install Tesseract OCR and language packs
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install additional language support (optional)
sudo apt install tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# Verify installation
tesseract --version
```

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install pytesseract pillow opencv-python

# Or install system-wide
pip3 install pytesseract pillow opencv-python
```

### 5. Enable USB Debugging on Android Device

1. Go to **Settings** > **About phone**
2. Tap **Build number** 7 times to enable Developer Options
3. Go to **Settings** > **Developer options**
4. Enable **USB debugging**
5. Connect your device via USB
6. Accept the USB debugging prompt on your device

### 6. Setup USB Permissions and Verify Device Connection

```bash
# Add your user to the plugdev group for USB access
sudo usermod -a -G plugdev $USER

# Create udev rules for Android devices (optional but recommended)
sudo tee /etc/udev/rules.d/51-android.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="04e8", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="22b8", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="0bb4", MODE="0666", GROUP="plugdev"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Log out and log back in for group changes to take effect
# Or restart your system

# Check if device is detected
adb devices

# You should see your device listed
# Example output:
# List of devices attached
# ABC123DEF456    device
```

### 7. Install Zepp Life App

Ensure the Zepp Life app is installed on your Android device from the Google Play Store.

## Usage

### Screenshot Automation

```bash
# Navigate to the project directory
cd /path/to/miband6finalfinal/

# Activate virtual environment if using one
source venv/bin/activate

# Run the screenshot automation script
python3 zepp_life_screenshot.py
```

### Text Extraction and Data Processing

```bash
# Extract text from all screenshots and save to CSV
python3 zepp_life_text_extractor.py

# This will:
# 1. Process all PNG files in the screenshots/ directory
# 2. Extract health metrics using OCR
# 3. Save data to zepp_life_data.csv
# 4. Optionally delete processed screenshots
```

### Complete Workflow

```bash
# Step 1: Take screenshots with coordinate markers
python3 zepp_life_screenshot.py

# Step 2: Extract health metrics from coordinates
python3 zepp_life_text_extractor.py

# Step 3: View extracted data
# Open zepp_life_data.csv in LibreOffice Calc or any spreadsheet application
libreoffice --calc zepp_life_data.csv
```

### How It Works

1. **Screenshot Tool**:
   - Connects to Android device via ADB
   - Takes screenshot of Zepp Life app
   - Overlays black mask except for 5 coordinate boxes
   - Labels each box (box1-box5) for identification
   - Saves screenshot with timestamp

2. **Text Extractor**:
   - Processes screenshots from the screenshots folder
   - Extracts text from 6 predefined coordinate regions:
     - Box1: Heart Rate
     - Box2: Stress
     - Box3: Steps
     - Box4: Distance
     - Box5: Calories
     - Box6: Blood Oxygen
   - Saves extracted data to CSV
   - Moves processed screenshots to processed_screenshots folder

### Stopping the Script

To stop the automation:
- Press `Ctrl+C` in the terminal
- The script will gracefully stop and show final statistics

## Output

### Screenshot Files

Screenshots are saved in the `screenshots/` directory with the following naming convention:

```
DD_MM_YYYY-HH_MM_SS.png
```

Example: `25_12_2024-14_30_45.png`

### Directory Structure

```
miband6finalfinal/
├── zepp_life_screenshot.py      # Screenshot tool with coordinate markers
├── zepp_life_text_extractor.py  # Coordinate-based text extraction
├── zepp_life_data.csv           # Extracted health metrics
├── screenshots/                 # Raw screenshots (empty after processing)
├── processed_screenshots/       # Processed screenshots with data
├── venv/                        # Python virtual environment (if used)
└── README.md                    # This documentation
```

### Data Output

#### CSV Data Format
The extracted data is saved to `zepp_life_data.csv` with the following columns:

- **Date**: Date of screenshot (YYYY-MM-DD)
- **Time**: Time of screenshot (HH:MM:SS)
- **Heart rate**: Heart rate in BPM
- **Stress**: Stress level (0-100)
- **Steps**: Step count
- **Distance**: Distance in kilometers
- **Calories**: Calories burned in kcal
- **Blood Oxygen**: Blood oxygen saturation percentage

#### Sample Data
```csv
Date,Time,Heart rate,Stress,Steps,Distance,Calories,Blood Oxygen
2025-07-27,20:36:41,99,62,215,0.13,7,99
2025-07-27,20:36:54,97,62,215,0.13,7,99
2025-07-27,20:37:08,97,62,215,0.13,7,99
```

### Log Output

The script provides detailed logging information:

```
2024-12-25 14:30:00 - INFO - Zepp Life Screenshot tool initialized
2024-12-25 14:30:01 - INFO - Connected to device: ABC123DEF456
2024-12-25 14:30:02 - INFO - Launching Zepp Life app...
2024-12-25 14:30:05 - INFO - Starting loop iteration 1
2024-12-25 14:30:05 - INFO - Scrolling up from (540, 1400) to (540, 1000)
2024-12-25 14:30:10 - INFO - Taking screenshot: 25_12_2024-14_30_10.png
```

## Automation Process

### Loop Cycle

The automation follows this continuous cycle:

1. **App Check**: Verifies app is in foreground (auto-launches if closed)
2. **Scroll Down**: Performs a slight downward scroll gesture
3. **Wait**: 5-second pause
4. **App Verification**: Ensures app is still open before screenshot
5. **Screenshot**: Captures and saves screenshot
6. **Wait**: 5-second pause before next iteration
7. **Repeat**: Returns to step 1

### Smart App Recovery

The tool includes intelligent app recovery features:

- **Automatic Detection**: Continuously monitors if the Zepp Life app is in foreground
- **Instant Recovery**: If you close the app during execution, it automatically reopens
- **Dual Verification**: Uses both resumed activity and focus window checks for accuracy
- **No Manual Intervention**: You can close the app anytime - the script handles everything
- **Robust Launching**: Multiple launch methods ensure reliable app opening

### Performance Statistics

Every 10 iterations, the script displays performance statistics:

```
=== Statistics ===
Runtime: 0:05:30
Screenshots taken: 50
Scrolls performed: 50
App launches: 1
Errors encountered: 0
==================
```

## Troubleshooting

### Common Issues

#### Device Not Detected

```bash
# Check USB connection
adb devices

# If no devices shown:
# 1. Check USB cable
# 2. Enable USB debugging
# 3. Accept debugging prompt on device
# 4. Try different USB port
# 5. Check USB permissions (see installation section)
# 6. Restart ADB server
adb kill-server
adb start-server
```

#### ADB Not Found

```bash
# Check if ADB is in PATH
adb version

# If command not found:
# 1. Reinstall via apt: sudo apt install adb
# 2. Or add platform-tools to PATH in ~/.bashrc
# 3. Reload shell: source ~/.bashrc
```

#### Tesseract OCR Issues

```bash
# Check Tesseract installation
tesseract --version

# If command not found:
sudo apt install tesseract-ocr tesseract-ocr-eng

# Check available languages
tesseract --list-langs

# If OCR accuracy is poor:
# 1. Ensure good screenshot quality
# 2. Check device screen brightness
# 3. Try different OCR languages: --oem 3 --psm 6 -l eng+chi_sim
```

#### Text Extraction Issues

- **Poor OCR Accuracy**: Ensure screenshots are clear and high contrast
- **Missing Health Data**: Check if the Zepp Life app is displaying the correct screen
- **CSV Export Errors**: Verify write permissions in the project directory
- **Language Recognition**: Install additional Tesseract language packs if needed

#### Python Not Found

```bash
# Check Python installation
python3 --version

# If command not found:
sudo apt install python3 python3-pip

# Create symlink if needed
sudo ln -s /usr/bin/python3 /usr/bin/python
```

#### App Not Installed

- Install Zepp Life from Google Play Store
- Verify package name: `com.xiaomi.hm.health`
- Check app permissions

#### Permission Denied

```bash
# Check device authorization
adb devices

# If "unauthorized":
# 1. Revoke USB debugging authorizations on device
# 2. Disconnect and reconnect USB
# 3. Accept new authorization prompt

# If USB permission issues:
# 1. Check if user is in plugdev group
groups $USER

# 2. Add user to plugdev group if not present
sudo usermod -a -G plugdev $USER

# 3. Log out and log back in
```

#### Screenshots Not Saving

- Check write permissions in project directory
- Ensure `screenshots/` directory exists
- Verify sufficient disk space
- Check file system permissions:
```bash
ls -la screenshots/
chmod 755 screenshots/
```

### Ubuntu-Specific Issues

#### USB Device Access
- Check USB device permissions and udev rules
- Try different USB ports (USB 2.0 vs 3.0)
- Verify device is recognized:
```bash
lsusb
dmesg | tail
```

#### Package Dependencies
- Install missing system packages:
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```
- Update package cache if needed:
```bash
sudo apt update
```

#### Python Virtual Environment
```bash
# If virtual environment issues occur
sudo apt install python3-venv python3-dev
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### Error Recovery

The script includes automatic error recovery:

- **Connection Issues**: Attempts to reconnect to device
- **App Crashes**: Continues with next iteration
- **Screenshot Failures**: Logs error and continues
- **Scroll Failures**: Logs warning and continues

## Performance Optimization

### System Performance

- **CPU Usage**: Minimal during wait periods
- **Memory Usage**: Low memory footprint
- **Storage**: ~2-5MB per screenshot
- **Network**: No network usage required

### Recommendations

- Use USB 3.0 ports for faster data transfer
- Ensure device has sufficient battery
- Close unnecessary apps on Android device
- Monitor available storage space
- Use SSD storage for better performance

## Data Analysis and Insights

### Health Metrics Tracking

The extracted CSV data enables comprehensive health tracking:

- **Daily Activity Monitoring**: Track steps, calories, and distance over time
- **Heart Rate Analysis**: Monitor resting and active heart rate patterns
- **Sleep Pattern Analysis**: Analyze sleep duration and quality trends
- **Activity Correlation**: Compare different health metrics for insights

### Data Visualization

Recommended tools for analyzing the CSV data:

- **Microsoft Excel**: Basic charts and pivot tables
- **Google Sheets**: Online collaboration and basic analysis
- **Python (pandas/matplotlib)**: Advanced statistical analysis
- **R**: Statistical computing and graphics
- **Tableau**: Professional data visualization

### Sample Analysis Queries

```python
# Example Python analysis using pandas
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('zepp_life_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Daily steps trend
df.groupby(df['date'].dt.date)['steps'].max().plot(kind='line')
plt.title('Daily Steps Trend')
plt.show()

# Average heart rate by hour
df['hour'] = df['date'].dt.hour
df.groupby('hour')['heart_rate'].mean().plot(kind='bar')
plt.title('Average Heart Rate by Hour')
plt.show()
```

## Environment Variables

Optional environment variables for customization:

```bash
# Set custom output directory
export SCREENSHOT_DIR=/path/to/custom/screenshots

# Set custom ADB path
export ADB_PATH=/custom/path/to/adb

# Set Tesseract data path if needed
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
```

For permanent environment variables:
```bash
# Add to ~/.bashrc for permanent settings
echo 'export SCREENSHOT_DIR=/path/to/custom/screenshots' >> ~/.bashrc
echo 'export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/' >> ~/.bashrc
source ~/.bashrc
```

## Support

### Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review log output for error messages
3. Verify all requirements are met
4. Test ADB connection manually

### System Information

When reporting issues, include:

- Ubuntu version: `lsb_release -a`
- Python version: `python --version`
- ADB version: `adb version`
- Device model and Android version
- Error messages from log output

### Running with Elevated Privileges

If you encounter permission issues:

1. Open terminal with elevated privileges:
```bash
sudo -i
```
2. Navigate to project directory
3. Run the script with appropriate permissions
4. Run the script

## Project Goals and Use Cases

### Primary Objectives

1. **Personal Health Monitoring**: Continuous tracking of fitness metrics
2. **Data Ownership**: Export and own your health data outside of proprietary apps
3. **Long-term Analysis**: Build historical datasets for trend analysis
4. **Research Applications**: Support personal health research and optimization

### Potential Applications

- **Fitness Goal Tracking**: Monitor progress towards daily/weekly targets
- **Health Pattern Recognition**: Identify correlations between activities and health metrics
- **Medical Documentation**: Provide health data to healthcare providers
- **Wellness Programs**: Track corporate or personal wellness initiatives
- **Research Studies**: Contribute to personal health research projects

### Data Privacy and Security

- **Local Processing**: All data remains on your local machine
- **No Cloud Dependencies**: No data is sent to external servers
- **User Control**: Complete control over data collection and retention
- **Open Source**: Transparent code for security review

---

**Note**: This tool is designed for personal use and automation with Mi Band 6 devices. Ensure you comply with the Zepp Life app's terms of service and your organization's policies when using automation tools. Always respect data privacy and use this tool responsibly. This project is specifically optimized for Mi Band 6 health metrics and coordinate mapping.