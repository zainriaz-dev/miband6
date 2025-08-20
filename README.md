# Zepp Life Health Metrics Extractor

A cross-platform Python tool that automatically extracts health metrics (heart rate, steps, stress, etc.) from Zepp Life app screenshots using advanced OCR. Works on Windows, macOS, and Linux with a simple point-and-click interface.

## 🚀 Quick Install

### Windows
```bash
# 1. Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# 2. Install Python packages
pip install -r requirements.txt
```

### macOS
```bash
# 1. Install Tesseract OCR
brew install tesseract
# 2. Install Python packages
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)
```bash
# 1. Install Tesseract OCR
sudo apt install tesseract-ocr tesseract-ocr-eng
# 2. Install Python packages
pip install -r requirements.txt
```

## 💻 Quick Usage

1. **Take screenshots** of your Zepp Life health metrics
2. **Place them** in the `screenshots/` folder (or current directory)
3. **Run the tool:**
   ```bash
   python run.py
   ```
4. **Choose option 1** (Auto-Process All Screenshots)
5. **Done!** Results saved to `zepp_life_metrics.csv`

## ✨ Features

- **🔄 Auto-Processing** - Finds and processes all screenshots automatically
- **📊 Multi-Metric Extraction** - Heart rate, stress, steps, distance, calories, blood oxygen
- **👀 Real-time Monitoring** - Watches for new screenshots and processes them instantly
- **🌐 Cross-Platform** - Works seamlessly on Windows, macOS, and Linux
- **🎯 Smart OCR** - Multiple OCR engines with error correction for high accuracy
- **📁 Batch Processing** - Handle hundreds of screenshots in one go
- **🛡️ Safe File Handling** - Robust error handling and automatic backup

## 🔧 Troubleshooting

### "Tesseract not found"
- **Windows:** Download and install from the link above, or set `TESSERACT_CMD` environment variable
- **macOS:** Run `brew install tesseract` 
- **Linux:** Run `sudo apt install tesseract-ocr`

### "No screenshots found"
- Place your Zepp Life screenshots in the `screenshots/` folder
- Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP

### OpenCV/PIL import errors
- **Windows:** Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- **macOS:** Run `brew install libjpeg libpng`
- **Linux:** Run `sudo apt install python3-dev libgl1-mesa-glx`

### File permission errors
- Run terminal/command prompt as administrator (Windows) or use `sudo` (Linux/macOS)
- Check that the folder is not read-only

### Poor OCR accuracy
- Ensure screenshots are clear and high-resolution
- Try taking screenshots in good lighting conditions
- Use PNG format for best quality

## 📋 Output

Results are saved to `zepp_life_metrics.csv` with columns:
- Date, Time, Heart Rate (BPM), Stress Level, Steps, Distance (km), Calories (kcal), Blood Oxygen (%)

## 🆔 License & Credits

MIT License. Created for personal health data extraction and analysis. 
Uses Tesseract OCR, OpenCV, and other open-source libraries.

---

**Note:** This tool is for personal use with your own health data. Ensure you comply with app terms of service.
