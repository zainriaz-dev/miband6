#!/usr/bin/env python3
"""
Configuration file for Zepp Life Health Data Extractor

This file contains all configurable parameters for both screenshot capture
and text extraction processes.
"""

# ============================================================================
# COORDINATE CONFIGURATIONS
# ============================================================================

# Screenshot tool coordinates (for drawing visible boxes)
SCREENSHOT_BOXES = [
    (109, 340, 351, 456, "Heart Rate"),   # box1
    (163, 649, 342, 785, "Stress"),      # box2  
    (83, 1183, 365, 1334, "Steps"),      # box3
    (591, 1042, 750, 1162, "Distance"),  # box4
    (475, 1177, 760, 1288, "Calories"),  # box5
    (429, 1552, 627, 1682, "Blood Oxygen") # box6
]

# Text extraction coordinates (should match screenshot boxes for accuracy)
EXTRACTION_REGIONS = {
    'box1': (109, 340, 351, 456),   # Heart Rate
    'box2': (163, 649, 342, 785),   # Stress  
    'box3': (83, 1183, 365, 1334),  # Steps
    'box4': (591, 1042, 750, 1162), # Distance
    'box5': (475, 1177, 760, 1288), # Calories
    'box6': (429, 1552, 627, 1682)  # Blood Oxygen
}

# Improved heart rate coordinates (tighter crop for better accuracy)
IMPROVED_HEART_RATE_COORDS = (115, 350, 345, 440)

# ============================================================================
# OCR CONFIGURATIONS
# ============================================================================

# Default Tesseract OCR configuration
DEFAULT_OCR_CONFIG = '--psm 6'

# Enhanced OCR configurations for different metrics
OCR_CONFIGS = {
    'heart_rate': {
        'standard': '--psm 6',
        'enhanced': '--psm 6 -c tessedit_char_whitelist=0123456789BPM ',
        'single_line': '--psm 8',
        'digits_only': '--psm 6 -c tessedit_char_whitelist=0123456789'
    },
    'stress': '--psm 6 -c tessedit_char_whitelist=0123456789',
    'steps': '--psm 6 -c tessedit_char_whitelist=0123456789',
    'distance': '--psm 6 -c tessedit_char_whitelist=0123456789.',
    'calories': '--psm 6 -c tessedit_char_whitelist=0123456789kcal ',
    'blood_oxygen': '--psm 6 -c tessedit_char_whitelist=0123456789%'
}

# ============================================================================
# VALIDATION RANGES
# ============================================================================

# Valid ranges for health metrics (used for data validation)
METRIC_RANGES = {
    'heart_rate': (40, 200),    # BPM
    'stress': (0, 100),         # Stress level
    'steps': (0, 100000),       # Daily steps
    'distance': (0.0, 100.0),   # km
    'calories': (0, 10000),     # kcal
    'blood_oxygen': (70, 100)   # % SpO2
}

# ============================================================================
# APP CONFIGURATIONS
# ============================================================================

# Android app package name
ZEPP_LIFE_PACKAGE = 'com.xiaomi.hm.health'

# Screenshot automation settings
AUTOMATION_CONFIG = {
    'scroll_wait_time': 5,       # seconds to wait after scroll
    'screenshot_wait_time': 5,   # seconds to wait before screenshot
    'app_launch_timeout': 15,    # seconds to wait for app launch
    'screenshot_timeout': 15     # seconds to wait for screenshot capture
}

# ============================================================================
# FILE PATHS AND NAMING
# ============================================================================

# Directory configurations
DIRECTORIES = {
    'screenshots': 'screenshots',
    'processed_screenshots': 'processed_screenshots',
    'output_csv': 'zepp_life_data.csv'
}

# Screenshot filename format
SCREENSHOT_FILENAME_FORMAT = "%d_%m_%Y-%H_%M_%S.png"

# CSV column headers
CSV_HEADERS = [
    'Date', 'Time', 'Heart rate', 'Stress', 'Steps', 
    'Distance', 'Calories', 'Blood Oxygen'
]

# ============================================================================
# OCR ERROR CORRECTIONS
# ============================================================================

# Common OCR misreads and their corrections
OCR_CORRECTIONS = {
    'O kcal': '0 kcal',
    'o kcal': '0 kcal',
    'O km': '0 km',
    'o km': '0 km',
    'BPIVl': 'BPM',
    'BPlVl': 'BPM',
    'BPIv1': 'BPM',
    'BPl\\1': 'BPM',
    'BPl\\l': 'BPM',
    'BPM1': 'BPM',
    '9g': '99',  # Common misread for 99
    'g7': '97',  # Common misread for 97
    'Q7': '97',  # Another common misread for 97
    '0Q': '09'   # Common misread for 09
}

# ============================================================================
# REGEX PATTERNS FOR HEALTH METRICS
# ============================================================================

HEALTH_PATTERNS = {
    'heart_rate': [
        # Enhanced patterns for heart rate extraction
        r'heart\s*rate\s*just\s*now[^\d]*(?:\d+\s+)?(\d{2,3})(?:\s*[a-z])?\s*[_-]*\s*bpm',
        r'just\s*now[^\d]*(?:\d+\s+)?(\d{2,3})(?:\s*[a-z])?\s*[_-]*\s*bpm',
        r'♥\s*(\d+)\s*bpm',
        r'~(\d+)\s*[_-]+\s*bpm',
        r'(?<!average\s)(?<!\()(?<!\d\s)(\d{2,3})\s*[_-]+\s*bpm\s*relaxed',
        r'(?<!average\s)(?<!\()(?<!\d\s)(\d{2,3})\s*bpm\s*relaxed',
        r'(\d{2,3})\s*BPM',  # Direct BPM match
        r'(\d{2,3})'  # Fallback: any 2-3 digit number
    ],
    'stress': [
        r'stress[^\d]*\d+[^\d]*(\d+)',
        r'jul\s*(\d+)',
        r'(\d+)\s*}\s*mild',
        r'(\d+)\s*(?:mild|moderate|high)',
        r'stress[^\d]*(\d+)'
    ],
    'steps': [
        r'distance[^\d]*[\d.,]+[^\d]*km[^\d]*[^\d]*(\d+)[^\d]*calories',
        r'km[^\d]*[^\d]*(\d+)[^\d]*calories',
        r'(?<!\d\s)(?<!\d)(\d+)(?=\s*calories)',
        r'(?<!\d)(\d+)(?=\s*calories)',
        r'steps[^\d]*(?!\d+\s*mins)(\d+)'
    ],
    'distance': [
        r'distance\s*([\d.,]+)\s*km',
        r'([\d.,]+)\s*km'
    ],
    'calories': [
        r'calories\s*(\d+)\s*kcal',
        r'(\d+)\s*kcal',
        r'o\s*kcal'  # OCR might read "0 kcal" as "O kcal"
    ],
    'blood_oxygen': [
        r'blood\s*oxygen[^\d]*(\d+)\s*%',
        r'spo2[^\d]*(\d+)\s*%',
        r'(\d+)\s*%'
    ]
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'handlers': ['console']
}

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

MONITORING_CONFIG = {
    'check_interval': 30,           # seconds between screenshot checks
    'delete_after_processing': False,  # whether to delete processed screenshots
    'max_retries': 3,              # max retries for failed operations
    'stats_display_interval': 10   # show stats every N iterations
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_metric_box_coords(metric_name: str) -> tuple:
    """Get coordinates for a specific metric"""
    box_mapping = {
        'heart_rate': 'box1',
        'stress': 'box2',
        'steps': 'box3', 
        'distance': 'box4',
        'calories': 'box5',
        'blood_oxygen': 'box6'
    }
    
    box_key = box_mapping.get(metric_name)
    if box_key:
        return EXTRACTION_REGIONS.get(box_key)
    return None

def validate_metric_value(metric_name: str, value: str) -> bool:
    """Validate if a metric value is within acceptable range"""
    if metric_name not in METRIC_RANGES or value == 'N/A':
        return True
    
    try:
        numeric_value = float(value)
        min_val, max_val = METRIC_RANGES[metric_name]
        return min_val <= numeric_value <= max_val
    except (ValueError, TypeError):
        return False

def get_ocr_config(metric_name: str, config_type: str = 'standard') -> str:
    """Get OCR configuration for a specific metric"""
    if metric_name in OCR_CONFIGS:
        if isinstance(OCR_CONFIGS[metric_name], dict):
            return OCR_CONFIGS[metric_name].get(config_type, DEFAULT_OCR_CONFIG)
        else:
            return OCR_CONFIGS[metric_name]
    return DEFAULT_OCR_CONFIG
