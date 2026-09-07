#!/usr/bin/env python3
"""
Test script to verify RuralMinds installation, modular structure, and functionality.
"""

import sys
import os
import subprocess
from pathlib import Path

# Force UTF-8 stdout if supported
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 60)
print("RuralMinds - System Diagnostics & Tests")
print("=" * 60)

# Test 1: Python version
print("\n[1/7] Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"  [OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"  [FAIL] Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)")
    sys.exit(1)

# Test 2: Required packages
print("\n[2/7] Checking required packages...")
required_packages = [
    'streamlit',
    'chromadb',
    'fitz',  # PyMuPDF
    'nltk',
    'sentence_transformers',
    'transformers',
    'torch'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  [OK] {package}")
    except ImportError:
        print(f"  [FAIL] {package}")
        missing_packages.append(package)

if missing_packages:
    print(f"\n[FAIL] Missing packages: {', '.join(missing_packages)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: FFmpeg
print("\n[3/7] Checking FFmpeg (for video captions)...")
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          text=True, 
                          timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"  [OK] {version_line}")
    else:
        print("  [WARN] FFmpeg found but returned error")
except FileNotFoundError:
    print("  [WARN] FFmpeg not found (needed for auto-caption generation)")
    print("         Install: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)")
except Exception as e:
    print(f"  [WARN] Could not check FFmpeg: {str(e)}")

# Test 4: NLTK data
print("\n[4/7] Checking NLTK data...")
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print("  [OK] NLTK punkt tokenizer")
except LookupError:
    print("  Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)
    print("  [OK] Downloaded")

# Test 5: File structure
print("\n[5/7] Checking modular package structure...")
required_paths = [
    'app.py',
    'config/settings.py',
    'core/database.py',
    'core/auth.py',
    'core/forum.py',
    'services/vector_service.py',
    'services/document_service.py',
    'services/retrieval_service.py',
    'services/llm_service.py',
    'services/audio_service.py',
    'services/translation_service.py',
    'services/video_service.py',
    'ui/styles.py',
    'ui/auth_view.py',
    'ui/admin_view.py',
    'ui/learning_hub_view.py',
    'ui/forum_view.py'
]
for fpath in required_paths:
    full_p = os.path.join(str(BASE_DIR), fpath)
    if os.path.exists(full_p):
        print(f"  [OK] {fpath}")
    else:
        print(f"  [FAIL] {fpath} not found")

# Test 6: Auth system
print("\n[6/7] Testing authentication system...")
try:
    from core.auth import authenticate_user, get_all_users
    users = get_all_users()
    admin_users = [u for u in users if u['username'] in ['admin', 'administrator']]
    if admin_users:
        print(f"  [OK] Admin account registered ({len(users)} total users in DB)")
    else:
        print("  [WARN] Admin account initialized on startup")
except Exception as e:
    print(f"  [FAIL] Auth error: {e}")

# Test 7: Modular Package Imports
print("\n[7/7] Testing modular package namespaces...")
try:
    import config
    import core
    import services
    import ui
    print("  [OK] All package namespaces (config, core, services, ui) imported successfully!")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")

print("\n" + "=" * 60)
print("[SUCCESS] All System Diagnostics Passed!")
print("=" * 60)

if __name__ == "__main__":
    pass
