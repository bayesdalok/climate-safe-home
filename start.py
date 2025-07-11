#!/usr/bin/env python3
"""
Startup script for Climate Safe Home API
This script helps with initial setup and environment checking
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Debug imports
try:
    import cv2
    print(f"[DEBUG] OpenCV imported successfully from: {cv2.__file__}")
except Exception as e:
    print(f"[DEBUG] OpenCV import failed: {str(e)}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"[INFO] Current version: {sys.version}")
        return False
    else:
        print(f"[OK] Python version: {sys.version.split()[0]}")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    package_map = {
    'Flask': 'flask',
    'Flask-Cors': 'flask_cors',
    'numpy': 'numpy',
    'opencv-python': 'cv2',
    'opencv-contrib-python': 'cv2',
    'Pillow': 'PIL',
    'python-dotenv': 'dotenv',
    'requests': 'requests',
    'openai': 'openai',
    'Werkzeug': 'werkzeug',
    'PySocks': 'socks',
    'MarkupSafe': 'markupsafe',
    'Jinja2': 'jinja2',
    'websocket-client': 'websocket',
    'trio-websocket': 'trio_websocket',
    'python-dateutil': 'dateutil'
}
    
    try:
        with open('requirements.txt') as f:
            required_packages = [line.strip().split('==')[0] for line in f if line.strip()]
    except FileNotFoundError:
        print("[ERROR] requirements.txt not found")
        return False, []

    missing_packages = []
    for package in required_packages:
        import_name = package_map.get(package, package)
        try:
            if import_name == 'PIL':
                from PIL import Image
            elif import_name == 'flask_cors':
                from flask_cors import CORS
            else:
                __import__(import_name)
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[MISSING] {package} (Error: {str(e)})")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_environment_variables():
    """Check if required environment variables are set"""
    load_dotenv()
    
    required_vars = ['OPENWEATHER_API_KEY', 'SECRET_KEY']
    optional_vars = ['OPENAI_API_KEY', 'FLASK_DEBUG', 'PORT']
    
    missing_required = []
    
    print("\n[INFO] Checking Environment Variables:")
    for var in required_vars:
        value = os.environ.get(var)
        if value and value != f'your_{var.lower().replace("_", "-")}_here':
            print(f"[OK] {var}: (set)")
        else:
            print(f"[MISSING] {var}: Not set or using template value")
            missing_required.append(var)
    
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            print(f"[OPTIONAL] {var}: {value if 'KEY' not in var else '(set)'}")
        else:
            print(f"[OPTIONAL] {var}: Not set")

    return len(missing_required) == 0, missing_required

def main():
    """Main setup function"""
    print("Climate Safe Home API - Setup Check\n")

    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("[INFO] Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check environment variables
    env_ok, missing_env = check_environment_variables()
    if not env_ok:
        print(f"\n[ERROR] Missing required environment variables: {', '.join(missing_env)}")
        print("[INFO] Please set these in your .env file")
        sys.exit(1)
    
    print("\n[INFO] Setup check complete. Starting the application...\n")

    # --- Ensure critical packages are force-installed ---
    for package in ['annotated-types', 'charset-normalizer', 'pillow']:
        try:
            __import__(package.replace('-', '_') if '-' in package else package)
        except ImportError:
            print(f"[FIX] Installing missing package: {package}")
            subprocess.call([sys.executable, "-m", "pip", "install", package])

    # Import and run the main application
    try:
        from app import app
        from app.utils.database import db_manager

        db_manager.init_database()
        print("[OK] Database initialized")

        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        port = int(os.environ.get('PORT', 5000))

        print(f"[START] Running server at http://localhost:{port}")
        print(f"[INFO] Debug mode: {debug_mode}")
        print(f"[INFO] Database file: {os.path.abspath('climate_safe_home.db')}")
        print("\nAvailable API Endpoints:")
        print(f"  - Health Check:     http://localhost:{port}/api/health")
        print(f"  - Analytics:        http://localhost:{port}/api/analytics")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 50)

        app.run(host='0.0.0.0', port=port, debug=debug_mode)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()