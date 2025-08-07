#!/usr/bin/env python3
"""
Startup script for Climate Safe Home API
This script helps with initial setup and environment checking
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from app import app 
from app.utils.database import DatabaseManager
import logging
from app.utils.logger import configure_logging


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
    required_packages = [
        'flask', 'flask_cors', 'cv2', 'numpy',
        'requests', 'PIL', 'sqlite3', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'flask_cors':
                from flask_cors import CORS
            elif package == 'dotenv':
                from dotenv import load_dotenv
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
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

def install_dependencies():
    """Install required dependencies"""
    print("\n[INFO] Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file from template"""
    if not os.path.exists('.env') and os.path.exists('.env.template'):
        print("\n[INFO] Creating .env file from template...")
        with open('.env.template', 'r') as template:
            with open('.env', 'w') as env_file:
                env_file.write(template.read())
        print("[OK] .env file created. Please edit it with your API keys.")
        return True
    elif os.path.exists('.env'):
        print("[OK] .env file already exists")
        return True
    else:
        print("[ERROR] No .env template found")
        return False

def main():
    """Main setup function"""
    # Configure logging first
    configure_logging()
    logger = logging.getLogger(__name__)
    print("Climate Safe Home API - Setup Check\n")

    # Check Python version
    if not check_python_version():
        return

    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        install_deps = input("Install missing dependencies? (y/n): ").lower().strip()
        if install_deps == 'y':
           if not install_dependencies():
               return
        else:
            print("[INFO] Please install dependencies manually: pip install -r requirements.txt")
        return
    
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
        #app = create_app()
        DatabaseManager("climate_safe_home.db").init_database()
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

        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug = False)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()