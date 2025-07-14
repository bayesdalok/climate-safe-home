from flask_cors import CORS
from .config import Config
from .utils.logger import configure_logging
from app.services.weather import weather_analyzer
from .utils.database import DatabaseManager
import logging
import sqlite3
from flask import Flask, send_from_directory, request, jsonify
import os

app = Flask(__name__, static_folder='../static')
CORS(app)
app.config.from_object(Config)

# Logging setup
configure_logging()
logger = logging.getLogger(__name__)

# Initialize database
DatabaseManager(Config.DATABASE_PATH).init_database()

# Before route imports in __init__.py
from app.services import (
    structural_analyzer,
    weather_analyzer,
    vulnerability_calculator
)

# Import routes
from .routes import (
    assessments, 
    builders, 
    weather,
    recommendations,
    uploads,
    types,
    errors,
    test
)

# Add sample builder data if table is empty
try:
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM builders')
    if cursor.fetchone()[0] == 0:
        sample_builders = [
            ('ABC Construction', 'Mumbai', 'Certified', 2, 85.5),
            ('XYZ Builders', 'Delhi', 'Pending', 5, 72.3),
            ('PQR Contractors', 'Bangalore', 'Certified', 1, 91.2)
        ]
        cursor.executemany('''
            INSERT INTO builders (name, location, certification_status, complaints, success_rate)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_builders)
        conn.commit()
    conn.close()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    exit(1)

def create_app():
    # If you need factory pattern
    app = Flask(__name__, static_folder='static')
    # Configure your app here
    return app

# After app initialization in __init__.py
if not os.path.exists(Config.DATABASE_PATH):
    logger.info(f"Creating new database at {Config.DATABASE_PATH}")
    open(Config.DATABASE_PATH, 'w').close()  # Create empty file

    
from flask import send_from_directory
import os

# Serve index.html for all frontend routes (SPA fallback)
# Add this catch-all route (MUST be last route definition)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_dir = os.path.join(app.root_path, '../static')
    # First try to serve static files
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    # Then fall back to index.html
    return send_from_directory(static_dir, 'index.html')