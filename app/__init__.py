from flask import Flask
from flask_cors import CORS
from .config import Config
from .utils.logger import configure_logging
from .utils.database import DatabaseManager
import logging
import sqlite3
from flask import jsonify

app = Flask(__name__, static_folder='static')
CORS(app)
app.config.from_object(Config)

# Logging setup
configure_logging()
logger = logging.getLogger(__name__)

# Initialize database
DatabaseManager(Config.DATABASE_PATH).init_database()

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

from flask import send_from_directory
import os

# Serve index.html for all frontend routes (SPA fallback)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    static_dir = os.path.join(app.root_path, 'static')
    
    # Handle API routes first
    if path.startswith('api/'):
        return jsonify({'success': False, 'error': 'API endpoint not found'}), 404

    # Check if the file exists
    file_path = os.path.join(static_dir, path)
    if path != "" and os.path.exists(file_path):
        return send_from_directory(static_dir, path)
    
    # Default to index.html for SPA routing
    return send_from_directory(static_dir, 'index.html')