from flask import jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import datetime
import base64
from app import app
from app.config import Config
import logging

logger = logging.getLogger(__name__)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload for assessment"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Add file validation
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Check file size (additional check)
        if len(file.read()) > Config.MAX_CONTENT_LENGTH:
            return jsonify({'success': False, 'error': 'File too large'}), 413
        file.seek(0)  # Reset file pointer
        
        if file:
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            # Save file
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Convert to base64 for processing
            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
                image_data = f"data:image/jpeg;base64,{image_data}"
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True, 
                'data': {
                    'image_data': image_data,
                    'filename': filename
                }
            })
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def serve_index():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        # Create a basic index.html if it doesn't exist
        with open(os.path.join(static_dir, 'index.html'), 'w') as f:
            f.write('<h1>Climate Safe Home API</h1><p>API is running. Use /api/test to test endpoints.</p>')
    return send_from_directory('static', 'index.html')