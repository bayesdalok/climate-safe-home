from flask import jsonify
from app import app

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for development"""
    return jsonify({
        'success': True,
        'message': 'Climate Safe Home API is running',
        'version': '1.0.0',
        'endpoints': [
            '/api/health',
            '/api/assess',
            '/api/analytics',
            '/api/weather/<location>',
            '/api/history',
            '/api/upload',
            '/api/structure-types',
            '/api/foundation-types',
            '/api/roof-types'
        ]
    })