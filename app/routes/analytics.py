from flask import jsonify
from app.utils.database import DatabaseManager
from app.config import Config

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard"""
    try:
        analytics = db_manager.get_analytics()
        return jsonify({'success': True, 'data': analytics})
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500