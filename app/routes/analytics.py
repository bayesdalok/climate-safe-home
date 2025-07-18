from flask import jsonify
from app import app
import logging
from app.config import Config
from app.utils.database import DatabaseManager

# Initialize logger and db_manager
logger = logging.getLogger(__name__)
db_manager = DatabaseManager(Config.DATABASE_PATH)

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard"""
    try:
        analytics = db_manager.get_analytics()
        return jsonify({
            'success': True,
            'data': analytics
        })
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500