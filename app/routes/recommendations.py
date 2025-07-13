from flask import jsonify, request
from app.services.recommender import recommendation_engine
from app import app
import json
import logging

logger = logging.getLogger(__name__)

@app.route('/api/recommendations/<structure_type>', methods=['GET'])
def get_recommendations_by_type(structure_type):
    """Get recommendations for a specific structure type"""
    try:
        # Get query parameters
        vulnerability_score = float(request.args.get('score', 50))
        weather_risks = json.loads(request.args.get('weather_risks', '{}'))
        structural_issues = json.loads(request.args.get('issues', '[]'))
        
        recommendations = recommendation_engine.generate_recommendations(
            structure_type,
            vulnerability_score,
            weather_risks,
            structural_issues
        )
        
        return jsonify({'success': True, 'data': recommendations})
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500