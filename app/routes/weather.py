from flask import jsonify
from app import app
from app.services.weather import weather_analyzer

@app.route('/api/weather/<location>', methods=['GET'])
def get_weather(location):
    """Get weather data for a specific location"""
    try:
        weather_data = weather_analyzer.get_weather_data(location)
        return jsonify({'success': True, 'data': weather_data})
    except Exception as e:
        logger.error(f"Error fetching weather for {location}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500