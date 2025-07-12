import requests
import logging
from app.config import Config

logger = logging.getLogger(__name__)

class WeatherAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def get_weather_data(self, location):
        """Get current weather and forecast data"""
        try:
        # Validate API key
            if not self.api_key or self.api_key == 'your-openweather-api-key':
                logger.warning("OpenWeather API key not configured, using mock data")
                return self._get_mock_weather_data()
        
        # Get current weather
            current_url = f"{self.base_url}/weather"
            current_params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
            }
        
            current_response = requests.get(current_url, params=current_params, timeout=10)
            current_data = current_response.json()
        
            if current_response.status_code != 200:
                logger.error(f"Weather API error: {current_data}")
                return self._get_mock_weather_data()
                    
            # Get forecast data
            forecast_url = f"{self.base_url}/forecast"
            forecast_params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
            forecast_data = forecast_response.json()
            
            return self._process_weather_data(current_data, forecast_data)
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data()
    
    def _process_weather_data(self, current, forecast):
        """Process weather data into risk assessment format"""
        # Calculate risks based on weather conditions
        wind_speed = current.get('wind', {}).get('speed', 0) * 3.6  # Convert to km/h
        humidity = current.get('main', {}).get('humidity', 0)
        
        # Analyze forecast for extreme weather patterns
        forecast_list = forecast.get('list', [])
        max_wind = max([item.get('wind', {}).get('speed', 0) * 3.6 for item in forecast_list[:8]])  # Next 24 hours
        rain_probability = max([item.get('pop', 0) * 100 for item in forecast_list[:8]])
        
        # Determine risk levels
        flood_risk = self._calculate_flood_risk(current, forecast_list)
        wind_risk = self._calculate_wind_risk(max_wind)
        heat_risk = self._calculate_heat_risk(current.get('main', {}).get('temp', 25))
        rain_risk = self._calculate_rain_risk(rain_probability)
        
        return {
            'current_weather': {
                'temperature': current.get('main', {}).get('temp', 0),
                'humidity': humidity,
                'wind_speed': wind_speed,
                'description': current.get('weather', [{}])[0].get('description', 'Clear')
            },
            'risks': {
                'flood_risk': flood_risk,
                'wind_risk': wind_risk,
                'heat_risk': heat_risk,
                'rain_risk': rain_risk
            },
            'forecast_summary': {
                'max_wind_24h': max_wind,
                'rain_probability': rain_probability
            }
        }
    
    def _calculate_flood_risk(self, current, forecast_list):
        """Calculate flood risk based on weather patterns"""
        rain_3h = current.get('rain', {}).get('3h', 0)
        if rain_3h > 20:
            return 'High'
        elif rain_3h > 10:
            return 'Medium'
        return 'Low'
    
    def _calculate_wind_risk(self, wind_speed):
        """Calculate wind risk based on speed"""
        if wind_speed > 50:
            return 'High'
        elif wind_speed > 25:
            return 'Medium'
        return 'Low'
    
    def _calculate_heat_risk(self, temperature):
        """Calculate heat risk based on temperature"""
        if temperature > 40:
            return 'High'
        elif temperature > 35:
            return 'Medium'
        return 'Low'
    
    def _calculate_rain_risk(self, rain_probability):
        """Calculate rain risk based on probability"""
        if rain_probability > 70:
            return 'High'
        elif rain_probability > 40:
            return 'Medium'
        return 'Low'
    
    def _get_mock_weather_data(self):
        """Return mock weather data when API is unavailable"""
        return {
            'current_weather': {
                'temperature': 28,
                'humidity': 65,
                'wind_speed': 15,
                'description': 'Partly cloudy'
            },
            'risks': {
                'flood_risk': 'Medium',
                'wind_risk': 'High',
                'heat_risk': 'Low',
                'rain_risk': 'High'
            },
            'forecast_summary': {
                'max_wind_24h': 25,
                'rain_probability': 80
            }
        }

# Create a function to get the weather analyzer instance
def get_weather_analyzer():
    """Get weather analyzer instance with proper error handling"""
    try:
        api_key = getattr(Config, 'OPENWEATHER_API_KEY', None)
        return WeatherAnalyzer(api_key)
    except AttributeError:
        logger.warning("Config.OPENWEATHER_API_KEY not found, using None")
        return WeatherAnalyzer(None)

# Create the instance using the function
weather_analyzer = get_weather_analyzer()