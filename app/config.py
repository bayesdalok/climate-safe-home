import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "climate-safe-key")
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # No default
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

    # DB
    DATABASE_PATH = "climate_safe_home.db"

# Vulnerability constants for risk assessment and recommendations
VULNERABILITY_CONSTANTS = {
    # Structural vulnerability weights
    'STRUCTURAL_WEIGHTS': {
        'foundation_damage': 0.35,
        'wall_cracks': 0.25,
        'roof_damage': 0.30,
        'window_damage': 0.10
    },
    
    # Weather risk thresholds
    'WEATHER_THRESHOLDS': {
        'high_wind_speed': 50,  # km/h
        'medium_wind_speed': 25,  # km/h
        'high_temperature': 40,  # Celsius
        'medium_temperature': 35,  # Celsius
        'high_rain_probability': 70,  # percentage
        'medium_rain_probability': 40,  # percentage
        'heavy_rain_3h': 20,  # mm in 3 hours
        'moderate_rain_3h': 10  # mm in 3 hours
    },
    
    # Risk scoring parameters
    'RISK_SCORES': {
        'low': 1,
        'medium': 3,
        'high': 5
    },
    
    # Recommendation categories
    'RECOMMENDATION_CATEGORIES': {
        'immediate': 'Immediate Action Required',
        'short_term': 'Short-term Improvements',
        'long_term': 'Long-term Planning',
        'preventive': 'Preventive Measures'
    },
    
    # Priority levels
    'PRIORITY_LEVELS': {
        'critical': 5,
        'high': 4,
        'medium': 3,
        'low': 2,
        'minimal': 1
    },
    
    # Vulnerability types
    'VULNERABILITY_TYPES': {
        'structural': 'Structural Damage',
        'weather': 'Weather-related Risk',
        'environmental': 'Environmental Hazard',
        'maintenance': 'Maintenance Issue'
    },
        # Vulnerability scoring constants
    'BASE_SCORES': {
        'wood': 60,
        'concrete': 50,
        'brick': 55,
        'bamboo': 65,
        'stone': 58
    },
    'AGE_MULTIPLIER': 0.3,
    'FLOOR_MULTIPLIER': {
        '1': 2,
        '2': 5,
        '3': 8
    },
    'FOUNDATION_MULTIPLIER': {
        'strip': 5,
        'raft': 6,
        'pile': 7,
        'concrete': 0
    },
    'ROOF_MULTIPLIER': {
        'sloped': 2,
        'flat': 5,
        'metal': 4,
        'thatch': 6
    },

}