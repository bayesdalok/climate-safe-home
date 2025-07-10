import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "climate-safe-key")
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "your-openweather-api-key")
    GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "your-google-vision-api-key")

    # DB
    DATABASE_PATH = "climate_safe_home.db"
