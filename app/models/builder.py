# app/models/builder.py
from app.utils.database import DatabaseManager
from app.config import Config

def get_builder_data(builder_name):
    """Fetch builder data from database"""
    try:
        with DatabaseManager(Config.DATABASE_PATH) as db:
            cursor = db.execute(
                "SELECT * FROM builders WHERE name = ?", 
                (builder_name,)
            )
            builder = cursor.fetchone()
            
            if builder:
                return {
                    'name': builder[0],
                    'location': builder[1],
                    'certification_status': builder[2],
                    'complaints': builder[3],
                    'success_rate': builder[4],
                    'last_updated': builder[5]
                }
            return None
    except Exception as e:
        raise e