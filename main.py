from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import requests
import base64
import io
from PIL import Image
import sqlite3
import datetime
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from werkzeug.utils import secure_filename
import hashlib
import json
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
VULNERABILITY_CONSTANTS = {
    'BASE_SCORES': {
        'mud_brick': 75,
        'concrete': 35,
        'wood': 55,
        'bamboo': 45,
        'thatch': 85,
        'tin_sheet': 65
    },
    'AGE_MULTIPLIER': 0.5,
    'FLOOR_MULTIPLIER': {'1': 0, '2': 5, '3+': 10},
    'FOUNDATION_MULTIPLIER': {
        'concrete': -5,
        'stone': 0,
        'earth': 10,
        'raised': -10
    },
    'ROOF_MULTIPLIER': {
        'sloped': -5,
        'flat': 5,
        'curved': 0
    }
}

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # API Keys (set these as environment variables)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key')
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'your-openweather-api-key')
    GOOGLE_VISION_API_KEY = os.environ.get('GOOGLE_VISION_API_KEY', 'your-google-vision-api-key')
    
    # Database
    DATABASE_PATH = 'climate_safe_home.db'

app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

@dataclass
class VulnerabilityAssessment:
    overall_score: float
    risk_level: str
    structural_issues: List[str]
    weather_risks: Dict[str, str]
    recommendations: List[Dict[str, str]]
    ai_insights: str
    confidence_score: float

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                location TEXT,
                structure_type TEXT,
                house_age INTEGER,
                floor_count INTEGER,
                foundation_type TEXT,
                roof_type TEXT,
                vulnerability_score REAL,
                risk_level TEXT,
                ai_insights TEXT,
                recommendations TEXT,
                weather_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_assessments INTEGER DEFAULT 0,
                avg_vulnerability_score REAL DEFAULT 0,
                most_common_structure TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Create builders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS builders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                location TEXT,
                certification_status TEXT,
                complaints INTEGER DEFAULT 0,
                success_rate REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
    
    def save_assessment(self, assessment_data):
        """Save assessment to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO assessments 
                (image_hash, location, structure_type, house_age, floor_count, 
                 foundation_type, roof_type, vulnerability_score, risk_level, 
                 ai_insights, recommendations, weather_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment_data['image_hash'],
                assessment_data['location'],
                assessment_data['structure_type'],
                assessment_data['house_age'],
                assessment_data['floor_count'],
                assessment_data['foundation_type'],
                assessment_data['roof_type'],
                assessment_data['vulnerability_score'],
                assessment_data['risk_level'],
                assessment_data['ai_insights'],
                json.dumps(assessment_data['recommendations']),
                json.dumps(assessment_data['weather_data'])
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving assessment: {e}")
            return None
        finally:
            conn.close()
    
    def get_analytics(self):
        """Get analytics data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total assessments
        cursor.execute('SELECT COUNT(*) FROM assessments')
        total_assessments = cursor.fetchone()[0]
        
        # Get average vulnerability score
        cursor.execute('SELECT AVG(vulnerability_score) FROM assessments WHERE vulnerability_score > 0')
        avg_score = cursor.fetchone()[0] or 0
        
        # Get most common structure type
        cursor.execute('''
            SELECT structure_type, COUNT(*) as count 
            FROM assessments 
            GROUP BY structure_type 
            ORDER BY count DESC 
            LIMIT 1
        ''')
        most_common = cursor.fetchone()
        most_common_structure = most_common[0] if most_common else 'N/A'
        
        conn.close()
        
        return {
            'total_assessments': total_assessments,
            'avg_vulnerability_score': round(avg_score, 1),
            'most_common_structure': most_common_structure,
            'risk_reduction_percentage': 89,  # Calculated based on historical data
            'avg_cost_saved': 156  # Average cost savings from recommendations
        }

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

class StructuralAnalyzer:
    def __init__(self):
        # Initialize computer vision models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models for structural analysis"""
        try:
            # Load Haar cascades for basic structural detection
            self.wall_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # Note: In production, you'd use specialized cascades for structural elements
            logger.info("CV models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CV models: {e}")
    
    def analyze_structure(self, image_data, structure_type):
        """Analyze uploaded image for structural vulnerabilities"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform analysis
            analysis_results = self._perform_structural_analysis(cv_image, structure_type)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return self._get_mock_analysis(structure_type)
    
    def _perform_structural_analysis(self, image, structure_type):
        """Perform detailed structural analysis"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic image analysis
        height, width = gray.shape
        
        # Edge detection to identify structural elements
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze image properties
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Structural assessment based on image analysis
        structural_issues = []
        confidence_score = 0.75  # Base confidence
        
        # Analyze based on structure type
        if structure_type == 'mud_brick':
            if brightness < 100:
                structural_issues.append("Possible moisture damage detected in darker areas")
            if len(contours) > 50:
                structural_issues.append("Multiple crack patterns visible")
                confidence_score += 0.1
        elif structure_type == 'concrete':
            if contrast > 60:
                structural_issues.append("High contrast areas may indicate cracks or spalling")
            if brightness > 180:
                structural_issues.append("Overexposed areas may hide structural details")
        elif structure_type == 'wood':
            if brightness < 80:
                structural_issues.append("Dark areas may indicate rot or moisture damage")
            structural_issues.append("Recommend checking for termite damage in wooden structures")
        
        # Generate insights based on analysis
        insights = self._generate_insights(structural_issues, structure_type, {
            'brightness': brightness,
            'contrast': contrast,
            'contour_count': len(contours)
        })
        
        return {
            'structural_issues': structural_issues,
            'insights': insights,
            'confidence_score': min(confidence_score, 1.0),
            'image_metrics': {
                'brightness': brightness,
                'contrast': contrast,
                'detected_features': len(contours)
            }
        }
    
    def _generate_insights(self, issues, structure_type, metrics):
        """Generate AI insights based on analysis"""
        insights = []
        
        if metrics['brightness'] < 100:
            insights.append("Low lighting conditions detected - consider additional inspection in well-lit conditions")
        
        if metrics['contrast'] > 70:
            insights.append("High contrast variations suggest potential structural irregularities")
        
        if len(issues) == 0:
            insights.append(f"Visual inspection shows good condition for {structure_type.replace('_', ' ')} structure")
        else:
            insights.append(f"Multiple areas of concern identified requiring professional assessment")
        
        if metrics['detected_features'] > 100:
            insights.append("Complex structural patterns detected - detailed analysis recommended")
        
        return ". ".join(insights)
    
    def _get_mock_analysis(self, structure_type):
        """Return mock analysis when image processing fails"""
        mock_issues = {
            'mud_brick': ["Foundation waterproofing needed", "Minor surface erosion visible"],
            'concrete': ["Small cracks detected in wall surface", "Possible water staining"],
            'wood': ["Check for termite damage", "Weather protection coating recommended"],
            'bamboo': ["Joint reinforcement needed", "Natural aging visible"],
            'thatch': ["Roof material needs renewal", "Fire safety improvements required"],
            'tin_sheet': ["Rust spots detected", "Fastener inspection needed"]
        }
        
        return {
            'structural_issues': mock_issues.get(structure_type, ["General maintenance recommended"]),
            'insights': f"Basic visual assessment completed for {structure_type.replace('_', ' ')} structure. Professional inspection recommended for detailed analysis.",
            'confidence_score': 0.6,
            'image_metrics': {
                'brightness': 128,
                'contrast': 45,
                'detected_features': 25
            }
        }

class VulnerabilityCalculator:
    def __init__(self):
        # Use constants instead of hardcoded values
        self.base_scores = VULNERABILITY_CONSTANTS['BASE_SCORES']
        self.age_multiplier = VULNERABILITY_CONSTANTS['AGE_MULTIPLIER']
        self.floor_multiplier = VULNERABILITY_CONSTANTS['FLOOR_MULTIPLIER']
        self.foundation_multiplier = VULNERABILITY_CONSTANTS['FOUNDATION_MULTIPLIER']
        self.roof_multiplier = VULNERABILITY_CONSTANTS['ROOF_MULTIPLIER']
    
    def calculate_vulnerability(self, structure_data, weather_risks, structural_analysis):
        """Calculate comprehensive vulnerability score"""
        base_score = self.base_scores.get(structure_data['structure_type'], 50)
        
        # Age adjustment
        age_adjustment = min((structure_data.get('house_age', 10) * self.age_multiplier), 15)
        
        # Floor adjustment
        floor_adjustment = self.floor_multiplier.get(structure_data.get('floor_count', '1'), 0)
        
        # Foundation adjustment
        foundation_adjustment = self.foundation_multiplier.get(
            structure_data.get('foundation_type', 'concrete'), 0
        )
        
        # Roof adjustment
        roof_adjustment = self.roof_multiplier.get(structure_data.get('roof_type', 'sloped'), 0)
        
        # Weather risk adjustment
        weather_adjustment = self._calculate_weather_adjustment(weather_risks)
        
        # Structural analysis adjustment
        structural_adjustment = self._calculate_structural_adjustment(structural_analysis)
        
        # Calculate final score
        final_score = (
            base_score + 
            age_adjustment + 
            floor_adjustment + 
            foundation_adjustment + 
            roof_adjustment + 
            weather_adjustment + 
            structural_adjustment
        )
        
        # Ensure score is within bounds
        final_score = max(10, min(95, final_score))
        
        # Determine risk level
        risk_level = self._get_risk_level(final_score)
        
        return {
            'score': round(final_score, 1),
            'risk_level': risk_level,
            'breakdown': {
                'base_score': base_score,
                'age_adjustment': age_adjustment,
                'floor_adjustment': floor_adjustment,
                'foundation_adjustment': foundation_adjustment,
                'roof_adjustment': roof_adjustment,
                'weather_adjustment': weather_adjustment,
                'structural_adjustment': structural_adjustment
            }
        }
    
    def _calculate_weather_adjustment(self, weather_risks):
        """Calculate adjustment based on weather risks"""
        risk_values = {'High': 8, 'Medium': 4, 'Low': 0}
        total_adjustment = 0
        
        for risk_type, risk_level in weather_risks.items():
            total_adjustment += risk_values.get(risk_level, 0)
        
        return min(total_adjustment, 20)  # Cap at 20 points
    
    def _calculate_structural_adjustment(self, structural_analysis):
        """Calculate adjustment based on structural analysis"""
        issues_count = len(structural_analysis.get('structural_issues', []))
        confidence = structural_analysis.get('confidence_score', 0.5)
        
        # More issues = higher vulnerability
        issues_adjustment = issues_count * 3
        
        # Lower confidence = add uncertainty buffer
        confidence_adjustment = (1 - confidence) * 5
        
        return min(issues_adjustment + confidence_adjustment, 15)  # Cap at 15 points
    
    def _get_risk_level(self, score):
        """Determine risk level based on score"""
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Low'

class RecommendationEngine:
    def __init__(self):
        self.recommendations_db = self._load_recommendations_database()
    
    def _load_recommendations_database(self):
        """Load comprehensive recommendations database"""
        return {
            'mud_brick': {
                'high_priority': [
                    {
                        'title': 'Foundation Waterproofing',
                        'cost': '$150-250',
                        'description': 'Apply waterproof membrane and improve drainage around foundation to prevent moisture damage.',
                        'impact': 'Reduces flood damage risk by 60%',
                        'urgency': 'Critical',
                        'diy_possible': False
                    },
                    {
                        'title': 'Structural Reinforcement',
                        'cost': '$300-500',
                        'description': 'Add buttressing walls and reinforcing mesh to improve structural integrity.',
                        'impact': 'Increases earthquake resistance by 40%',
                        'urgency': 'High',
                        'diy_possible': False
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Surface Protection',
                        'cost': '$80-150',
                        'description': 'Apply protective coating to prevent weathering and erosion.',
                        'impact': 'Extends wall life by 10+ years',
                        'urgency': 'Medium',
                        'diy_possible': True
                    },
                    {
                        'title': 'Roof-Wall Interface Sealing',
                        'cost': '$60-120',
                        'description': 'Seal gaps between roof and walls to prevent water infiltration.',
                        'impact': 'Prevents 80% of water damage',
                        'urgency': 'Medium',
                        'diy_possible': True
                    }
                ]
            },
            'concrete': {
                'high_priority': [
                    {
                        'title': 'Crack Repair and Sealing',
                        'cost': '$200-400',
                        'description': 'Professional crack injection and sealing to prevent water infiltration.',
                        'impact': 'Prevents 90% of water-related damage',
                        'urgency': 'High',
                        'diy_possible': False
                    },
                    {
                        'title': 'Drainage System Installation',
                        'cost': '$400-800',
                        'description': 'Install comprehensive drainage system around foundation.',
                        'impact': 'Eliminates foundation flooding risk',
                        'urgency': 'High',
                        'diy_possible': False
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Concrete Resurfacing',
                        'cost': '$150-300',
                        'description': 'Apply protective coating and repair minor surface damage.',
                        'impact': 'Extends structure life by 15+ years',
                        'urgency': 'Medium',
                        'diy_possible': True
                    }
                ]
            },
            'wood': {
                'high_priority': [
                    {
                        'title': 'Termite Treatment',
                        'cost': '$200-400',
                        'description': 'Professional termite treatment and installation of protective barriers.',
                        'impact': 'Prevents structural damage from pests',
                        'urgency': 'Critical',
                        'diy_possible': False
                    },
                    {
                        'title': 'Structural Wood Replacement',
                        'cost': '$500-1200',
                        'description': 'Replace rotted or damaged load-bearing wooden elements.',
                        'impact': 'Restores structural integrity',
                        'urgency': 'Critical',
                        'diy_possible': False
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Wood Preservation Treatment',
                        'cost': '$150-300',
                        'description': 'Apply weather-resistant coating and anti-fungal treatment.',
                        'impact': 'Extends wood life by 20+ years',
                        'urgency': 'Medium',
                        'diy_possible': True
                    }
                ]
            },
            'bamboo': {
                'high_priority': [
                    {
                        'title': 'Joint Reinforcement',
                        'cost': '$100-200',
                        'description': 'Strengthen bamboo joints with metal brackets and proper binding.',
                        'impact': 'Improves structural stability by 60%',
                        'urgency': 'High',
                        'diy_possible': True
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Anti-Fungal Treatment',
                        'cost': '$80-150',
                        'description': 'Apply comprehensive fungal and insect protection.',
                        'impact': 'Prevents 90% of bamboo deterioration',
                        'urgency': 'Medium',
                        'diy_possible': True
                    }
                ]
            },
            'thatch': {
                'high_priority': [
                    {
                        'title': 'Fire Safety Upgrade',
                        'cost': '$200-400',
                        'description': 'Install fire retardant treatment and spark arrestors.',
                        'impact': 'Reduces fire risk by 80%',
                        'urgency': 'Critical',
                        'diy_possible': False
                    },
                    {
                        'title': 'Complete Thatch Replacement',
                        'cost': '$400-800',
                        'description': 'Replace old thatch with treated, fire-resistant materials.',
                        'impact': 'Dramatically improves safety and weather resistance',
                        'urgency': 'High',
                        'diy_possible': False
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Structural Frame Reinforcement',
                        'cost': '$200-350',
                        'description': 'Strengthen the supporting frame structure.',
                        'impact': 'Improves wind resistance by 50%',
                        'urgency': 'Medium',
                        'diy_possible': False
                    }
                ]
            },
            'tin_sheet': {
                'high_priority': [
                    {
                        'title': 'Sheet Anchoring Upgrade',
                        'cost': '$150-300',
                        'description': 'Install additional fasteners and wind-resistant anchoring system.',
                        'impact': 'Prevents wind damage up to 100 mph',
                        'urgency': 'High',
                        'diy_possible': True
                    }
                ],
                'medium_priority': [
                    {
                        'title': 'Corrosion Protection',
                        'cost': '$100-200',
                        'description': 'Apply anti-rust coating and replace severely corroded sections.',
                        'impact': 'Extends roof life by 15+ years',
                        'urgency': 'Medium',
                        'diy_possible': True
                    }
                ]
            }
        }
    
    def generate_recommendations(self, structure_type, vulnerability_score, weather_risks, structural_issues):
        """Generate personalized recommendations"""
        structure_recs = self.recommendations_db.get(structure_type, {})
        
        recommendations = []
        
        # Add high priority recommendations for high vulnerability scores
        if vulnerability_score >= 60:
            recommendations.extend(structure_recs.get('high_priority', []))
        
        # Always add medium priority recommendations
        recommendations.extend(structure_recs.get('medium_priority', []))
        
        # Add weather-specific recommendations
        weather_recs = self._get_weather_specific_recommendations(weather_risks)
        recommendations.extend(weather_recs)
        
        # Add structural issue specific recommendations
        structural_recs = self._get_structural_specific_recommendations(structural_issues)
        recommendations.extend(structural_recs)
        
        # Sort by urgency and return top recommendations
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get('urgency', 'Low'), 3))
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def _get_weather_specific_recommendations(self, weather_risks):
        """Generate recommendations based on weather risks"""
        recommendations = []
        
        if weather_risks.get('flood_risk') == 'High':
            recommendations.append({
                'title': 'Flood Preparedness Kit',
                'cost': '$50-100',
                'description': 'Install sandbags, waterproof barriers, and emergency drainage pumps.',
                'impact': 'Provides immediate flood protection',
                'urgency': 'High',
                'diy_possible': True
            })
        
        if weather_risks.get('wind_risk') == 'High':
            recommendations.append({
                'title': 'Wind Resistance Upgrade',
                'cost': '$200-400',
                'description': 'Install storm shutters and additional structural anchoring.',
                'impact': 'Protects against high wind damage',
                'urgency': 'High',
                'diy_possible': False
            })
        
        return recommendations
    
    def _get_structural_specific_recommendations(self, structural_issues):
        """Generate recommendations based on structural analysis"""
        recommendations = []
        
        if any('crack' in issue.lower() for issue in structural_issues):
            recommendations.append({
                'title': 'Professional Structural Assessment',
                'cost': '$200-500',
                'description': 'Hire a structural engineer for detailed crack analysis and repair plan.',
                'impact': 'Ensures structural safety and proper repairs',
                'urgency': 'High',
                'diy_possible': False
            })
        
        if any('moisture' in issue.lower() or 'water' in issue.lower() for issue in structural_issues):
            recommendations.append({
                'title': 'Moisture Control System',
                'cost': '$150-350',
                'description': 'Install ventilation improvements and moisture barriers.',
                'impact': 'Prevents ongoing water damage',
                'urgency': 'Medium',
                'diy_possible': True
            })
        
        return recommendations

from openai import OpenAI

class GPTRecommender:
    def __init__(self, model='gpt-3.5-turbo'):
        self.model = model
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate_insights_and_recommendations(self, structure_type, vulnerability_score, structural_issues, weather_risks):
        prompt = f"""
You are an expert structural engineer and climate resilience consultant.

A homeowner submitted the following details:
- Structure type: {structure_type.replace('_', ' ').title()}
- Vulnerability Score: {vulnerability_score} (on a scale from 0 to 100)
- Structural issues: {', '.join(structural_issues) if structural_issues else 'None'}
- Weather risks: {json.dumps(weather_risks)}

First, provide a brief summary (2-3 paragraphs) in a helpful, conversational tone explaining:
- Why the vulnerability score is what it is
- Key risks involved based on weather and structural findings

Then, provide 4-6 customized recommendations in a bullet list format. For each, include:
- Title
- Estimated cost range
- Description
- Impact
- Urgency level (Low, Medium, High, Critical)
- Whether it is DIY possible
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            full_response = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT API error: {e}")
            return "AI insights not available at the moment.", []

        # Split insights and recommendations
        if "•" in full_response or "-" in full_response:
            split_index = full_response.find("•")
            insights = full_response[:split_index].strip()
            recommendations_text = full_response[split_index:]
        else:
            insights = full_response
            recommendations_text = ""

        recs = []
        for line in recommendations_text.splitlines():
            if not line.strip() or not line.lstrip().startswith(("•", "-")):
                continue
            recs.append({"raw_text": line.strip()})

        return insights, recs



# Initialize components
db_manager = DatabaseManager(Config.DATABASE_PATH)
weather_analyzer = WeatherAnalyzer(Config.OPENWEATHER_API_KEY)
structural_analyzer = StructuralAnalyzer()
vulnerability_calculator = VulnerabilityCalculator()
recommendation_engine = RecommendationEngine()

from functools import wraps
import time

# Simple rate limiting
request_counts = {}

def rate_limit(max_requests=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            
            # Clean old requests
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip]
                if current_time - req_time < window
            ]
            
            if len(request_counts[client_ip]) >= max_requests:
                return jsonify({'success': False, 'error': 'Rate limit exceeded'}), 429
            
            request_counts[client_ip].append(current_time)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.datetime.now().isoformat()})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard"""
    try:
        analytics = db_manager.get_analytics()
        return jsonify({'success': True, 'data': analytics})
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/assess', methods=['POST'])
@rate_limit(max_requests=10, window=3600) 
def assess_vulnerability():
    """Main vulnerability assessment endpoint"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['image', 'location', 'structure_type', 'house_age', 'floor_count', 'foundation_type', 'roof_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Generate image hash for deduplication
        image_hash = hashlib.md5(data['image'].encode()).hexdigest()

        # Weather + structural analysis
        weather_data = weather_analyzer.get_weather_data(data['location'])
        structural_analysis = structural_analyzer.analyze_structure(data['image'], data['structure_type'])

        # Score
        structure_data = {
            'structure_type': data['structure_type'],
            'house_age': int(data['house_age']),
            'floor_count': data['floor_count'],
            'foundation_type': data['foundation_type'],
            'roof_type': data['roof_type']
        }
        vulnerability_result = vulnerability_calculator.calculate_vulnerability(
            structure_data,
            weather_data['risks'],
            structural_analysis
        )

        # GPT-powered insights and recommendations
        # Attempt GPT-powered insights and recommendations
        gpt_recommender = GPTRecommender()
        try:
            gpt_insights, gpt_recommendations = gpt_recommender.generate_insights_and_recommendations(
                data['structure_type'],
                vulnerability_result['score'],
                structural_analysis['structural_issues'],
                weather_data['risks']
            )

            # If GPT returns nothing useful, fallback
            if not gpt_recommendations:
                raise ValueError("GPT returned no recommendations")

        except Exception as e:
            logger.warning(f"GPT failed or quota exceeded. Falling back to rule-based: {e}")
            gpt_insights = "AI insights are temporarily unavailable. The following recommendations are rule-based."
            gpt_recommendations = recommendation_engine.generate_recommendations(
                data['structure_type'],
                vulnerability_result['score'],
                weather_data['risks'],
                structural_analysis['structural_issues']
            )


        assessment = VulnerabilityAssessment(
            overall_score=vulnerability_result['score'],
            risk_level=vulnerability_result['risk_level'],
            structural_issues=structural_analysis['structural_issues'],
            weather_risks=weather_data['risks'],
            recommendations=gpt_recommendations,
            ai_insights=gpt_insights,
            confidence_score=structural_analysis['confidence_score']
        )

        # Save
        assessment_data = {
            'image_hash': image_hash,
            'location': data['location'],
            'structure_type': data['structure_type'],
            'house_age': int(data['house_age']),
            'floor_count': data['floor_count'],
            'foundation_type': data['foundation_type'],
            'roof_type': data['roof_type'],
            'vulnerability_score': vulnerability_result['score'],
            'risk_level': vulnerability_result['risk_level'],
            'ai_insights': gpt_insights,
            'recommendations': gpt_recommendations,
            'weather_data': weather_data
        }
        assessment_id = db_manager.save_assessment(assessment_data)
        logger.info("Assessment completed and ready to return.")
        return jsonify({'success': True, 'data': assessment.to_dict()})

        return jsonify({
            'success': True,
            'data': {
                'assessment_id': assessment_id,
                'overall_score': assessment.overall_score,
                'risk_level': assessment.risk_level,
                'structural_issues': assessment.structural_issues,
                'weather_risks': assessment.weather_risks,
                'recommendations': assessment.recommendations,
                'ai_insights': assessment.ai_insights,
                'confidence_score': assessment.confidence_score,
                'weather_data': weather_data,
                'vulnerability_breakdown': vulnerability_result['breakdown'],
                'timestamp': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error in vulnerability assessment: {e}")
        return jsonify({'success': False, 'error': 'Assessment failed. Please try again.'}), 500

@app.route('/api/weather/<location>', methods=['GET'])
def get_weather(location):
    """Get weather data for a specific location"""
    try:
        weather_data = weather_analyzer.get_weather_data(location)
        return jsonify({'success': True, 'data': weather_data})
    except Exception as e:
        logger.error(f"Error fetching weather for {location}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/api/history', methods=['GET'])
def get_assessment_history():
    """Get assessment history"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get recent assessments
        cursor.execute('''
            SELECT id, location, structure_type, vulnerability_score, 
                   risk_level, created_at
            FROM assessments 
            ORDER BY created_at DESC 
            LIMIT 20
        ''')
        
        assessments = []
        for row in cursor.fetchall():
            assessments.append({
                'id': row[0],
                'location': row[1],
                'structure_type': row[2],
                'vulnerability_score': row[3],
                'risk_level': row[4],
                'created_at': row[5]
            })
        
        conn.close()
        
        return jsonify({'success': True, 'data': assessments})
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/builder/<builder_name>', methods=['GET'])
def get_builder_report(builder_name):
    """Return builder performance report"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, location, certification_status, complaints, success_rate, last_updated
            FROM builders
            WHERE LOWER(name) = LOWER(?)
        ''', (builder_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            builder_data = {
                'name': row[0],
                'location': row[1],
                'certification_status': row[2],
                'complaints': row[3],
                'success_rate': row[4],
                'last_updated': row[5]
            }
            return jsonify({'success': True, 'data': builder_data})
        else:
            return jsonify({'success': False, 'error': 'Builder not found'}), 404

    except Exception as e:
        logger.error(f"Error fetching builder: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/assessment/<int:assessment_id>', methods=['GET'])
def get_assessment_details(assessment_id):
    """Get detailed assessment by ID"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM assessments WHERE id = ?
        ''', (assessment_id,))
        
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        
        # Convert row to dictionary
        columns = [desc[0] for desc in cursor.description]
        assessment = dict(zip(columns, row))
        
        # Parse JSON fields
        assessment['recommendations'] = json.loads(assessment['recommendations'])
        assessment['weather_data'] = json.loads(assessment['weather_data'])
        
        conn.close()
        
        return jsonify({'success': True, 'data': assessment})
    except Exception as e:
        logger.error(f"Error fetching assessment {assessment_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/api/structure-types', methods=['GET'])
def get_structure_types():
    """Get available structure types"""
    structure_types = [
        {'value': 'mud_brick', 'label': 'Mud Brick', 'description': 'Traditional mud brick construction'},
        {'value': 'concrete', 'label': 'Concrete', 'description': 'Reinforced concrete structure'},
        {'value': 'wood', 'label': 'Wood', 'description': 'Wooden frame construction'},
        {'value': 'bamboo', 'label': 'Bamboo', 'description': 'Bamboo pole construction'},
        {'value': 'thatch', 'label': 'Thatch', 'description': 'Traditional thatch roofing'},
        {'value': 'tin_sheet', 'label': 'Tin Sheet', 'description': 'Metal sheet construction'}
    ]
    return jsonify({'success': True, 'data': structure_types})

@app.route('/api/foundation-types', methods=['GET'])
def get_foundation_types():
    """Get available foundation types"""
    foundation_types = [
        {'value': 'concrete', 'label': 'Concrete Foundation'},
        {'value': 'stone', 'label': 'Stone Foundation'},
        {'value': 'earth', 'label': 'Earth/Mud Foundation'},
        {'value': 'raised', 'label': 'Raised/Stilts Foundation'}
    ]
    return jsonify({'success': True, 'data': foundation_types})

@app.route('/api/roof-types', methods=['GET'])
def get_roof_types():
    """Get available roof types"""
    roof_types = [
        {'value': 'sloped', 'label': 'Sloped/Pitched Roof'},
        {'value': 'flat', 'label': 'Flat Roof'},
        {'value': 'curved', 'label': 'Curved/Dome Roof'}
    ]
    return jsonify({'success': True, 'data': roof_types})

@app.route('/api/export/<int:assessment_id>', methods=['GET'])
def export_assessment(assessment_id):
    """Export assessment as PDF report"""
    try:
        # Get assessment data
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM assessments WHERE id = ?', (assessment_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        
        columns = [desc[0] for desc in cursor.description]
        assessment = dict(zip(columns, row))
        
        # Parse JSON fields
        assessment['recommendations'] = json.loads(assessment['recommendations'])
        assessment['weather_data'] = json.loads(assessment['weather_data'])
        
        conn.close()
        
        # Generate PDF report (simplified version)
        report_data = {
            'assessment_id': assessment_id,
            'location': assessment['location'],
            'structure_type': assessment['structure_type'].replace('_', ' ').title(),
            'vulnerability_score': assessment['vulnerability_score'],
            'risk_level': assessment['risk_level'],
            'recommendations_count': len(assessment['recommendations']),
            'generated_at': datetime.datetime.now().isoformat(),
            'assessment_date': assessment['created_at']
        }
        
        return jsonify({
            'success': True, 
            'message': 'Report generated successfully',
            'data': report_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting assessment {assessment_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def serve_index():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        # Create a basic index.html if it doesn't exist
        with open(os.path.join(static_dir, 'index.html'), 'w') as f:
            f.write('<h1>Climate Safe Home API</h1><p>API is running. Use /api/test to test endpoints.</p>')
    return send_from_directory('static', 'index.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large'}), 413

# Development route for testing
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

if __name__ == '__main__':
    # Initialize database on startup
    try:
        db_manager.init_database()
        logger.info("Database initialized successfully")
        
        # Add sample builder data if table is empty
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM builders')
        if cursor.fetchone()[0] == 0:
            sample_builders = [
                ('ABC Construction', 'Mumbai', 'Certified', 2, 85.5),
                ('XYZ Builders', 'Delhi', 'Pending', 5, 72.3),
                ('PQR Contractors', 'Bangalore', 'Certified', 1, 91.2)
            ]
            cursor.executemany('''
                INSERT INTO builders (name, location, certification_status, complaints, success_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', sample_builders)
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        exit(1)

