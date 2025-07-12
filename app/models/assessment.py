import sqlite3
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
from app.config import Config

@dataclass
class VulnerabilityAssessment:
    overall_score: float
    risk_level: str
    structural_issues: List[str]
    weather_risks: Dict[str, str]
    recommendations: List[Dict[str, str]]
    ai_insights: str
    confidence_score: float
    timestamp: Optional[str] = None
    location: Optional[str] = None
    assessment_id: Optional[str] = None

    def __post_init__(self):
        """Set default values after initialization"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.assessment_id is None:
            self.assessment_id = self._generate_assessment_id()

    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID based on content and timestamp"""
        content = f"{self.timestamp}_{self.overall_score}_{len(self.structural_issues)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self):
        """Convert assessment to dictionary"""
        return asdict(self)

    def to_json(self):
        """Convert assessment to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict):
        """Create VulnerabilityAssessment from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str):
        """Create VulnerabilityAssessment from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            location TEXT,
            overall_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            structural_issues TEXT,
            weather_risks TEXT,
            recommendations TEXT,
            ai_insights TEXT,
            confidence_score REAL,
            raw_data TEXT
        )
    ''')
    
    # Create images table for storing uploaded images
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_type TEXT,
            analysis_result TEXT,
            FOREIGN KEY (assessment_id) REFERENCES assessments (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_assessment(assessment: VulnerabilityAssessment, image_paths: List[str] = None) -> str:
    """Save vulnerability assessment to database"""
    try:
        init_database()
        
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Insert assessment data
        cursor.execute('''
            INSERT OR REPLACE INTO assessments (
                id, timestamp, location, overall_score, risk_level,
                structural_issues, weather_risks, recommendations,
                ai_insights, confidence_score, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment.assessment_id,
            assessment.timestamp,
            assessment.location,
            assessment.overall_score,
            assessment.risk_level,
            json.dumps(assessment.structural_issues),
            json.dumps(assessment.weather_risks),
            json.dumps(assessment.recommendations),
            assessment.ai_insights,
            assessment.confidence_score,
            assessment.to_json()
        ))
        
        # Save associated images if provided
        if image_paths:
            for image_path in image_paths:
                cursor.execute('''
                    INSERT INTO assessment_images (assessment_id, image_path, image_type)
                    VALUES (?, ?, ?)
                ''', (assessment.assessment_id, image_path, 'structural_damage'))
        
        conn.commit()
        conn.close()
        
        return assessment.assessment_id
        
    except Exception as e:
        print(f"Error saving assessment: {e}")
        return None

def get_assessment(assessment_id: str) -> Optional[VulnerabilityAssessment]:
    """Retrieve assessment from database by ID"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT raw_data FROM assessments WHERE id = ?
        ''', (assessment_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return VulnerabilityAssessment.from_json(result[0])
        return None
        
    except Exception as e:
        print(f"Error retrieving assessment: {e}")
        return None

def get_recent_assessments(limit: int = 10) -> List[VulnerabilityAssessment]:
    """Get recent assessments from database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT raw_data FROM assessments 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        assessments = []
        for result in results:
            try:
                assessment = VulnerabilityAssessment.from_json(result[0])
                assessments.append(assessment)
            except Exception as e:
                print(f"Error parsing assessment: {e}")
                continue
        
        return assessments
        
    except Exception as e:
        print(f"Error retrieving recent assessments: {e}")
        return []

def delete_assessment(assessment_id: str) -> bool:
    """Delete assessment from database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete associated images
        cursor.execute('DELETE FROM assessment_images WHERE assessment_id = ?', (assessment_id,))
        
        # Delete assessment
        cursor.execute('DELETE FROM assessments WHERE id = ?', (assessment_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
        
    except Exception as e:
        print(f"Error deleting assessment: {e}")
        return False

def get_assessment_statistics() -> Dict:
    """Get statistics about assessments"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM assessments')
        total_count = cursor.fetchone()[0]
        
        # Get risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) 
            FROM assessments 
            GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        # Get average scores
        cursor.execute('''
            SELECT AVG(overall_score), AVG(confidence_score)
            FROM assessments
        ''')
        avg_scores = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_assessments': total_count,
            'risk_distribution': risk_distribution,
            'average_overall_score': avg_scores[0] if avg_scores[0] else 0,
            'average_confidence_score': avg_scores[1] if avg_scores[1] else 0
        }
        
    except Exception as e:
        print(f"Error getting assessment statistics: {e}")
        return {
            'total_assessments': 0,
            'risk_distribution': {},
            'average_overall_score': 0,
            'average_confidence_score': 0
        }

# Initialize database when module is imported
try:
    init_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")