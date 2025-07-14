import logging
from openai import OpenAI
from app.config import Config, VULNERABILITY_CONSTANTS

logger = logging.getLogger(__name__)

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

vulnerability_calculator = VulnerabilityCalculator()