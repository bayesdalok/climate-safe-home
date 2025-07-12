import logging
from openai import OpenAI
from app.config import Config, VULNERABILITY_CONSTANTS

logger = logging.getLogger(__name__)

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

recommendation_engine = RecommendationEngine()