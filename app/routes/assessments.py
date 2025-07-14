from flask import jsonify, request
from app.services.structural import structural_analyzer
from app.services.weather import weather_analyzer
from app.services.recommender import recommendation_engine, GPTRecommender
from app.models.assessment import VulnerabilityAssessment, save_assessment
from app.utils.limiter import rate_limit
from app import app
import numpy as np
import hashlib
from app.services.vulnerabity import vulnerability_calculator

import logging
logger = logging.getLogger(__name__)

@app.route('/api/assess', methods=['POST'])
@rate_limit(max_requests=1000, window=3600) 
def assess_vulnerability():
    """Main vulnerability assessment endpoint"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['images', 'location', 'structure_type', 'house_age', 'floor_count', 'foundation_type', 'roof_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Generate image hash for deduplication
        images = data['images']
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({'success': False, 'error': 'At least one image is required'}), 400

        combined_issues = []
        total_confidence = 0
        all_metrics = []

        for img_b64 in images[:5]:  # Limit to 5 images
            analysis = structural_analyzer.analyze_structure(img_b64, data['structure_type'])
            combined_issues.extend(analysis.get('structural_issues', []))
            total_confidence += analysis.get('confidence_score', 0.6)
            all_metrics.append(analysis.get('image_metrics', {}))

        avg_confidence = round(total_confidence / len(images), 2)

        # Combine all structural analysis into one object
        structural_analysis = {
            'structural_issues': list(set(combined_issues)),  # Remove duplicates
            'confidence_score': avg_confidence,
            'image_metrics': {
                'brightness': round(np.mean([m['brightness'] for m in all_metrics]), 1),
                'contrast': round(np.mean([m['contrast'] for m in all_metrics]), 1),
                'detected_features': int(np.mean([m['detected_features'] for m in all_metrics]))
            }
        }

        # Use first image's hash for caching or identification
        image_hash = hashlib.md5(images[0].encode()).hexdigest()

        weather_data = weather_analyzer.get_weather_data(data['location'])
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
            # Ensure fallback recommendations are fully structured
            if not gpt_recommendations:
                fallback_recs = []

                structure_type = structure_data.get("structure_type", "").lower()
                foundation_type = structure_data.get("foundation_type", "").lower()
                roof_type = structure_data.get("roof_type", "").lower()
                house_age = int(structure_data.get("house_age", 0))
                floor_count = int(structure_data.get("floor_count", 0))
                location = structure_data.get("location", "").lower()

                risk_level = risk_level.lower()
                issues_text = " ".join(structural_issues).lower()

                # 🔹 Weather risk logic
                if weather_data.get("flood_risk") in ["High", "Medium"]:
                    fallback_recs.append({
                        "title": "Install Foundation Drainage",
                        "description": "Improves water flow away from structure during floods.",
                        "cost": "$400-700",
                        "impact": "Reduces basement seepage and erosion",
                        "urgency": "High",
                        "diy_possible": False
                    })

                if weather_data.get("wind_risk") in ["High", "Medium"]:
                    fallback_recs.append({
                        "title": "Roof Bracing & Anchor Systems",
                        "description": "Prevents roof displacement during storms.",
                        "cost": "$300-500",
                        "impact": "Increases wind stability",
                        "urgency": "High",
                        "diy_possible": False
                    })

                if weather_data.get("rain_risk") in ["High", "Medium"]:
                    fallback_recs.append({
                        "title": "Apply Waterproof Coatings",
                        "description": "Protect walls and roof from moisture ingress.",
                        "cost": "$150-300",
                        "impact": "Reduces rain seepage and mold",
                        "urgency": "Medium",
                        "diy_possible": True
                    })

                if weather_data.get("heat_risk") == "High":
                    fallback_recs.append({
                        "title": "Cool Roof System",
                        "description": "Install reflective coatings or tiles.",
                        "cost": "$200-400",
                        "impact": "Lowers indoor temp by 4-7°C",
                        "urgency": "Medium",
                        "diy_possible": True
                    })

                # 🔹 Structural issues
                if "crack" in issues_text:
                    fallback_recs.append({
                        "title": "Crack Injection & Sealing",
                        "description": "Prevent moisture and structural weakening.",
                        "cost": "$250-500",
                        "impact": "Improves longevity of walls",
                        "urgency": "High",
                        "diy_possible": True
                    })

                if "damp" in issues_text or "stain" in issues_text:
                    fallback_recs.append({
                        "title": "Moisture Remediation",
                        "description": "Check drainage slope and seal damp spots.",
                        "cost": "$150-350",
                        "impact": "Eliminates mold and wall decay",
                        "urgency": "Medium",
                        "diy_possible": True
                    })

                # 🔹 Structure type
                if structure_type == "wood":
                    fallback_recs.append({
                        "title": "Termite Barrier System",
                        "description": "Treat soil and wooden parts against pests.",
                        "cost": "$200-400",
                        "impact": "Prevents termite infestations",
                        "urgency": "High",
                        "diy_possible": False
                    })

                if structure_type == "concrete" and house_age > 25:
                    fallback_recs.append({
                        "title": "Concrete Resurfacing",
                        "description": "Refinish surface cracks and aging concrete.",
                        "cost": "$300-600",
                        "impact": "Restores strength of older slabs",
                        "urgency": "Medium",
                        "diy_possible": False
                    })

                # 🔹 Roof-based
                if roof_type == "flat" and weather_data.get("rain_risk") == "High":
                    fallback_recs.append({
                        "title": "Add Roof Drains or Sloping",
                        "description": "Flat roofs require enhanced drainage to prevent pooling.",
                        "cost": "$200-500",
                        "impact": "Eliminates stagnant water risks",
                        "urgency": "High",
                        "diy_possible": False
                    })

                if roof_type == "metal" and weather_data.get("heat_risk") == "High":
                    fallback_recs.append({
                        "title": "Install Insulated Roof Panels",
                        "description": "Reduce radiated heat in metal-roof homes.",
                        "cost": "$350-600",
                        "impact": "Improves comfort by reducing heat gain",
                        "urgency": "Medium",
                        "diy_possible": False
                    })

                # 🔹 Foundation type
                if foundation_type in ["strip", "raft"]:
                    fallback_recs.append({
                        "title": "Waterproof Basement Walls",
                        "description": "Essential for shallow or wide foundations.",
                        "cost": "$300-700",
                        "impact": "Reduces rising damp",
                        "urgency": "High",
                        "diy_possible": False
                    })

                # 🔹 Floor count
                if floor_count > 2:
                    fallback_recs.append({
                        "title": "Seismic Retrofitting",
                        "description": "Improves structural stability for multi-storey buildings.",
                        "cost": "$500-1000",
                        "impact": "Reduces collapse risk during tremors",
                        "urgency": "High",
                        "diy_possible": False
                    })

                # 🔹 Risk-level based
                if risk_level == "high":
                    fallback_recs.append({
                        "title": "Urgent Climate Audit",
                        "description": "Consult a certified assessor to review structural and location risks.",
                        "cost": "$500+",
                        "impact": "Provides expert-certified risk plan",
                        "urgency": "High",
                        "diy_possible": False
                    })

                if not fallback_recs:
                    fallback_recs.append({
                        "title": "Basic Home Safety Check",
                        "description": "Perform routine maintenance and visual inspection.",
                        "cost": "Varies",
                        "impact": "Identifies hidden risks",
                        "urgency": "Low",
                        "diy_possible": True
                    })

                gpt_insights = "AI unavailable. Recommendations are generated using structure, location, foundation, roof type, age, and weather risk rules."
                gpt_recommendations = fallback_recs


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

    except Exception as e:
        logger.error(f"Error in vulnerability assessment: {e}")
        return jsonify({'success': False, 'error': 'Assessment failed. Please try again.'}), 500

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
        import traceback
        print("Unhandled exception in /api/assess")
        traceback.print_exc()

        # If logger fails for some reason, fallback to print
        try:
            logger.error(f"Error in vulnerability assessment: {e}")
        except Exception:
            pass

        return jsonify({'success': False, 'error': 'Assessment failed. Please try again.'}), 500
