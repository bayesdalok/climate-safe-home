import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import json
from enum import Enum
from models.llm_fallback import LLMFallback
import openai
from config import Config
from utils.logger import logger

logger = logging.getLogger(__name__)

class StructureType(Enum):
    """Supported structure types"""
    MUD_BRICK = "mud_brick"
    CONCRETE = "concrete"
    WOOD = "wood"
    BAMBOO = "bamboo"
    THATCH = "thatch"
    TIN_SHEET = "tin_sheet"
    STONE = "stone"
    BRICK = "brick"
    STEEL = "steel"

class VulnerabilityLevel(Enum):
    """Vulnerability assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnalysisResult:
    """Structured analysis result"""
    structural_issues: List[str]
    vulnerability_level: VulnerabilityLevel
    confidence_score: float
    climate_risks: List[str]
    recommendations: List[str]
    insights: str
    image_metrics: Dict[str, Any]
    detailed_analysis: Dict[str, Any]

class OpenAIAnalyzer:
    def __init__(self):
        self.client = openai
        openai.api_key = Config.OPENAI_API_KEY  # Ensure this is in config.py

    async def analyze(
        self, 
        prompt: str, 
        images: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Primary analysis using OpenAI with enhanced error handling.
        Returns: 
            {
                "analysis": str, 
                "confidence": float,
                "raw_response": dict (optional)
            }
        """
        try:
            messages = [{"role": "user", "content": ""}]
            model = "gpt-3.5-turbo"  # Default model
            
            # Handle image analysis if provided
            if images:
                content = [{"type": "text", "text": prompt}]
                for img in images:
                    if not isinstance(img, str):
                        raise ValueError("Image must be a base64 string")
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    })
                messages[0]["content"] = content
                model = "gpt-4-vision-preview"
            else:
                messages[0]["content"] = prompt

            response = await self.client.ChatCompletion.acreate(
                model=model,
                messages=messages,
                max_tokens=1000
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "confidence": 0.9,  # High confidence for OpenAI
                "raw_response": response  # For debugging
            }

        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError("OpenAI API unavailable")
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            raise RuntimeError("OpenAI processing error")

class StructuralAnalyzer:
    def __init__(self):
        """Initialize the structural analyzer with enhanced capabilities"""
        self.structure_properties = self._load_structure_properties()
        self.climate_vulnerability_matrix = self._load_climate_vulnerability_matrix()
        self.analysis_thresholds = self._load_analysis_thresholds()
        self.load_models()
        logger.info("Enhanced Structural Analyzer initialized successfully")
    
    def _load_structure_properties(self) -> Dict[str, Dict[str, Any]]:
        """Load material properties for different structure types"""
        return {
            StructureType.MUD_BRICK.value: {
                'water_resistance': 'low',
                'thermal_mass': 'high',
                'earthquake_resistance': 'medium',
                'durability': 'medium',
                'common_issues': ['erosion', 'cracking', 'moisture_damage', 'foundation_settlement'],
                'maintenance_frequency': 'high'
            },
            StructureType.CONCRETE.value: {
                'water_resistance': 'high',
                'thermal_mass': 'high',
                'earthquake_resistance': 'high',
                'durability': 'high',
                'common_issues': ['cracking', 'spalling', 'rebar_corrosion', 'carbonation'],
                'maintenance_frequency': 'low'
            },
            StructureType.WOOD.value: {
                'water_resistance': 'medium',
                'thermal_mass': 'low',
                'earthquake_resistance': 'high',
                'durability': 'medium',
                'common_issues': ['rot', 'termite_damage', 'warping', 'moisture_damage'],
                'maintenance_frequency': 'medium'
            },
            StructureType.BAMBOO.value: {
                'water_resistance': 'low',
                'thermal_mass': 'low',
                'earthquake_resistance': 'high',
                'durability': 'medium',
                'common_issues': ['joint_failure', 'insect_damage', 'splitting', 'weathering'],
                'maintenance_frequency': 'high'
            },
            StructureType.THATCH.value: {
                'water_resistance': 'medium',
                'thermal_mass': 'medium',
                'earthquake_resistance': 'medium',
                'durability': 'low',
                'common_issues': ['fire_risk', 'pest_infestation', 'weathering', 'sagging'],
                'maintenance_frequency': 'very_high'
            },
            StructureType.TIN_SHEET.value: {
                'water_resistance': 'high',
                'thermal_mass': 'very_low',
                'earthquake_resistance': 'low',
                'durability': 'medium',
                'common_issues': ['rust', 'thermal_expansion', 'wind_damage', 'noise'],
                'maintenance_frequency': 'medium'
            }
        }
    
    def _load_climate_vulnerability_matrix(self) -> Dict[str, Dict[str, str]]:
        """Load climate vulnerability matrix for different materials"""
        return {
            'high_temperature': {
                StructureType.MUD_BRICK.value: 'low',
                StructureType.CONCRETE.value: 'medium',
                StructureType.WOOD.value: 'medium',
                StructureType.TIN_SHEET.value: 'high'
            },
            'heavy_rainfall': {
                StructureType.MUD_BRICK.value: 'high',
                StructureType.CONCRETE.value: 'low',
                StructureType.WOOD.value: 'medium',
                StructureType.THATCH.value: 'medium'
            },
            'flooding': {
                StructureType.MUD_BRICK.value: 'critical',
                StructureType.CONCRETE.value: 'medium',
                StructureType.WOOD.value: 'high',
                StructureType.BAMBOO.value: 'high'
            },
            'high_humidity': {
                StructureType.WOOD.value: 'high',
                StructureType.BAMBOO.value: 'high',
                StructureType.TIN_SHEET.value: 'medium',
                StructureType.CONCRETE.value: 'low'
            },
            'cyclone_winds': {
                StructureType.THATCH.value: 'critical',
                StructureType.TIN_SHEET.value: 'high',
                StructureType.BAMBOO.value: 'medium',
                StructureType.CONCRETE.value: 'low'
            }
        }
    
    def _load_analysis_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load analysis thresholds for different metrics"""
        return {
            'brightness': {
                'very_dark': 50,
                'dark': 100,
                'normal': 150,
                'bright': 200,
                'overexposed': 230
            },
            'contrast': {
                'very_low': 20,
                'low': 40,
                'normal': 60,
                'high': 80,
                'very_high': 100
            },
            'edge_density': {
                'smooth': 0.02,
                'normal': 0.05,
                'textured': 0.08,
                'highly_textured': 0.12
            }
        }
    
    def load_models(self):
        """Load computer vision models for structural analysis"""
        try:
            # Initialize edge detection parameters
            self.canny_low = 50
            self.canny_high = 150
            
            # Initialize morphological kernels
            self.crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Color space conversion parameters
            self.hsv_moisture_lower = np.array([10, 50, 50])
            self.hsv_moisture_upper = np.array([30, 255, 255])
            
            logger.info("CV models and parameters loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CV models: {e}")
            raise
    
    def analyze_structure(self, image_data: str, structure_type: str) -> Dict[str, Any]:
        """
        Analyze uploaded image for structural vulnerabilities with enhanced analysis
        
        Args:
            image_data: Base64 encoded image data
            structure_type: Type of structure being analyzed
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Validate inputs
            if not self._validate_structure_type(structure_type):
                raise ValueError(f"Unsupported structure type: {structure_type}")
            
            # Decode and validate image
            image = self._decode_and_validate_image(image_data)
            if image is None:
                raise ValueError("Invalid image data")
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform comprehensive analysis
            analysis_result = self._perform_enhanced_structural_analysis(cv_image, structure_type)
            
            # Convert to dictionary for JSON serialization
            return self._convert_result_to_dict(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return self._get_enhanced_fallback_analysis(structure_type, str(e))
    
    def _validate_structure_type(self, structure_type: str) -> bool:
        """Validate if structure type is supported"""
        return structure_type in [st.value for st in StructureType]
    
    def _decode_and_validate_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode and validate base64 image data"""
        try:
            if ',' in image_data:
                image_bytes = base64.b64decode(image_data.split(',')[1])
            else:
                image_bytes = base64.b64decode(image_data)
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # Validate image dimensions
            if image.width < 100 or image.height < 100:
                raise ValueError("Image too small for analysis")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _perform_enhanced_structural_analysis(self, image: np.ndarray, structure_type: str) -> AnalysisResult:
        """Perform comprehensive structural analysis"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Basic image metrics
        height, width = gray.shape
        image_metrics = self._calculate_image_metrics(gray, hsv, lab)
        
        # Structural feature detection
        structural_features = self._detect_structural_features(gray, hsv, structure_type)
        
        # Damage detection
        damage_analysis = self._detect_damage_patterns(gray, hsv, structure_type)
        
        # Climate vulnerability assessment
        climate_risks = self._assess_climate_vulnerability(structure_type, structural_features, damage_analysis)
        
        # Generate structural issues
        structural_issues = self._compile_structural_issues(structural_features, damage_analysis, structure_type)
        
        # Calculate vulnerability level
        vulnerability_level = self._calculate_vulnerability_level(structural_issues, damage_analysis, climate_risks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(structural_issues, vulnerability_level, structure_type)
        
        # Generate insights
        insights = self._generate_enhanced_insights(structural_issues, image_metrics, structure_type, vulnerability_level)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(image_metrics, structural_features)
        
        return AnalysisResult(
            structural_issues=structural_issues,
            vulnerability_level=vulnerability_level,
            confidence_score=confidence_score,
            climate_risks=climate_risks,
            recommendations=recommendations,
            insights=insights,
            image_metrics=image_metrics,
            detailed_analysis={
                'structural_features': structural_features,
                'damage_analysis': damage_analysis,
                'material_properties': self.structure_properties.get(structure_type, {})
            }
        )
    
    def _calculate_image_metrics(self, gray: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive image metrics"""
        # Basic metrics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Texture analysis
        texture_score = self._calculate_texture_score(gray)
        
        # Color analysis
        color_diversity = self._calculate_color_diversity(hsv)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'texture_score': texture_score,
            'color_diversity': color_diversity,
            'image_quality': self._assess_image_quality(brightness, contrast, edge_density)
        }
    
    def _calculate_texture_score(self, gray: np.ndarray) -> float:
        """Calculate texture score using gradient analysis"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return float(np.mean(magnitude))
    
    def _calculate_color_diversity(self, hsv: np.ndarray) -> float:
        """Calculate color diversity in the image"""
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        
        # Normalize
        hist_norm = hist / hist.sum()
        
        # Calculate entropy as measure of diversity
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        return float(entropy)
    
    def _assess_image_quality(self, brightness: float, contrast: float, edge_density: float) -> str:
        """Assess overall image quality for analysis"""
        thresholds = self.analysis_thresholds
        
        quality_score = 0
        
        # Brightness assessment
        if thresholds['brightness']['dark'] < brightness < thresholds['brightness']['bright']:
            quality_score += 1
        
        # Contrast assessment
        if contrast > thresholds['contrast']['normal']:
            quality_score += 1
        
        # Edge density assessment
        if edge_density > thresholds['edge_density']['normal']:
            quality_score += 1
        
        if quality_score >= 3:
            return "excellent"
        elif quality_score >= 2:
            return "good"
        elif quality_score >= 1:
            return "fair"
        else:
            return "poor"
    
    def _detect_structural_features(self, gray: np.ndarray, hsv: np.ndarray, structure_type: str) -> Dict[str, Any]:
        """Detect structural features specific to material type"""
        features = {}
        
        # Edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features['contour_count'] = len(contours)
        features['edge_coverage'] = float(np.sum(edges > 0) / edges.size)
        
        # Line detection for structural elements
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        features['line_count'] = len(lines) if lines is not None else 0
        
        # Material-specific feature detection
        if structure_type == StructureType.MUD_BRICK.value:
            features.update(self._detect_mud_brick_features(gray, hsv))
        elif structure_type == StructureType.CONCRETE.value:
            features.update(self._detect_concrete_features(gray, hsv))
        elif structure_type == StructureType.WOOD.value:
            features.update(self._detect_wood_features(gray, hsv))
        
        return features
    
    def _detect_mud_brick_features(self, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect mud brick specific features"""
        # Look for brick patterns and mortar joints
        # Apply morphological operations to detect rectangular patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Detect horizontal patterns (mortar joints)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(morph, cv2.MORPH_OPEN, horizontal_kernel)
        
        return {
            'mortar_joint_visibility': float(np.sum(horizontal_lines > 0) / horizontal_lines.size),
            'surface_uniformity': float(np.std(gray) / np.mean(gray))
        }
    
    def _detect_concrete_features(self, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect concrete specific features"""
        # Look for concrete texture and potential reinforcement patterns
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect circular patterns (potential rebar)
        circles = cv2.HoughCircles(filtered, cv2.HOUGH_GRADIENT, dp=1, minDist=30, 
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        
        return {
            'surface_smoothness': float(np.mean(cv2.Laplacian(filtered, cv2.CV_64F).var())),
            'potential_rebar_count': len(circles[0]) if circles is not None else 0
        }
    
    def _detect_wood_features(self, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect wood specific features"""
        # Look for wood grain patterns and potential defects
        # Apply Gabor filters to detect wood grain
        kernel = cv2.getGaborKernel((21, 21), 5, 0, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        return {
            'grain_visibility': float(np.std(gabor_response)),
            'surface_variation': float(np.ptp(gray))  # Peak-to-peak range
        }
    
    def _detect_damage_patterns(self, gray: np.ndarray, hsv: np.ndarray, structure_type: str) -> Dict[str, Any]:
        """Detect various damage patterns"""
        damage_analysis = {}
        
        # Crack detection
        damage_analysis['cracks'] = self._detect_cracks(gray)
        
        # Moisture damage detection
        damage_analysis['moisture'] = self._detect_moisture_damage(hsv)
        
        # Surface deterioration
        damage_analysis['surface_deterioration'] = self._detect_surface_deterioration(gray)
        
        # Structural deformation
        damage_analysis['deformation'] = self._detect_structural_deformation(gray)
        
        return damage_analysis
    
    def _detect_cracks(self, gray: np.ndarray) -> Dict[str, Any]:
        """Enhanced crack detection"""
        # Apply morphological operations to enhance crack-like features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Apply threshold to isolate dark linear features
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours and filter for crack-like shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crack_contours = []
        for contour in contours:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-10)
            area = cv2.contourArea(contour)
            
            # Filter for crack-like features (high aspect ratio, sufficient area)
            if aspect_ratio > 3 and area > 50:
                crack_contours.append(contour)
        
        return {
            'detected': len(crack_contours) > 0,
            'count': len(crack_contours),
            'total_length': sum(cv2.arcLength(contour, False) for contour in crack_contours),
            'severity': self._assess_crack_severity(crack_contours)
        }
    
    def _assess_crack_severity(self, crack_contours: List[np.ndarray]) -> str:
        """Assess crack severity based on detected features"""
        if not crack_contours:
            return "none"
        
        total_length = sum(cv2.arcLength(contour, False) for contour in crack_contours)
        max_length = max(cv2.arcLength(contour, False) for contour in crack_contours)
        
        if total_length > 500 or max_length > 200:
            return "severe"
        elif total_length > 200 or max_length > 100:
            return "moderate"
        else:
            return "minor"
    
    def _detect_moisture_damage(self, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect moisture damage indicators"""
        # Create mask for moisture-related colors (browns, yellows, dark spots)
        mask = cv2.inRange(hsv, self.hsv_moisture_lower, self.hsv_moisture_upper)
        
        # Calculate affected area
        total_pixels = hsv.shape[0] * hsv.shape[1]
        affected_pixels = np.sum(mask > 0)
        affected_percentage = (affected_pixels / total_pixels) * 100
        
        return {
            'detected': affected_percentage > 0.5,
            'affected_area_percentage': float(affected_percentage),
            'severity': self._assess_moisture_severity(affected_percentage)
        }
    
    def _assess_moisture_severity(self, affected_percentage: float) -> str:
        """Assess moisture damage severity"""
        if affected_percentage > 15:
            return "severe"
        elif affected_percentage > 8:
            return "moderate"
        elif affected_percentage > 2:
            return "minor"
        else:
            return "none"
    
    def _detect_surface_deterioration(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect surface deterioration patterns"""
        # Calculate local variance to identify areas of deterioration
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Identify areas with high variance (potential deterioration)
        high_variance_threshold = np.percentile(local_variance, 90)
        deterioration_mask = local_variance > high_variance_threshold
        
        deterioration_percentage = (np.sum(deterioration_mask) / deterioration_mask.size) * 100
        
        return {
            'detected': deterioration_percentage > 5,
            'affected_area_percentage': float(deterioration_percentage),
            'severity': self._assess_deterioration_severity(deterioration_percentage)
        }
    
    def _assess_deterioration_severity(self, deterioration_percentage: float) -> str:
        """Assess surface deterioration severity"""
        if deterioration_percentage > 25:
            return "severe"
        elif deterioration_percentage > 15:
            return "moderate"
        elif deterioration_percentage > 8:
            return "minor"
        else:
            return "none"
    
    def _detect_structural_deformation(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect structural deformation indicators"""
        # Use line detection to identify structural elements
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {'detected': False, 'severity': 'none'}
        
        # Analyze line angles to detect potential deformation
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Check for deviation from expected angles (horizontal/vertical)
        angle_deviations = []
        for angle in angles:
            deviation = min(abs(angle), abs(angle - 90), abs(angle + 90), abs(angle - 180))
            angle_deviations.append(deviation)
        
        max_deviation = max(angle_deviations) if angle_deviations else 0
        
        return {
            'detected': max_deviation > 10,
            'max_deviation_degrees': float(max_deviation),
            'severity': self._assess_deformation_severity(max_deviation)
        }
    
    def _assess_deformation_severity(self, max_deviation: float) -> str:
        """Assess structural deformation severity"""
        if max_deviation > 30:
            return "severe"
        elif max_deviation > 20:
            return "moderate"
        elif max_deviation > 10:
            return "minor"
        else:
            return "none"
    
    def _assess_climate_vulnerability(self, structure_type: str, structural_features: Dict[str, Any], 
                                    damage_analysis: Dict[str, Any]) -> List[str]:
        """Assess climate-specific vulnerability"""
        climate_risks = []
        
        # Get vulnerability matrix for this structure type
        vulnerability_matrix = self.climate_vulnerability_matrix
        
        # Check each climate risk
        for climate_factor, vulnerabilities in vulnerability_matrix.items():
            if structure_type in vulnerabilities:
                vulnerability_level = vulnerabilities[structure_type]
                
                # Adjust based on detected damage
                if damage_analysis.get('moisture', {}).get('detected', False) and 'rainfall' in climate_factor:
                    climate_risks.append(f"High vulnerability to {climate_factor.replace('_', ' ')} due to existing moisture damage")
                elif damage_analysis.get('surface_deterioration', {}).get('detected', False):
                    climate_risks.append(f"Increased {climate_factor.replace('_', ' ')} vulnerability due to surface deterioration")
                elif vulnerability_level in ['high', 'critical']:
                    climate_risks.append(f"Inherent vulnerability to {climate_factor.replace('_', ' ')}")
        
        return climate_risks
    
    def _compile_structural_issues(self, structural_features: Dict[str, Any], 
                                  damage_analysis: Dict[str, Any], structure_type: str) -> List[str]:
        """Compile all detected structural issues"""
        issues = []
        
        # Check crack issues
        if damage_analysis.get('cracks', {}).get('detected', False):
            severity = damage_analysis['cracks']['severity']
            count = damage_analysis['cracks']['count']
            issues.append(f"{severity.capitalize()} cracking detected - {count} crack(s) identified")
        
        # Check moisture issues
        if damage_analysis.get('moisture', {}).get('detected', False):
            severity = damage_analysis['moisture']['severity']
            percentage = damage_analysis['moisture']['affected_area_percentage']
            issues.append(f"{severity.capitalize()} moisture damage affecting {percentage:.1f}% of visible area")
        
        # Check surface deterioration
        if damage_analysis.get('surface_deterioration', {}).get('detected', False):
            severity = damage_analysis['surface_deterioration']['severity']
            issues.append(f"{severity.capitalize()} surface deterioration detected")
        
        # Check structural deformation
        if damage_analysis.get('deformation', {}).get('detected', False):
            severity = damage_analysis['deformation']['severity']
            deviation = damage_analysis['deformation']['max_deviation_degrees']
            issues.append(f"{severity.capitalize()} structural deformation - {deviation:.1f}° deviation from expected alignment")
        
        # Add material-specific issues
        material_properties = self.structure_properties.get(structure_type, {})
        common_issues = material_properties.get('common_issues', [])
        
        # Add recommendations for common issues based on material type
        if not issues:  # If no specific issues detected, mention prevention
            issues.append(f"No major structural issues detected. Monitor for common {structure_type.replace('_', ' ')} issues: {', '.join(common_issues[:2])}")
        
        return issues
    
    def _calculate_vulnerability_level(self, structural_issues: List[str], 
                                     damage_analysis: Dict[str, Any], 
                                     climate_risks: List[str]) -> VulnerabilityLevel:
        """Calculate overall vulnerability level"""
        score = 0
        
        # Base score from number of issues
        score += len(structural_issues) * 0.2
        
        # Severity adjustments
        for damage_type, analysis in damage_analysis.items():
            if isinstance(analysis, dict) and 'severity' in analysis:
                severity = analysis['severity']
                if severity == 'severe':
                    score += 0.4
                elif severity == 'moderate':
                    score += 0.25
                elif severity == 'minor':
                    score += 0.1
        
        # Climate risk adjustments
        score += len(climate_risks) * 0.15
        
        # Determine vulnerability level
        if score >= 1.0:
            return VulnerabilityLevel.CRITICAL
        elif score >= 0.7:
            return VulnerabilityLevel.HIGH
        elif score >= 0.4:
            return VulnerabilityLevel.MEDIUM
        else:
            return VulnerabilityLevel.LOW
    
    def _generate_recommendations(self, structural_issues: List[str], 
                                vulnerability_level: VulnerabilityLevel, 
                                structure_type: str) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        # Get material properties
        material_properties = self.structure_properties.get(structure_type, {})
        maintenance_freq = material_properties.get('maintenance_frequency', 'medium')
        
        # Vulnerability-based recommendations
        if vulnerability_level == VulnerabilityLevel.CRITICAL:
            recommendations.append("URGENT: Immediate structural assessment by qualified engineer required")
            recommendations.append("Consider temporary relocation until safety is confirmed")
            recommendations.append("Implement emergency structural reinforcement measures")
        
        elif vulnerability_level == VulnerabilityLevel.HIGH:
            recommendations.append("Schedule professional structural inspection within 30 days")
            recommendations.append("Implement preventive measures to halt further deterioration")
            recommendations.append("Consider structural reinforcement options")
        
        elif vulnerability_level == VulnerabilityLevel.MEDIUM:
            recommendations.append("Schedule routine maintenance within 90 days")
            recommendations.append("Monitor identified issues for progression")
            recommendations.append("Implement preventive maintenance program")
        
        else:  # LOW
            recommendations.append("Continue regular maintenance schedule")
            recommendations.append("Monitor for early signs of deterioration")
        
        # Issue-specific recommendations
        if any("crack" in issue.lower() for issue in structural_issues):
            recommendations.append("Seal cracks to prevent water infiltration")
            recommendations.append("Monitor crack progression with regular measurements")
        
        if any("moisture" in issue.lower() for issue in structural_issues):
            recommendations.append("Improve ventilation and drainage around structure")
            recommendations.append("Apply water-resistant treatments if appropriate")
            recommendations.append("Address source of moisture infiltration")
        
        if any("deterioration" in issue.lower() for issue in structural_issues):
            recommendations.append("Replace or repair deteriorated sections")
            recommendations.append("Apply protective coatings where applicable")
        
        # Material-specific recommendations
        if structure_type == StructureType.MUD_BRICK.value:
            recommendations.append("Apply lime plaster for weather protection")
            recommendations.append("Ensure proper roof overhang to protect walls")
            recommendations.append("Regular repointing of mortar joints")
        
        elif structure_type == StructureType.WOOD.value:
            recommendations.append("Apply wood preservative treatments")
            recommendations.append("Ensure proper ventilation to prevent moisture buildup")
            recommendations.append("Regular inspection for termite activity")
        
        elif structure_type == StructureType.CONCRETE.value:
            recommendations.append("Apply concrete sealers to prevent carbonation")
            recommendations.append("Ensure proper drainage around foundation")
            recommendations.append("Monitor for rebar corrosion signs")
        
        elif structure_type == StructureType.THATCH.value:
            recommendations.append("Replace thatch material every 10-15 years")
            recommendations.append("Maintain proper roof pitch for water runoff")
            recommendations.append("Install fire-resistant barriers where possible")
        
        elif structure_type == StructureType.TIN_SHEET.value:
            recommendations.append("Apply rust-resistant coatings regularly")
            recommendations.append("Secure loose sheets to prevent wind damage")
            recommendations.append("Install proper insulation for thermal comfort")
        
        elif structure_type == StructureType.BAMBOO.value:
            recommendations.append("Treat bamboo with natural preservatives")
            recommendations.append("Ensure joints are properly secured")
            recommendations.append("Replace damaged sections promptly")
        
        return recommendations
    
    def _generate_enhanced_insights(self, structural_issues: List[str], 
                                  image_metrics: Dict[str, Any], 
                                  structure_type: str, 
                                  vulnerability_level: VulnerabilityLevel) -> str:
        """Generate comprehensive insights about the structure"""
        insights = []
        
        # Structure type insights
        material_props = self.structure_properties.get(structure_type, {})
        durability = material_props.get('durability', 'unknown')
        
        insights.append(f"This {structure_type.replace('_', ' ')} structure shows {durability} durability characteristics.")
        
        # Image quality insights
        image_quality = image_metrics.get('image_quality', 'fair')
        brightness = image_metrics.get('brightness', 0)
        contrast = image_metrics.get('contrast', 0)
        
        insights.append(f"Image analysis quality: {image_quality} (brightness: {brightness:.1f}, contrast: {contrast:.1f}).")
        
        # Vulnerability insights
        if vulnerability_level == VulnerabilityLevel.CRITICAL:
            insights.append("Critical structural concerns identified requiring immediate attention.")
        elif vulnerability_level == VulnerabilityLevel.HIGH:
            insights.append("Significant structural issues detected that need prompt intervention.")
        elif vulnerability_level == VulnerabilityLevel.MEDIUM:
            insights.append("Moderate structural concerns that should be addressed through regular maintenance.")
        else:
            insights.append("Structure appears to be in good condition with minor or no issues detected.")
        
        # Issue-specific insights
        if structural_issues:
            insights.append(f"Primary concerns include: {', '.join(structural_issues[:2])}.")
        else:
            insights.append("No major structural defects were detected in the visible areas.")
        
        # Maintenance insights
        maintenance_freq = material_props.get('maintenance_frequency', 'medium')
        insights.append(f"This material type requires {maintenance_freq} maintenance frequency.")
        
        # Climate considerations
        water_resistance = material_props.get('water_resistance', 'medium')
        insights.append(f"Water resistance is {water_resistance} - consider climate-appropriate protection measures.")
        
        return " ".join(insights)
    
    def _calculate_confidence_score(self, image_metrics: Dict[str, Any], 
                                  structural_features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Image quality factors
        image_quality = image_metrics.get('image_quality', 'fair')
        if image_quality == 'excellent':
            confidence += 0.3
        elif image_quality == 'good':
            confidence += 0.2
        elif image_quality == 'fair':
            confidence += 0.1
        
        # Feature detection factors
        edge_coverage = structural_features.get('edge_coverage', 0)
        if edge_coverage > 0.05:
            confidence += 0.1
        
        contour_count = structural_features.get('contour_count', 0)
        if contour_count > 50:
            confidence += 0.1
        
        # Brightness and contrast factors
        brightness = image_metrics.get('brightness', 0)
        contrast = image_metrics.get('contrast', 0)
        
        if 80 < brightness < 200 and contrast > 30:
            confidence += 0.1
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def _convert_result_to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult to dictionary for JSON serialization"""
        return {
            'structural_issues': result.structural_issues,
            'vulnerability_level': result.vulnerability_level.value,
            'confidence_score': result.confidence_score,
            'climate_risks': result.climate_risks,
            'recommendations': result.recommendations,
            'insights': result.insights,
            'image_metrics': result.image_metrics,
            'detailed_analysis': result.detailed_analysis
        }
    
    def _get_enhanced_fallback_analysis(self, structure_type: str, error_msg: str) -> Dict[str, Any]:
        """Enhanced fallback analysis when image processing fails"""
        material_properties = self.structure_properties.get(structure_type, {})
        common_issues = material_properties.get('common_issues', [])
        
        return {
            'structural_issues': [f"Unable to analyze image: {error_msg}"],
            'vulnerability_level': VulnerabilityLevel.MEDIUM.value,
            'confidence_score': 0.1,
            'climate_risks': [f"General climate vulnerability for {structure_type.replace('_', ' ')} structures"],
            'recommendations': [
                "Manual inspection recommended due to image analysis failure",
                "Consider retaking image with better lighting and resolution",
                f"Monitor for common {structure_type.replace('_', ' ')} issues: {', '.join(common_issues[:3])}"
            ],
            'insights': f"Image analysis failed, but {structure_type.replace('_', ' ')} structures typically require {material_properties.get('maintenance_frequency', 'regular')} maintenance.",
            'image_metrics': {
                'brightness': 50.0,
                'contrast': 50.0,
                'edge_density': 0.0,
                'image_quality': 'failed_to_process',
                'error': error_msg
            },

            'detailed_analysis': {
                'error': error_msg,
                'material_properties': material_properties
            }
        }
    
    def get_structure_types(self) -> List[str]:
        """Get list of supported structure types"""
        return [st.value for st in StructureType]
    
    def get_vulnerability_levels(self) -> List[str]:
        """Get list of vulnerability levels"""
        return [vl.value for vl in VulnerabilityLevel]
    
    def validate_image_size(self, image_data: str) -> bool:
        """Validate if image size is appropriate for analysis"""
        try:
            image = self._decode_and_validate_image(image_data)
            return image is not None and image.width >= 300 and image.height >= 300
        except Exception:
            return False
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Get a brief summary of the analysis"""
        vulnerability = analysis_result.get('vulnerability_level', 'unknown')
        confidence = analysis_result.get('confidence_score', 0)
        issue_count = len(analysis_result.get('structural_issues', []))
        
        return f"Vulnerability: {vulnerability.upper()}, Issues: {issue_count}, Confidence: {confidence:.1%}"

# Usage example and testing functions
def create_analyzer():
    """Factory function to create analyzer instance"""
    return StructuralAnalyzer()

def test_analyzer():
    """Test function for the analyzer"""
    analyzer = create_analyzer()
    
    # Test structure types
    print("Supported structure types:")
    for st in analyzer.get_structure_types():
        print(f"  - {st}")
    
    # Test vulnerability levels
    print("\nVulnerability levels:")
    for vl in analyzer.get_vulnerability_levels():
        print(f"  - {vl}")
    
    print("\nAnalyzer initialized successfully!")
    return analyzer

structural_analyzer = None

def initialize_analyzer():
    """Initialize the global analyzer instance"""
    global structural_analyzer
    if structural_analyzer is None:
        structural_analyzer = StructuralAnalyzer()
    return structural_analyzer

# Initialize the analyzer when module is imported
try:
    structural_analyzer = initialize_analyzer()
    logger.info("Structural analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize structural analyzer: {e}")
    structural_analyzer = None

if __name__ == "__main__":
    # Run test
    analyzer = test_analyzer()