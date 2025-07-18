import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    OPENAI = "openai"
    OLLAMA_MISTRAL = "ollama/mistral"
    RULE_BASED = "rule-based"

@dataclass
class AnalysisResult:
    analysis: str
    model_used: ModelType
    is_fallback: bool = False
    confidence: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None

class LLMFallback:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.primary = None
        self.fallback = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize services with proper error handling."""
        try:
            from services.analyzer import OpenAIAnalyzer
            self.primary = OpenAIAnalyzer()
            logger.info("OpenAI service initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import OpenAI service: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {e}")
        
        try:
            from services.ollama_client import OllamaClient
            self.fallback = OllamaClient()
            logger.info("Ollama service initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import Ollama service: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama service: {e}")

    async def assess_vulnerability(
        self, 
        prompt: str, 
        images: Optional[List] = None,
        opencv_analysis: Optional[Dict] = None
    ) -> AnalysisResult:
        """
        Multi-model fallback pipeline with enhanced error handling.
        
        Args:
            prompt: Text prompt for analysis
            images: List of images for analysis
            opencv_analysis: Pre-processed OpenCV analysis results
        
        Returns:
            AnalysisResult with analysis and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Enhance prompt with OpenCV analysis if available
        enhanced_prompt = self._enhance_prompt_with_opencv(prompt, opencv_analysis)
        
        # Try OpenAI first
        if self.primary:
            try:
                logger.info("Attempting OpenAI analysis...")
                response = await asyncio.wait_for(
                    self.primary.analyze(enhanced_prompt, images),
                    timeout=self.timeout_seconds
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                return AnalysisResult(
                    analysis=response.get("analysis", ""),
                    model_used=ModelType.OPENAI,
                    confidence=response.get("confidence", 0.8),
                    processing_time=processing_time
                )
                
            except asyncio.TimeoutError:
                logger.warning("OpenAI request timed out")
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
        
        # Fallback to Ollama + Mistral
        if self.fallback:
            try:
                logger.info("Falling back to Ollama/Mistral...")
                response = await asyncio.wait_for(
                    self.fallback.generate(enhanced_prompt, images),
                    timeout=self.timeout_seconds
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                return AnalysisResult(
                    analysis=response,
                    model_used=ModelType.OLLAMA_MISTRAL,
                    is_fallback=True,
                    confidence=0.6,
                    processing_time=processing_time
                )
                
            except asyncio.TimeoutError:
                logger.warning("Ollama request timed out")
            except Exception as e:
                logger.error(f"Ollama analysis failed: {e}")
        
        # Ultimate fallback to rule-based
        logger.info("Using rule-based fallback...")
        processing_time = asyncio.get_event_loop().time() - start_time
        return self._rule_based_fallback(enhanced_prompt, opencv_analysis, processing_time)

    def _enhance_prompt_with_opencv(self, prompt: str, opencv_analysis: Optional[Dict]) -> str:
        """Enhance prompt with OpenCV analysis results."""
        if not opencv_analysis:
            return prompt
        
        enhanced = f"{prompt}\n\nOpenCV Analysis Results:\n"
        
        # Add detected features
        if opencv_analysis.get("structural_damage"):
            enhanced += f"- Structural damage detected: {opencv_analysis['structural_damage']}\n"
        
        if opencv_analysis.get("roof_condition"):
            enhanced += f"- Roof condition: {opencv_analysis['roof_condition']}\n"
        
        if opencv_analysis.get("foundation_issues"):
            enhanced += f"- Foundation issues: {opencv_analysis['foundation_issues']}\n"
        
        if opencv_analysis.get("environmental_risks"):
            enhanced += f"- Environmental risks: {opencv_analysis['environmental_risks']}\n"
        
        return enhanced

    def _rule_based_fallback(self, prompt: str, opencv_analysis: Optional[Dict], processing_time: float) -> AnalysisResult:
        """Enhanced rule-based fallback with OpenCV integration."""
        keywords = prompt.lower()
        recommendations = []
        
        # Wind-related keywords
        if any(word in keywords for word in ["wind", "hurricane", "storm", "gust"]):
            recommendations.append("Reinforce roof structures and secure loose materials")
        
        # Flood-related keywords
        if any(word in keywords for word in ["flood", "water", "rain", "drainage"]):
            recommendations.append("Improve drainage systems and waterproofing")
        
        # Fire-related keywords
        if any(word in keywords for word in ["fire", "heat", "wildfire", "ignition"]):
            recommendations.append("Create defensible space and use fire-resistant materials")
        
        # Earthquake-related keywords
        if any(word in keywords for word in ["earthquake", "seismic", "shake", "tremor"]):
            recommendations.append("Strengthen structural connections and foundation")
        
        # Incorporate OpenCV analysis
        if opencv_analysis:
            if opencv_analysis.get("structural_damage"):
                recommendations.append("Address identified structural damage immediately")
            if opencv_analysis.get("roof_condition") == "poor":
                recommendations.append("Roof replacement or major repairs required")
            if opencv_analysis.get("foundation_issues"):
                recommendations.append("Foundation inspection and repairs needed")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Conduct comprehensive building inspection")
        
        analysis = "Rule-based Analysis:\n" + "\n".join(f"• {rec}" for rec in recommendations)
        
        return AnalysisResult(
            analysis=analysis,
            model_used=ModelType.RULE_BASED,
            is_fallback=True,
            confidence=0.3,
            processing_time=processing_time,
            error_message="Both AI services unavailable"
        )

    def get_service_status(self) -> Dict[str, bool]:
        """Check the status of all services."""
        return {
            "openai": self.primary is not None,
            "ollama": self.fallback is not None,
            "rule_based": True  # Always available
        }