import aiohttp
import asyncio
import logging
import base64
from typing import Optional, List, Dict, Any
import json
import os

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None, model: str = "mistral"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate(
        self, 
        prompt: str, 
        images: Optional[List] = None,
        stream: bool = False,
        options: Optional[Dict] = None
    ) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Text prompt
            images: List of image data (bytes or base64)
            stream: Whether to stream response
            options: Additional options for the model
        
        Returns:
            Generated text response
        """
        session = await self._get_session()
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {}
        }
        
        # Add images if provided
        if images:
            payload["images"] = await self._process_images(images)
        
        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                if stream:
                    return await self._handle_stream_response(response)
                else:
                    result = await response.json()
                    return result.get("response", "")
        
        except aiohttp.ClientError as e:
            logger.error(f"Ollama client error: {e}")
            raise Exception(f"Failed to connect to Ollama: {e}")
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            raise Exception("Ollama request timed out")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generate: {e}")
            raise
    
    async def _process_images(self, images: List) -> List[str]:
        """Process images to base64 format."""
        processed_images = []
        
        for image in images:
            if isinstance(image, bytes):
                # Convert bytes to base64
                b64_image = base64.b64encode(image).decode('utf-8')
                processed_images.append(b64_image)
            elif isinstance(image, str):
                # Assume it's already base64 or a file path
                if os.path.isfile(image):
                    with open(image, 'rb') as f:
                        b64_image = base64.b64encode(f.read()).decode('utf-8')
                        processed_images.append(b64_image)
                else:
                    processed_images.append(image)
            else:
                logger.warning(f"Unsupported image type: {type(image)}")
        
        return processed_images
    
    async def _handle_stream_response(self, response) -> str:
        """Handle streaming response from Ollama."""
        full_response = ""
        
        async for line in response.content:
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        full_response += chunk['response']
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return full_response
    
    async def check_health(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def pull_model(self, model_name: str = None) -> bool:
        """Pull a model to ensure it's available."""
        model_name = model_name or self.model
        
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            async with session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for model download
            ) as response:
                
                if response.status != 200:
                    logger.error(f"Failed to pull model {model_name}")
                    return False
                
                # Handle streaming response for model pull
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get('status') == 'success':
                                logger.info(f"Successfully pulled model {model_name}")
                                return True
                        except json.JSONDecodeError:
                            continue
                
                return True
        
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def list_models(self) -> List[Dict]:
        """List available models."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status != 200:
                    return []
                
                result = await response.json()
                return result.get('models', [])
        
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None