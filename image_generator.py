"""
Image Generator Library
Library for generating/editing images using Google's Generative AI API
"""
import os
import base64
import re
import time
import logging
from io import BytesIO
from typing import Optional, Union
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class ImageGenerator:
    """Library for generating and editing images using Google Generative AI"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the ImageGenerator
        
        Args:
            api_key: Google API key. If None, will try to get from GOOGLE_API_KEY env var
            model_name: Model name to use. If None, will try to get from GOOGLE_MODEL env var or use default
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = model_name or os.getenv("GOOGLE_MODEL", "nano-banana-pro")
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        try:
            genai.configure(api_key=self.api_key)
            logger.info(f"✅ Google Generative AI configured successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to configure Google Generative AI: {e}")
            raise
    
    def generate_image(
        self, 
        image_data: bytes, 
        prompt: str,
        model_name: Optional[str] = None
    ) -> bytes:
        """
        Generate/Edit an image using Google Generative AI API
        
        Args:
            image_data: Input image as bytes
            prompt: Text prompt describing the desired edit/generation
            model_name: Optional model name override
            
        Returns:
            bytes: Generated/edited image as PNG bytes
            
        Raises:
            ValueError: If image_data or prompt is invalid
            RuntimeError: If API call fails
        """
        if not image_data:
            raise ValueError("image_data cannot be empty")
        
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")
        
        model_to_use = model_name or self.model_name
        
        try:
            # Initialize the model
            model = genai.GenerativeModel(model_to_use)
            
            # Create PIL Image from bytes for Google API
            image = Image.open(BytesIO(image_data))
            
            # Prepare the prompt with image editing instruction
            full_prompt = f"Edit this image according to the following instruction: {prompt}. Generate and return the edited image."
            
            logger.info(f"Sending request to Google Generative AI (model: {model_to_use})...")
            logger.info(f"Prompt: {prompt}")
            start_time = time.time()
            
            # Generate content with image and prompt
            response = model.generate_content([image, full_prompt])
            elapsed = time.time() - start_time
            logger.info(f"Received response from Google API in {elapsed:.2f} seconds")
            
            # Extract image from response
            generated_image_bytes = self._extract_image_from_response(response)
            
            if generated_image_bytes:
                logger.info(f"Successfully generated image ({len(generated_image_bytes)} bytes)")
                return generated_image_bytes
            else:
                raise RuntimeError("Could not extract image from API response")
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error calling Google Generative AI: {e}", exc_info=True)
            raise RuntimeError(f"Error from Google API: {str(e)}")
    
    def _extract_image_from_response(self, response) -> Optional[bytes]:
        """
        Extract image bytes from Google API response
        
        Args:
            response: Response object from Google Generative AI API
            
        Returns:
            bytes: Image bytes if found, None otherwise
        """
        # Check if response has candidates with content
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Check for inline data (base64 encoded image)
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data'):
                            image_bytes = base64.b64decode(part.inline_data.data)
                            return image_bytes
                    # Check for image object
                    if hasattr(part, 'image') and part.image:
                        image_buffer = BytesIO()
                        part.image.save(image_buffer, format='PNG')
                        image_buffer.seek(0)
                        return image_buffer.getvalue()
        
        # Check if response has parts directly
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data'):
                        image_bytes = base64.b64decode(part.inline_data.data)
                        return image_bytes
                if hasattr(part, 'image') and part.image:
                    image_buffer = BytesIO()
                    part.image.save(image_buffer, format='PNG')
                    image_buffer.seek(0)
                    return image_buffer.getvalue()
        
        # Check if response has images attribute
        if hasattr(response, 'images') and response.images:
            generated_image = response.images[0]
            image_buffer = BytesIO()
            if isinstance(generated_image, Image.Image):
                generated_image.save(image_buffer, format='PNG')
                image_buffer.seek(0)
                return image_buffer.getvalue()
            elif isinstance(generated_image, bytes):
                return generated_image
            else:
                logger.warning(f"Unexpected image type from response.images: {type(generated_image)}")
                return None
        
        # If response contains text, try to extract base64 image data
        if hasattr(response, 'text') and response.text:
            # Try to find base64 image data in the text
            base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', response.text)
            if base64_match:
                image_bytes = base64.b64decode(base64_match.group(1))
                return image_bytes
            # Log the text response for debugging
            logger.warning(f"API returned text instead of image: {response.text[:200]}")
        
        # If we can't find an image, log the full response structure for debugging
        logger.error(f"Could not extract image from response. Response type: {type(response)}, Attributes: {dir(response)}")
        return None


# Convenience function for easy usage
def generate_image(image_data: bytes, prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> bytes:
    """
    Convenience function to generate/edit an image
    
    Args:
        image_data: Input image as bytes
        prompt: Text prompt describing the desired edit/generation
        api_key: Optional Google API key (will use env var if not provided)
        model_name: Optional model name (will use env var or default if not provided)
        
    Returns:
        bytes: Generated/edited image as PNG bytes
    """
    generator = ImageGenerator(api_key=api_key, model_name=model_name)
    return generator.generate_image(image_data, prompt)

