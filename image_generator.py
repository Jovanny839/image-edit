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
        self.model_name = model_name or os.getenv("GOOGLE_MODEL", "gemini-3-pro-image-preview")
        
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
            # Be explicit that we want an image output, not text
            full_prompt = f"""Edit this image according to the following instruction: {prompt}. 
            
IMPORTANT: You must generate and return ONLY an image file. Do not provide any text description or explanation. Return the edited image directly."""
            
            logger.info(f"Sending request to Google Generative AI (model: {model_to_use})...")
            logger.info(f"Prompt: {prompt}")
            start_time = time.time()
            
            # Generate content with image and prompt
            # Try different methods to configure response_modalities
            response = None
            error_messages = []
            
            # Method 1: Try with GenerationConfig using types
            try:
                from google.generativeai import types
                generation_config = types.GenerationConfig(
                    response_modalities=['Image']
                )
                response = model.generate_content(
                    [image, full_prompt],
                    generation_config=generation_config
                )
                logger.info("Successfully used GenerationConfig with response_modalities")
            except Exception as e1:
                error_messages.append(f"Method 1 (GenerationConfig): {str(e1)}")
                
                # Method 2: Try with dict as generation_config
                try:
                    response = model.generate_content(
                        [image, full_prompt],
                        generation_config={'response_modalities': ['Image']}
                    )
                    logger.info("Successfully used dict generation_config with response_modalities")
                except Exception as e2:
                    error_messages.append(f"Method 2 (dict config): {str(e2)}")
                    
                    # Method 3: Try with config parameter
                    try:
                        response = model.generate_content(
                            [image, full_prompt],
                            config={'response_modalities': ['Image']}
                        )
                        logger.info("Successfully used config parameter with response_modalities")
                    except Exception as e3:
                        error_messages.append(f"Method 3 (config param): {str(e3)}")
                        
                        # Method 4: Try without config (fallback)
                        try:
                            logger.warning("response_modalities configuration not supported, trying without config")
                            logger.warning("The API may return text instead of images")
                            response = model.generate_content([image, full_prompt])
                        except Exception as e4:
                            error_messages.append(f"Method 4 (no config): {str(e4)}")
                            raise RuntimeError(f"All methods failed to generate content: {'; '.join(error_messages)}")
            
            if response is None:
                raise RuntimeError(f"Failed to get response from API: {'; '.join(error_messages)}")
            
            elapsed = time.time() - start_time
            logger.info(f"Received response from Google API in {elapsed:.2f} seconds")
            
            # Log response structure for debugging
            logger.debug(f"Response type: {type(response)}")
            if hasattr(response, 'text'):
                logger.debug(f"Response text (first 500 chars): {response.text[:500] if response.text else 'None'}")
            
            # Extract image from response
            generated_image_bytes = self._extract_image_from_response(response)
            
            if generated_image_bytes:
                logger.info(f"Successfully generated image ({len(generated_image_bytes)} bytes)")
                return generated_image_bytes
            else:
                # Log more details about the response structure
                logger.error(f"Could not extract image. Response structure: {dir(response)}")
                if hasattr(response, 'text') and response.text:
                    logger.error(f"Response text: {response.text[:1000]}")
                raise RuntimeError("Could not extract image from API response. The API may have returned text instead of an image.")
                
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
        # First, check if response has candidates with content (most common structure)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            logger.debug(f"Candidate structure: {dir(candidate)}")
            
            if hasattr(candidate, 'content') and candidate.content:
                logger.debug(f"Content structure: {dir(candidate.content)}")
                
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    logger.debug(f"Number of parts: {len(candidate.content.parts)}")
                    for i, part in enumerate(candidate.content.parts):
                        logger.debug(f"Part {i} type: {type(part)}, attributes: {dir(part)}")
                        
                        # Check for inline data (base64 encoded image)
                        if hasattr(part, 'inline_data') and part.inline_data:
                            logger.debug(f"Found inline_data in part {i}")
                            if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                try:
                                    image_bytes = base64.b64decode(part.inline_data.data)
                                    logger.info(f"Successfully decoded base64 image from inline_data ({len(image_bytes)} bytes)")
                                    return image_bytes
                                except Exception as e:
                                    logger.error(f"Error decoding base64 data: {e}")
                        
                        # Check for image object (PIL Image)
                        if hasattr(part, 'image') and part.image:
                            logger.debug(f"Found image object in part {i}")
                            try:
                                image_buffer = BytesIO()
                                if isinstance(part.image, Image.Image):
                                    part.image.save(image_buffer, format='PNG')
                                else:
                                    # Try to convert to PIL Image
                                    img = Image.open(BytesIO(part.image))
                                    img.save(image_buffer, format='PNG')
                                image_buffer.seek(0)
                                image_bytes = image_buffer.getvalue()
                                logger.info(f"Successfully extracted image from image object ({len(image_bytes)} bytes)")
                                return image_bytes
                            except Exception as e:
                                logger.error(f"Error extracting image from image object: {e}")
                        
                        # Check for mime_type to identify image parts
                        if hasattr(part, 'mime_type') and part.mime_type and part.mime_type.startswith('image/'):
                            logger.debug(f"Found image mime_type: {part.mime_type} in part {i}")
                            if hasattr(part, 'data'):
                                try:
                                    image_bytes = base64.b64decode(part.data)
                                    logger.info(f"Successfully decoded image from data field ({len(image_bytes)} bytes)")
                                    return image_bytes
                                except Exception as e:
                                    logger.error(f"Error decoding image data: {e}")
        
        # Check if response has parts directly (alternative structure)
        if hasattr(response, 'parts') and response.parts:
            logger.debug(f"Response has parts directly: {len(response.parts)}")
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                        try:
                            image_bytes = base64.b64decode(part.inline_data.data)
                            logger.info(f"Successfully decoded base64 image from parts ({len(image_bytes)} bytes)")
                            return image_bytes
                        except Exception as e:
                            logger.error(f"Error decoding base64 data from parts: {e}")
                if hasattr(part, 'image') and part.image:
                    try:
                        image_buffer = BytesIO()
                        part.image.save(image_buffer, format='PNG')
                        image_buffer.seek(0)
                        image_bytes = image_buffer.getvalue()
                        logger.info(f"Successfully extracted image from parts ({len(image_bytes)} bytes)")
                        return image_bytes
                    except Exception as e:
                        logger.error(f"Error extracting image from parts: {e}")
        
        # Check if response has images attribute
        if hasattr(response, 'images') and response.images:
            logger.debug(f"Response has images attribute: {len(response.images)}")
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
            logger.debug("Response contains text, attempting to extract base64 image")
            # Try to find base64 image data in the text
            base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', response.text)
            if base64_match:
                try:
                    image_bytes = base64.b64decode(base64_match.group(1))
                    logger.info(f"Successfully extracted base64 image from text ({len(image_bytes)} bytes)")
                    return image_bytes
                except Exception as e:
                    logger.error(f"Error decoding base64 from text: {e}")
            
            # Log the text response for debugging
            logger.warning(f"API returned text instead of image. Text preview: {response.text[:500]}")
            logger.warning("This might indicate that the model is not configured to return images, or the prompt needs adjustment.")
        
        # If we can't find an image, log the full response structure for debugging
        logger.error(f"Could not extract image from response. Response type: {type(response)}")
        logger.error(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
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

