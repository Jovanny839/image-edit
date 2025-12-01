"""
Image Generation Library using Google AI Studio (Nano Banana Pro Model)

This library provides functions for generating images using the Google AI Studio API.
Supports both text-only prompts and text prompts with input images.
"""

import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def setup_api_key(api_key: str = None):
    """
    Setup the Google AI Studio API key.
    
    Args:
        api_key: Your API key from aistudio.google.com
                 If None, will try to get from environment variable GOOGLE_API_KEY
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        ValueError: If API key is not provided and not found in environment
    """
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Try to get from environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "API key not provided. Either pass it as argument or set GOOGLE_API_KEY environment variable."
            )
        genai.configure(api_key=api_key)
    
    logger.info("✓ API key configured successfully!")
    return True


def generate_image(
    prompt: str,
    model_name: str = "gemini-3-pro-image-preview",
    output_path: str = None,
    input_image: Image.Image = None,
    input_image_path: str = None
):
    """
    Generate an image using Google AI Studio.
    
    Args:
        prompt: Text description of the image to generate
        model_name: Name of the model to use (default: nano-banana-pro)
        output_path: Optional path where the generated image will be saved
        input_image: Optional PIL Image object to use with the prompt
        input_image_path: Optional path to an input image file to use with the prompt
    
    Returns:
        bytes: Image data as bytes if successful, None otherwise
    
    Raises:
        FileNotFoundError: If input_image_path is provided but file doesn't exist
        ValueError: If model configuration fails
        Exception: For other API errors
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        logger.info(f"Generating image with prompt: '{prompt[:100]}...' (truncated)" if len(prompt) > 100 else f"Generating image with prompt: '{prompt}'")
        logger.info(f"Using model: {model_name}")
        
        # Prepare content - either text only or text + image
        if input_image:
            # Use provided PIL Image object
            logger.info("Using provided input image (PIL Image)")
            response = model.generate_content([input_image, prompt])
        elif input_image_path:
            if not os.path.exists(input_image_path):
                raise FileNotFoundError(f"Input image file not found: {input_image_path}")
            
            logger.info(f"Loading input image: {input_image_path}")
            # Load the input image
            input_image = Image.open(input_image_path)
            
            # Generate with both image and text prompt
            response = model.generate_content([input_image, prompt])
        else:
            # Generate with text prompt only
            response = model.generate_content(prompt)
        
        # Extract image data from response
        image_bytes = None
        
        # Check if response has candidates with content
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Check for inline data (base64 encoded image)
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data'):
                            import base64
                            image_bytes = base64.b64decode(part.inline_data.data)
                            break
                    # Check for image object
                    if hasattr(part, 'image') and part.image:
                        image_buffer = BytesIO()
                        part.image.save(image_buffer, format='PNG')
                        image_buffer.seek(0)
                        image_bytes = image_buffer.getvalue()
                        break
        
        # Check if response has parts directly
        if not image_bytes and hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data'):
                        import base64
                        image_bytes = base64.b64decode(part.inline_data.data)
                        break
                if hasattr(part, 'image') and part.image:
                    image_buffer = BytesIO()
                    part.image.save(image_buffer, format='PNG')
                    image_buffer.seek(0)
                    image_bytes = image_buffer.getvalue()
                    break
        
        # Check if response has images attribute
        if not image_bytes and hasattr(response, 'images') and response.images:
            generated_image = response.images[0]
            if isinstance(generated_image, Image.Image):
                image_buffer = BytesIO()
                generated_image.save(image_buffer, format='PNG')
                image_buffer.seek(0)
                image_bytes = image_buffer.getvalue()
            elif isinstance(generated_image, bytes):
                image_bytes = generated_image
        
        if not image_bytes:
            logger.error("Could not extract image from response")
            logger.debug(f"Response type: {type(response)}, Attributes: {dir(response)}")
            raise ValueError("Unexpected response format. Could not extract image from response.")
        
        # Save to file if output_path is provided
        if output_path:
            with open(output_path, 'wb') as image_file:
                image_file.write(image_bytes)
            logger.info(f"✓ Image successfully saved to: {output_path}")
        
        logger.info("✓ Image successfully generated")
        return image_bytes
            
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise Exception(f"Error generating image: {str(e)}")


def list_available_models():
    """
    List all available models in your Google AI Studio account.
    
    Returns:
        list: List of available model names, or None if error occurs
    """
    try:
        models = genai.list_models()
        logger.info("="*60)
        logger.info("Available models for image generation:")
        logger.info("="*60)
        image_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                logger.info(f"  - {model.name}")
                image_models.append(model.name)
        logger.info("="*60)
        return image_models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return None

