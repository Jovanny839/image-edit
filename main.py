from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import requests
import base64
import time
import uvicorn
import json
import re
from io import BytesIO
from fastapi.responses import StreamingResponse
import logging
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image as PILImage
from google import genai
from google.genai import types
from google.genai.types import Image as GeminiImage
from story_lib import generate_story
from typing import List, Optional, Dict, Any
from queue_manager import QueueManager
from batch_processor import BatchProcessor
from validation_utils import ConsistencyValidationResult
import asyncio
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gemini-3-pro-image-preview"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"  # Model for text generation (scenes)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for storage operations
STORAGE_BUCKET = "images"

# Initialize Gemini client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY not found. Image generation will be disabled.")

supabase: Client = None
if SUPABASE_URL:
    # Try service key first (bypasses RLS), then anon key
    key_to_use = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
    key_type = "service" if SUPABASE_SERVICE_KEY else "anon"
    
    if key_to_use:
        try:
            supabase = create_client(SUPABASE_URL, key_to_use)
            logger.info(f"✅ Supabase client initialized successfully using {key_type} key")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
    else:
        logger.warning("⚠️ No Supabase key found (SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY)")
else:
    logger.warning("⚠️ Supabase URL not found. Storage upload will be disabled.")

# Initialize queue manager and batch processor
queue_manager = None
batch_processor = None
worker_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for background tasks"""
    global queue_manager, batch_processor, worker_task
    
    # Initialize queue manager
    if supabase:
        queue_manager = QueueManager(supabase)
        batch_processor = BatchProcessor(
            queue_manager=queue_manager,
            gemini_client=gemini_client,
            openai_api_key=OPENAI_API_KEY,
            supabase_client=supabase,
            gemini_text_model=GEMINI_TEXT_MODEL
        )
        logger.info("✅ Queue manager and batch processor initialized")
        
        # Start background worker
        worker_task = asyncio.create_task(background_worker())
        logger.info("✅ Background worker started")
    
    yield
    
    # Cleanup
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("✅ Background worker stopped")

# FastAPI app
app = FastAPI(
    title="AI Image Editor API",
    description="API for editing images using Google Gemini's image generation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add trusted host middleware (helps prevent invalid requests)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, specify actual domains
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler for better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Handle validation errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format or data"}
    )

# Request model to receive input data
class ImageRequest(BaseModel):
    image_url: HttpUrl  # This validates the URL format
    prompt: str
    
    class Config:
        # Example values for API documentation
        schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "prompt": "Make this image more colorful and vibrant"
            }
        }

# Response model for image editing
class ImageResponse(BaseModel):
    success: bool
    message: str
    storage_info: dict = None
    quality_validation: Optional[Dict[str, Any]] = None

# Response model for quality validation
class QualityValidationResponse(BaseModel):
    success: bool
    validation: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "validation": {
                    "is_valid": True,
                    "quality_score": 0.85,
                    "is_appropriate": True,
                    "is_clear": True,
                    "has_sufficient_detail": True,
                    "issues": [],
                    "recommendations": ["Image quality is good"],
                    "details": {
                        "image_properties": {
                            "actual_resolution": "1024x768",
                            "format": "JPEG",
                            "clarity": "high"
                        }
                    }
                }
            }
        }
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Image edited and uploaded successfully",
                "storage_info": {
                    "uploaded": True,
                    "url": "https://your-project.supabase.co/storage/v1/object/public/images/edited_image_123.jpg",
                    "filename": "edited_image_123.jpg",
                    "bucket": "images"
                }
            }
        }

# Request model for story generation
class StoryRequest(BaseModel):
    character_name: str
    character_type: str
    special_ability: str
    age_group: str  # Must be "3-6", "7-10", or "11-12"
    story_world: str
    adventure_type: str
    occasion_theme: Optional[str] = None
    character_image_url: Optional[HttpUrl] = None  # Supabase URL of the character reference image
    story_text_prompt: Optional[str] = None  # Full prompt for story text generation (from frontend)
    scene_prompts: Optional[List[str]] = None  # List of 5 scene prompts, one for each page (from frontend)
    reading_level: Optional[str] = None  # Reading level (early_reader / developing_reader / independent_reader)
    story_title: Optional[str] = None  # Story title
    
    class Config:
        schema_extra = {
            "example": {
                "character_name": "Luna",
                "character_type": "a brave dragon",
                "special_ability": "fly through clouds",
                "age_group": "7-10",
                "story_world": "the Enchanted Forest",
                "adventure_type": "treasure hunt",
                "occasion_theme": None,
                "character_image_url": "https://your-project.supabase.co/storage/v1/object/public/images/character_reference.jpg",
                "story_text_prompt": "Create a personalized 5-page children's storybook...",
                "scene_prompts": ["Scene prompt for page 1...", "Scene prompt for page 2...", ...],
                "reading_level": "developing_reader",
                "story_title": "The Great Adventure of Luna"
            }
        }

# Page model for story pages with text and scene image
class StoryPage(BaseModel):
    text: str
    scene: Optional[HttpUrl] = None  # URL to the generated scene image
    consistency_validation: Optional[ConsistencyValidationResult] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Meet Luna, a brave dragon who loves adventures. Luna has a special power: Luna can fly through clouds.",
                "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page1_20240101_120000_abc123.jpg",
                "consistency_validation": {
                    "is_consistent": True,
                    "similarity_score": 0.85,
                    "validation_time_seconds": 3.2,
                    "flagged": False
                }
            }
        }

# Response model for story generation
class StoryResponse(BaseModel):
    success: bool
    pages: List[StoryPage]
    full_story: str
    word_count: int
    page_word_counts: List[int]
    consistency_summary: Optional[Dict[str, Any]] = None  # Overall validation summary
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "pages": [
                    {
                        "text": "Meet Luna, a brave dragon who loves adventures. Luna has a special power: Luna can fly through clouds.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page1_20240101_120000_abc123.jpg"
                    },
                    {
                        "text": "While exploring, Luna discovered a magical entrance that led to the Enchanted Forest.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page2_20240101_120001_def456.jpg"
                    },
                    {
                        "text": "Suddenly, Luna realized that a treasure hunt was beginning, and Luna was right in the middle of it.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page3_20240101_120002_ghi789.jpg"
                    },
                    {
                        "text": "When the moment of truth arrived, Luna faced the challenge head-on, even though it seemed impossible at first.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page4_20240101_120003_jkl012.jpg"
                    },
                    {
                        "text": "The adventure came to a wonderful conclusion, and Luna felt proud of what had been accomplished.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page5_20240101_120004_mno345.jpg"
                    }
                ],
                "full_story": "Meet Luna, a brave dragon who loves adventures...",
                "word_count": 250,
                "page_word_counts": [20, 25, 30, 28, 27]
            }
        }

def get_content_type_from_url(url):
    """Determine content type based on URL extension"""
    url_lower = url.lower()
    if url_lower.endswith(('.png', '.PNG')):
        return "image/png"
    elif url_lower.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        return "image/jpeg"
    elif url_lower.endswith(('.gif', '.GIF')):
        return "image/gif"
    elif url_lower.endswith(('.webp', '.WEBP')):
        return "image/webp"
    else:
        return "image/jpeg"  # default fallback

def detect_image_mime_type(image_data: bytes) -> str:
    """Detect MIME type from image bytes using PIL"""
    try:
        image = PILImage.open(BytesIO(image_data))
        format_to_mime = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'GIF': 'image/gif',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
            'TIFF': 'image/tiff'
        }
        return format_to_mime.get(image.format, 'image/jpeg')
    except Exception as e:
        logger.warning(f"Could not detect image format, defaulting to image/jpeg: {e}")
        return "image/jpeg"

def download_image_from_url(url):
    """Download image from URL and return image data"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL {url}: {e}")

def optimize_image_to_jpg(image_data: bytes, quality: int = 85) -> bytes:
    """Convert and optimize image to JPG format with compression while preserving original resolution"""
    try:
        # Open image from bytes
        image = PILImage.open(BytesIO(image_data))
        original_size_info = f"{image.width}x{image.height}"
        
        # Convert to RGB if necessary (PNG with transparency, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = PILImage.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG with compression (keeping original resolution)
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
        optimized_data = output_buffer.getvalue()
        
        # Log compression results
        original_size = len(image_data)
        optimized_size = len(optimized_data)
        compression_ratio = (1 - optimized_size / original_size) * 100
        logger.info(f"Image optimized ({original_size_info}): {original_size:,} bytes → {optimized_size:,} bytes ({compression_ratio:.1f}% reduction)")
        
        return optimized_data
        
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        # Return original data if optimization fails
        return image_data

def upload_to_supabase(image_data: bytes, filename: str) -> dict:
    """Upload image to Supabase storage and return the public URL"""
    if not supabase:
        logger.warning("Supabase client not available, skipping upload")
        return {"uploaded": False, "url": None, "message": "Supabase not configured"}

    try:
        logger.info(f"Uploading {filename} to Supabase storage bucket '{STORAGE_BUCKET}'")

        # Pass image_data directly as bytes to Supabase storage

        response = supabase.storage.from_(STORAGE_BUCKET).upload(filename, image_data, {
            'content-type' : 'image/jpeg',
            'upsert' : 'true'
        })

        # Check response type - response is an UploadResponse object
        if hasattr(response, 'full_path') and response.full_path:
            public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(filename)
            logger.info(f"✅ Successfully uploaded to Supabase: {public_url}")

            return {
                "uploaded": True,
                "url": public_url,
                "filename": filename,
                "bucket": STORAGE_BUCKET,
                "message": "Successfully uploaded to Supabase storage"
            }

        logger.error(f"❌ Unexpected Supabase response: {response}")
        return {"uploaded": False, "url": None, "message": f"Unexpected response: {response}"}

    except Exception as e:
        logger.error(f"❌ Error uploading to Supabase: {e}")
        return {"uploaded": False, "url": None, "message": f"Upload error: {e}"}

def edit_image(image_data, prompt, image_url=None):
    """Send image to Gemini API for editing/generation"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Please check GEMINI_API_KEY.")
    
    logger.info(f"Sending request to Gemini API with model: {MODEL}")
    
    try:
        start_time = time.time()
        
        # Detect MIME type from image data
        mime_type = detect_image_mime_type(image_data)
        logger.info(f"Detected image MIME type: {mime_type}")
        
        # Encode image to base64 for the dictionary format
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Generate content with Gemini API using the expected dictionary format
        # The API expects contents to be a list with role and parts
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Gemini API response received in {elapsed:.2f} seconds")
        
        # Extract image from response
        # Prioritize inline_data as it's the most direct source of image bytes
        edited_image_bytes = None
        for part in response.parts:
            if part.text is not None:
                logger.info(f"Gemini text response: {part.text}")
            
            # Check inline_data first - this is the most reliable source
            if hasattr(part, 'inline_data'):
                try:
                    inline_data = part.inline_data
                    logger.info(f"Found inline_data, type: {type(inline_data)}")
                    
                    # Try to get data from inline_data
                    if inline_data and hasattr(inline_data, 'data'):
                        data = inline_data.data
                        if isinstance(data, bytes):
                            edited_image_bytes = data
                            logger.info(f"✅ Image extracted from inline_data.data (bytes) ({len(edited_image_bytes)} bytes)")
                        elif isinstance(data, str):
                            # Try to decode base64
                            try:
                                edited_image_bytes = base64.b64decode(data)
                                logger.info(f"✅ Image extracted from inline_data.data (base64) ({len(edited_image_bytes)} bytes)")
                            except Exception as e:
                                logger.warning(f"Failed to decode base64 data: {e}")
                                # If it's not base64, try encoding as latin-1 (unlikely but possible)
                                edited_image_bytes = data.encode('latin-1')
                                logger.info(f"✅ Image extracted from inline_data.data (string) ({len(edited_image_bytes)} bytes)")
                    elif inline_data and hasattr(inline_data, 'bytes'):
                        edited_image_bytes = inline_data.bytes
                        logger.info(f"✅ Image extracted from inline_data.bytes ({len(edited_image_bytes)} bytes)")
                    
                    # Validate the extracted data
                    if edited_image_bytes and len(edited_image_bytes) > 1000:
                        logger.info(f"✅ Valid image extracted from inline_data ({len(edited_image_bytes)} bytes)")
                        break
                    elif edited_image_bytes:
                        logger.warning(f"Extracted data from inline_data is suspiciously small ({len(edited_image_bytes)} bytes), trying other methods...")
                        edited_image_bytes = None  # Reset to try other methods
                    else:
                        logger.warning(f"inline_data exists but no valid data found. inline_data attributes: {[a for a in dir(inline_data) if not a.startswith('_')]}")
                except Exception as e:
                    logger.warning(f"Error extracting from inline_data: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to as_image() if inline_data didn't work
            if not edited_image_bytes and hasattr(part, 'as_image'):
                try:
                    gemini_image = part.as_image()
                    logger.info(f"Got Gemini Image object: {type(gemini_image)}")
                    
                    # Check if it's already a PIL Image
                    if isinstance(gemini_image, PILImage.Image):
                        img_buffer = BytesIO()
                        gemini_image.save(img_buffer, format='PNG')
                        edited_image_bytes = img_buffer.getvalue()
                        logger.info(f"✅ Image extracted from PIL Image ({len(edited_image_bytes)} bytes)")
                        break
                    # Try to get bytes from Gemini Image object
                    elif hasattr(gemini_image, 'to_bytes'):
                        edited_image_bytes = gemini_image.to_bytes()
                    elif hasattr(gemini_image, 'bytes'):
                        edited_image_bytes = gemini_image.bytes
                    elif hasattr(gemini_image, 'data'):
                        data = gemini_image.data
                        if isinstance(data, bytes):
                            edited_image_bytes = data
                        elif isinstance(data, str):
                            edited_image_bytes = base64.b64decode(data)
                    else:
                        # Log available attributes for debugging
                        attrs = [a for a in dir(gemini_image) if not a.startswith('_')]
                        logger.warning(f"Gemini Image object attributes: {attrs}")
                        # Try accessing mime_type and data if they exist
                        if hasattr(gemini_image, 'mime_type') and hasattr(gemini_image, 'data'):
                            if isinstance(gemini_image.data, bytes):
                                edited_image_bytes = gemini_image.data
                            elif isinstance(gemini_image.data, str):
                                edited_image_bytes = base64.b64decode(gemini_image.data)
                    
                    # Validate size before accepting
                    if edited_image_bytes and len(edited_image_bytes) > 1000:
                        logger.info(f"✅ Image extracted from as_image() ({len(edited_image_bytes)} bytes)")
                        break
                    elif edited_image_bytes:
                        logger.warning(f"Extracted data from as_image() too small ({len(edited_image_bytes)} bytes), trying other methods...")
                        edited_image_bytes = None  # Reset to try other methods
                except Exception as e:
                    logger.warning(f"Error extracting from as_image(): {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if not edited_image_bytes:
            # Log more details for debugging
            logger.error(f"No valid image found in response. Response has {len(response.parts)} parts")
            for i, part in enumerate(response.parts):
                part_type = type(part).__name__
                attrs = [a for a in dir(part) if not a.startswith('_')]
                logger.error(f"Part {i}: type={part_type}, attributes={attrs}")
                # Try to log more details about each part
                if hasattr(part, 'inline_data'):
                    logger.error(f"  Part {i} inline_data: {part.inline_data}")
                if hasattr(part, 'text'):
                    logger.error(f"  Part {i} text: {part.text}")
            raise HTTPException(status_code=500, detail="No valid image was generated in the response from Gemini API")
        
        # Validate that we have a valid image before returning
        try:
            test_image = PILImage.open(BytesIO(edited_image_bytes))
            logger.info(f"✅ Validated image: {test_image.size[0]}x{test_image.size[1]}, format: {test_image.format}")
        except Exception as e:
            logger.error(f"Extracted data is not a valid image: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid image data extracted from Gemini API response: {str(e)}")
        
        return edited_image_bytes
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error from Gemini API: {str(e)}")

def validate_character_consistency(scene_image_data: bytes, reference_image_data: bytes, page_number: int, timeout_seconds: int = 15) -> ConsistencyValidationResult:
    """Wrapper for validation_utils.validate_character_consistency"""
    from validation_utils import validate_character_consistency as _validate_character_consistency
    return _validate_character_consistency(
        scene_image_data=scene_image_data,
        reference_image_data=reference_image_data,
        page_number=page_number,
        gemini_client=gemini_client,
        gemini_text_model=GEMINI_TEXT_MODEL,
        timeout_seconds=timeout_seconds
    )


def validate_image_quality(image_data: bytes, image_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate image quality using Gemini Vision API.
    Checks for: image quality, appropriateness, clarity, and basic properties.
    
    Returns a dictionary with validation results including:
    - is_valid: bool
    - quality_score: float (0-1)
    - issues: List[str]
    - recommendations: List[str]
    - details: Dict with specific checks
    """
    if not gemini_client:
        logger.warning("Gemini client not available for quality validation")
        return {
            "is_valid": True,  # Default to valid if validation unavailable
            "quality_score": 0.5,
            "issues": [],
            "recommendations": ["Quality validation unavailable - Gemini client not initialized"],
            "details": {"validation_available": False}
        }
    
    try:
        logger.info("Starting image quality validation with Gemini Vision API")
        
        # Detect MIME type
        mime_type = detect_image_mime_type(image_data)
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create validation prompt
        validation_prompt = """Analyze this image and provide a quality assessment in the following JSON format:
{
  "quality_score": <float 0.0-1.0>,
  "is_appropriate": <boolean>,
  "is_clear": <boolean>,
  "has_sufficient_detail": <boolean>,
  "issues": [<array of issue strings>],
  "recommendations": [<array of recommendation strings>],
  "image_properties": {
    "estimated_resolution": "<width>x<height>",
    "clarity": "<low/medium/high>",
    "brightness": "<too_dark/normal/too_bright>",
    "composition": "<poor/fair/good/excellent>"
  }
}

Focus on:
1. Image clarity and sharpness
2. Appropriate content for children (no violence, adult content, etc.)
3. Sufficient detail and resolution
4. Overall visual quality
5. Any technical issues (blur, distortion, artifacts)

Be strict but fair. Return ONLY valid JSON, no additional text."""
        
        # Call Gemini API for validation
        response = gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,  # Use text model for analysis
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": validation_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT'],
                temperature=0.1  # Lower temperature for more consistent validation
            )
        )
        
        # Extract text response
        validation_text = ""
        for part in response.parts:
            if part.text:
                validation_text += part.text
        
        # Parse JSON response
        # Try to extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
        if json_match:
            validation_json = json.loads(json_match.group())
        else:
            # Try parsing the whole response
            validation_json = json.loads(validation_text)
        
        # Extract validation results
        quality_score = validation_json.get("quality_score", 0.5)
        is_appropriate = validation_json.get("is_appropriate", True)
        is_clear = validation_json.get("is_clear", True)
        has_sufficient_detail = validation_json.get("has_sufficient_detail", True)
        issues = validation_json.get("issues", [])
        recommendations = validation_json.get("recommendations", [])
        image_properties = validation_json.get("image_properties", {})
        
        # Determine overall validity
        # Image is valid if: appropriate, clear, and quality score > 0.5
        is_valid = (
            is_appropriate and 
            is_clear and 
            quality_score >= 0.5 and
            has_sufficient_detail
        )
        
        # Add basic image properties from PIL
        try:
            pil_image = PILImage.open(BytesIO(image_data))
            image_properties["actual_resolution"] = f"{pil_image.width}x{pil_image.height}"
            image_properties["format"] = pil_image.format or "unknown"
            image_properties["mode"] = pil_image.mode
            image_properties["file_size_bytes"] = len(image_data)
        except Exception as e:
            logger.warning(f"Could not extract PIL image properties: {e}")
        
        result = {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "is_appropriate": is_appropriate,
            "is_clear": is_clear,
            "has_sufficient_detail": has_sufficient_detail,
            "issues": issues,
            "recommendations": recommendations,
            "details": {
                "image_properties": image_properties,
                "validation_available": True,
                "model_used": GEMINI_TEXT_MODEL
            }
        }
        
        logger.info(f"Quality validation completed: valid={is_valid}, score={quality_score:.2f}")
        if issues:
            logger.info(f"Validation issues found: {', '.join(issues)}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse validation JSON response: {e}")
        logger.error(f"Response text: {validation_text[:500] if 'validation_text' in locals() else 'N/A'}")
        return {
            "is_valid": True,  # Default to valid on parse error
            "quality_score": 0.5,
            "issues": ["Could not parse validation response"],
            "recommendations": ["Validation service error - proceeding with caution"],
            "details": {"validation_available": False, "error": "JSON parse error"}
        }
    except Exception as e:
        logger.error(f"Error during quality validation: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {
            "is_valid": True,  # Default to valid on error
            "quality_score": 0.5,
            "issues": [f"Validation error: {str(e)}"],
            "recommendations": ["Validation service error - proceeding with caution"],
            "details": {"validation_available": False, "error": str(e)}
        }

def create_blank_base_image(width: int = 768, height: int = 512) -> bytes:
    """Create a blank white image in 768x512 dimensions to use as base for image generation"""
    try:
        # Create a white image in 768x512 dimensions (default)
        blank_image = PILImage.new('RGB', (width, height), color=(255, 255, 255))
        img_buffer = BytesIO()
        blank_image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error creating blank base image: {e}")
        raise

def get_environment_details(story_world: str) -> str:
    """Get environment-specific details based on story world."""
    world_lower = story_world.lower()
    if 'enchanted forest' in world_lower or world_lower == 'forest':
        return "ENVIRONMENT DETAILS: Include magical trees with glowing elements, mystical flora, enchanted atmosphere with soft magical light, fairy-tale forest setting with whimsical details."
    elif 'outer space' in world_lower or world_lower == 'space':
        return "ENVIRONMENT DETAILS: Include planets, stars, alien landscapes, cosmic scenery, space nebulas, celestial bodies, and otherworldly terrain."
    elif 'underwater kingdom' in world_lower or world_lower == 'underwater':
        return "ENVIRONMENT DETAILS: Include coral reefs, sea creatures, underwater flora, aquatic plants, marine life, and oceanic elements."
    else:
        return "ENVIRONMENT DETAILS: Match the setting and atmosphere of the story world."

def generate_story_scene_image(story_page_text: str, page_number: int, character_name: str, character_type: str, story_world: str, reference_image_url: Optional[str] = None, scene_prompt: Optional[str] = None) -> str:
    """Generate a scene image for a story page using edit_image function and return the image URL.
    
    If scene_prompt is provided, use it; otherwise generate prompt from parameters.
    """
    if not gemini_client:
        logger.warning("Gemini client not available, returning empty scene URL")
        return ""
    
    logger.info(f"Generating scene image for page {page_number} using edit_image function")
    if reference_image_url:
        logger.info(f"Using reference character image from: {reference_image_url}")
    
    try:
        # Get base image - use reference image if provided, otherwise create a blank image
        base_image_data = None
        if reference_image_url:
            try:
                logger.info(f"Downloading reference image from: {reference_image_url}")
                base_image_data = download_image_from_url(reference_image_url)
                logger.info(f"✅ Reference image downloaded successfully ({len(base_image_data)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to download reference image, creating blank base image: {e}")
                base_image_data = None
        
        # If no reference image, create a blank white image in 768x512 dimensions
        if not base_image_data:
            logger.info("Creating blank base image for scene generation")
            base_image_data = create_blank_base_image()
            logger.info(f"✅ Blank base image created ({len(base_image_data)} bytes)")
        
        # Use provided prompt if available, otherwise generate one (for backward compatibility)
        if scene_prompt:
            prompt = scene_prompt
            logger.info(f"Using scene prompt from frontend for page {page_number}")
        else:
            # Fallback: generate prompt from parameters (for backward compatibility)
            character_reference_note = ""
            character_consistency_enforcement = ""
            negative_prompts = ""
            
            if reference_image_url and base_image_data:
                character_reference_note = f"""
CHARACTER REFERENCE:
- A reference image of {character_name} is provided below
- Use this reference image to maintain consistent character appearance across all scenes
- The character in the scene must match the appearance, style, and features shown in the reference image
- Keep the character's visual identity consistent with the reference image
"""
                character_consistency_enforcement = f"""
=== MANDATORY CHARACTER STYLE CONSISTENCY REQUIREMENTS ===
CRITICAL: The character from the provided reference image MUST be embedded with EXACT visual fidelity.

REQUIRED CHARACTER FEATURES (DO NOT CHANGE):
* Face: Exact same facial features, eye shape, nose, mouth, and expression style as reference
* Limbs: Exact same proportions, length, and structure as reference
* Body proportions: Exact same height-to-width ratio and body shape as reference
* Hair: Exact same hair style, color, texture, and length as reference
* Skin tone: Exact same skin color and tone as reference
* Clothing: Exact same clothing design, colors, patterns, and details as reference
* Overall design: Exact same character design language, style, and visual identity as reference
* Anatomy: Exact same anatomical structure - no changes to bone structure, muscle definition, or body type
* Style: The character's artistic style must remain consistent with the reference image

STRICT PROHIBITIONS:
* DO NOT alter the character's facial features
* DO NOT change the character's body proportions or anatomy
* DO NOT modify the character's hair style, color, or texture
* DO NOT change the character's skin tone or color
* DO NOT alter the character's clothing design, colors, or patterns
* DO NOT modify the character's overall design or visual identity
* DO NOT apply different artistic styles to the character than what appears in the reference
* DO NOT distort, stretch, or resize the character in ways that change their appearance
* DO NOT add features not present in the reference image
* DO NOT remove features present in the reference image

ENFORCEMENT:
The character must be reproduced with pixel-perfect fidelity to the reference image. Any deviation from the reference character's appearance is strictly prohibited. The scene style may vary, but the character's appearance must remain identical to the reference image in all aspects.
"""
                negative_prompts = """
=== NEGATIVE PROMPTS (STRICTLY AVOID) ===
DO NOT:
* Alter the character's facial features, proportions, or anatomy
* Change the character's hair style, color, or texture
* Modify the character's skin tone or color
* Alter the character's clothing design, colors, or patterns
* Change the character's body proportions or structure
* Apply different artistic styles to the character than the reference
* Distort, stretch, or resize the character in ways that change appearance
* Add features not present in the reference image
* Remove features present in the reference image
* Create variations of the character - use the exact reference character only
"""
            
            environment_details = get_environment_details(story_world)
            
            prompt = f"""Create a beautiful, colorful children's storybook illustration for this story page.

STORY PAGE TEXT (Page {page_number}):
{story_page_text}
- Embed this story page text into the image naturally.

CHARACTER INFORMATION:
- Character Name: {character_name}
- Character Type: {character_type}
- Story World: {story_world}
{environment_details}
{character_reference_note}
{character_consistency_enforcement}
ILLUSTRATION REQUIREMENTS:
1. Create a vibrant, age-appropriate children's book illustration
2. Include the main character ({character_name}) as a {character_type} - {character_name} is the clear hero of this story
3. CHARACTER PROMINENCE: The character ({character_name}) must occupy 60-70% of the composition. The character should be the dominant visual element, clearly visible and prominent in the scene
4. Match the mood, setting, and events from the story text
5. Use bright, cheerful colors suitable for children
6. Make it visually appealing and engaging
7. Ensure the scene is positive and appropriate for children
8. Include relevant details about the setting and characters
9. Style should be like a professional children's book illustration
10. IMPORTANT: The image must be in 768x512 dimensions
{"11. CRITICAL: The character must match the appearance shown in the reference image provided" if reference_image_url and base_image_data else ""}
{negative_prompts}

Generate a high-quality illustration that perfectly captures this story moment in 768x512 dimensions."""

        # Use edit_image function to generate the scene
        logger.info(f"Calling edit_image function with prompt for page {page_number}")
        scene_image_bytes = edit_image(base_image_data, prompt, reference_image_url)
        
        # Optimize image to JPG format
        logger.info("Optimizing scene image to JPG format...")
        optimized_image = optimize_image_to_jpg(scene_image_bytes)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"story_scene_page{page_number}_{timestamp}_{unique_id}.jpg"
        
        # Upload to Supabase and get URL
        storage_result = upload_to_supabase(optimized_image, filename)
        
        if storage_result.get("uploaded") and storage_result.get("url"):
            logger.info(f"✅ Scene image generated and uploaded for page {page_number}: {storage_result['url']}")
            return storage_result['url']
        else:
            logger.warning(f"Failed to upload scene image for page {page_number}")
            return ""
        
    except HTTPException as e:
        logger.error(f"HTTP error generating scene image for page {page_number}: {e.detail}")
        return ""
    except Exception as e:
        logger.error(f"Error generating scene image for page {page_number}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return ""

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Image Editor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/validate-image-quality/", response_model=QualityValidationResponse)
async def validate_image_quality_endpoint(request: ImageRequest):
    """
    Standalone endpoint to validate image quality without editing.
    Useful for pre-validation before processing.
    """
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(request.image_url)
        
        # Download the image from the URL provided
        logger.info(f"Downloading image for quality validation from: {image_url_str}")
        image_data = download_image_from_url(image_url_str)
        
        # Validate image quality
        validation_result = validate_image_quality(image_data, image_url_str)
        
        return QualityValidationResponse(
            success=True,
            validation=validation_result
        )
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in validate_image_quality_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_client_initialized": bool(gemini_client is not None),
        "openai_api_key_configured": bool(OPENAI_API_KEY),
        "model": MODEL,
        "supabase_configured": bool(supabase is not None),
        "storage_bucket": STORAGE_BUCKET if supabase else None,
        "quality_validation_enabled": bool(gemini_client is not None)
    }

@app.post("/edit-image/", response_model=ImageResponse)
async def edit_image_endpoint(request: ImageRequest):
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(request.image_url)
        
        # Download the image from the URL provided
        logger.info(f"Downloading image from: {image_url_str}")
        image_data = download_image_from_url(image_url_str)

        # Validate image quality before processing
        logger.info("Validating image quality...")
        quality_validation = validate_image_quality(image_data, image_url_str)
        
        # Log validation results
        if not quality_validation.get("is_valid", True):
            logger.warning(f"Image quality validation failed: {quality_validation.get('issues', [])}")
            # Optionally raise error or continue with warning
            # For now, we'll continue but include validation in response
        
        # Send the image to Gemini API for editing
        logger.info(f"Received prompt: {request.prompt}")
        edited_image = edit_image(image_data, request.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        logger.info("Optimizing image to JPG format...")
        optimized_image = optimize_image_to_jpg(edited_image)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"edited_image_{timestamp}_{unique_id}.jpg"
        
        # Upload optimized image to Supabase storage
        storage_result = upload_to_supabase(optimized_image, filename)
        
        if storage_result["uploaded"]:
            return ImageResponse(
                success=True,
                message="Image edited and uploaded successfully to Supabase storage",
                storage_info=storage_result,
                quality_validation=quality_validation
            )
        else:
            # Even if upload fails, we can still return the image data
            logger.warning("Supabase upload failed, but image was processed successfully")
            return ImageResponse(
                success=True,
                message="Image edited successfully, but storage upload failed",
                storage_info=storage_result,
                quality_validation=quality_validation
            )
            
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in edit_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/edit-image-stream/")
async def edit_image_stream_endpoint(request: ImageRequest):
    """Alternative endpoint that returns the image as a stream (for direct download)"""
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(request.image_url)
        
        # Download the image from the URL provided
        logger.info(f"Downloading image from: {image_url_str}")
        image_data = download_image_from_url(image_url_str)

        # Send the image to Gemini API for editing
        logger.info(f"Received prompt: {request.prompt}")
        edited_image = edit_image(image_data, request.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        logger.info("Optimizing image to JPG format...")
        optimized_image = optimize_image_to_jpg(edited_image)
        
        return StreamingResponse(
            BytesIO(optimized_image), 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=edited_image.jpg"}
        )
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in edit_image_stream_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# Request model for batch job creation
class BatchJobRequest(BaseModel):
    job_type: str  # 'interactive_search' or 'story_adventure'
    character_name: str
    character_type: str
    special_ability: str
    age_group: str
    story_world: str
    adventure_type: str
    occasion_theme: Optional[str] = None
    character_image_url: Optional[HttpUrl] = None
    priority: int = 5  # 1-10, 1 is highest
    user_id: Optional[str] = None
    child_profile_id: Optional[int] = None

# Response model for job creation
class JobResponse(BaseModel):
    success: bool
    job_id: int
    message: str

# Response model for job status
class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    overall_progress: int
    stages: List[Dict[str, Any]]
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

async def background_worker():
    """Background worker that processes jobs from the queue"""
    logger.info("Background worker started")
    while True:
        try:
            if not queue_manager:
                await asyncio.sleep(5)
                continue
            
            # Get next job
            job = queue_manager.get_next_job()
            
            if job:
                job_id = job["id"]
                logger.info(f"Processing job {job_id}")
                await batch_processor.process_job(job_id)
            else:
                # No jobs available, wait before checking again
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.info("Background worker cancelled")
            break
        except Exception as e:
            logger.error(f"Error in background worker: {e}")
            await asyncio.sleep(5)

@app.post("/api/books/generate", response_model=JobResponse)
async def create_book_generation_job(request: BatchJobRequest):
    """Create a new book generation job"""
    try:
        if not queue_manager:
            raise HTTPException(
                status_code=500,
                detail="Queue manager not initialized"
            )
        
        # Validate job_type
        if request.job_type not in ["interactive_search", "story_adventure"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid job_type. Must be 'interactive_search' or 'story_adventure'"
            )
        
        # Validate age_group
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if request.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {request.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        # Validate priority
        if request.priority < 1 or request.priority > 10:
            raise HTTPException(
                status_code=400,
                detail="Priority must be between 1 and 10 (1 is highest)"
            )
        
        # Prepare job data
        job_data = {
            "character_name": request.character_name,
            "character_type": request.character_type,
            "special_ability": request.special_ability,
            "age_group": request.age_group,
            "story_world": request.story_world,
            "adventure_type": request.adventure_type,
            "occasion_theme": request.occasion_theme,
            "character_image_url": str(request.character_image_url) if request.character_image_url else None
        }
        
        # Create job
        job = queue_manager.create_job(
            job_type=request.job_type,
            job_data=job_data,
            user_id=request.user_id,
            child_profile_id=request.child_profile_id,
            priority=request.priority
        )
        
        return JobResponse(
            success=True,
            job_id=job["id"],
            message=f"Job {job['id']} created successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")

@app.get("/api/books/{book_id}/status", response_model=JobStatusResponse)
async def get_book_status(book_id: int):
    """Get the status of a book generation job"""
    try:
        if not queue_manager:
            raise HTTPException(
                status_code=500,
                detail="Queue manager not initialized"
            )
        
        job_status = queue_manager.get_job_status(book_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail=f"Job {book_id} not found"
            )
        
        job = job_status["job"]
        
        return JobStatusResponse(
            job_id=book_id,
            status=job["status"],
            overall_progress=job_status["overall_progress"],
            stages=job_status["stages"],
            error_message=job.get("error_message"),
            result_data=job.get("result_data")
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")

@app.post("/generate-story/", response_model=StoryResponse)
async def generate_story_endpoint(request: StoryRequest):
    """Generate a 5-page children's story based on the provided parameters"""
    try:
        # Validate age_group
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if request.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {request.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        logger.info(f"Generating story for character: {request.character_name}")
        logger.info(f"Age group: {request.age_group}, Adventure: {request.adventure_type}")
        
        # Validate API keys
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        if not GEMINI_API_KEY or not gemini_client:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key not configured or client not initialized. Please set GEMINI_API_KEY environment variable."
            )
        
        # Generate the story using OpenAI (GPT-4) via the story library
        logger.info("Generating story with OpenAI GPT-4...")
        story_result = generate_story(
            character_name=request.character_name,
            character_type=request.character_type,
            special_ability=request.special_ability,
            age_group=request.age_group,
            story_world=request.story_world,
            adventure_type=request.adventure_type,
            occasion_theme=request.occasion_theme,
            use_api=True,  # Use OpenAI API for story generation
            api_key=OPENAI_API_KEY,
            story_text_prompt=request.story_text_prompt  # Use prompt from frontend if provided
        )
        
        logger.info(f"Story generated successfully. Word count: {story_result['word_count']}")
        
        # Generate scene images for each page using Gemini Pro image preview model
        logger.info("Generating scene images with Gemini Pro image preview model for each story page...")
        reference_image_url = str(request.character_image_url) if request.character_image_url else None
        
        # Download reference image once for consistency validation
        reference_image_data = None
        if reference_image_url:
            try:
                logger.info(f"Downloading reference image for consistency validation: {reference_image_url}")
                reference_image_data = download_image_from_url(reference_image_url)
                logger.info(f"✅ Reference image downloaded for validation ({len(reference_image_data)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to download reference image for validation: {e}")
                reference_image_data = None
        
        story_pages = []
        consistency_results = []
        flagged_pages = []
        
        for i, page_text in enumerate(story_result['pages'], 1):
            logger.info(f"Generating scene image for page {i}/5...")
            # Use scene prompt from frontend if available, otherwise use None (will generate from params)
            scene_prompt = None
            if request.scene_prompts and len(request.scene_prompts) >= i:
                scene_prompt = request.scene_prompts[i - 1]  # i is 1-indexed, list is 0-indexed
                # Replace placeholder with actual page text
                scene_prompt = scene_prompt.replace(
                    f"[Page {i} text will be inserted here after story generation]",
                    page_text
                )
                logger.info(f"Using scene prompt from frontend for page {i} (with actual page text)")
            
            scene_url = generate_story_scene_image(
                story_page_text=page_text,
                page_number=i,
                character_name=request.character_name,
                character_type=request.character_type,
                story_world=request.story_world,
                reference_image_url=reference_image_url,
                scene_prompt=scene_prompt
            )
            # Convert string URL to HttpUrl if not empty, otherwise None
            scene_http_url = None
            scene_image_data = None
            consistency_validation = None
            
            if scene_url:
                try:
                    scene_http_url = HttpUrl(scene_url)
                    # Download scene image for consistency validation
                    try:
                        scene_image_data = download_image_from_url(scene_url)
                        logger.info(f"✅ Scene image downloaded for validation ({len(scene_image_data)} bytes)")
                    except Exception as e:
                        logger.warning(f"Failed to download scene image for validation: {e}")
                except Exception as e:
                    logger.warning(f"Invalid scene URL for page {i}: {e}")
                    scene_http_url = None
            
            # Perform character consistency validation if both images are available
            if reference_image_data and scene_image_data:
                logger.info(f"Performing character consistency validation for page {i}...")
                try:
                    consistency_validation = validate_character_consistency(
                        scene_image_data=scene_image_data,
                        reference_image_data=reference_image_data,
                        page_number=i,
                        timeout_seconds=15
                    )
                    consistency_results.append(consistency_validation)
                    
                    if consistency_validation.flagged:
                        flagged_pages.append(i)
                        logger.warning(f"⚠️ Page {i} flagged as INCONSISTENT (similarity: {consistency_validation.similarity_score:.3f})")
                    else:
                        logger.info(f"✅ Page {i} validated as CONSISTENT (similarity: {consistency_validation.similarity_score:.3f})")
                except Exception as e:
                    logger.error(f"Error during consistency validation for page {i}: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            elif not reference_image_data:
                logger.info(f"Skipping consistency validation for page {i} - no reference image available")
            elif not scene_image_data:
                logger.warning(f"Skipping consistency validation for page {i} - scene image not available")
            
            story_pages.append(StoryPage(
                text=page_text, 
                scene=scene_http_url,
                consistency_validation=consistency_validation
            ))
        
        logger.info("All scene images generated successfully")
        
        # Create consistency summary
        consistency_summary = None
        if consistency_results:
            avg_score = sum(r.similarity_score for r in consistency_results) / len(consistency_results)
            min_score = min(r.similarity_score for r in consistency_results)
            max_score = max(r.similarity_score for r in consistency_results)
            total_validation_time = sum(r.validation_time_seconds for r in consistency_results)
            consistent_count = sum(1 for r in consistency_results if r.is_consistent)
            
            consistency_summary = {
                "total_pages_validated": len(consistency_results),
                "consistent_pages": consistent_count,
                "inconsistent_pages": len(consistency_results) - consistent_count,
                "flagged_pages": flagged_pages,
                "average_similarity_score": round(avg_score, 3),
                "min_similarity_score": round(min_score, 3),
                "max_similarity_score": round(max_score, 3),
                "total_validation_time_seconds": round(total_validation_time, 2),
                "average_validation_time_seconds": round(total_validation_time / len(consistency_results), 2),
                "all_consistent": len(flagged_pages) == 0
            }
            
            logger.info("=" * 60)
            logger.info("CHARACTER CONSISTENCY VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total pages validated: {consistency_summary['total_pages_validated']}")
            logger.info(f"Consistent pages: {consistency_summary['consistent_pages']}")
            logger.info(f"Inconsistent pages: {consistency_summary['inconsistent_pages']}")
            if flagged_pages:
                logger.warning(f"⚠️ Flagged pages (inconsistent): {flagged_pages}")
            logger.info(f"Average similarity score: {avg_score:.3f}")
            logger.info(f"Score range: {min_score:.3f} - {max_score:.3f}")
            logger.info(f"Total validation time: {total_validation_time:.2f}s")
            logger.info(f"Average validation time per page: {total_validation_time / len(consistency_results):.2f}s")
            logger.info("=" * 60)
        
        return StoryResponse(
            success=True,
            pages=story_pages,
            full_story=story_result['full_story'],
            word_count=story_result['word_count'],
            page_word_counts=story_result['page_word_counts'],
            consistency_summary=consistency_summary
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_story_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting AI Image Editor Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚡ Server running on: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
