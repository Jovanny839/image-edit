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
from lib.story_lib import generate_story
from typing import List, Optional


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3-pro-image-preview"

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

# FastAPI app
app = FastAPI(
    title="AI Image Editor API",
    description="API for editing images using Google Gemini's image generation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
    
    class Config:
        schema_extra = {
            "example": {
                "character_name": "Luna",
                "character_type": "a brave dragon",
                "special_ability": "fly through clouds",
                "age_group": "7-10",
                "story_world": "the Enchanted Forest",
                "adventure_type": "treasure hunt",
                "occasion_theme": None
            }
        }

# Response model for story generation
class StoryResponse(BaseModel):
    success: bool
    pages: List[str]
    full_story: str
    word_count: int
    page_word_counts: List[int]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "pages": [
                    "Meet Luna, a brave dragon who loves adventures. Luna has a special power: Luna can fly through clouds.",
                    "While exploring, Luna discovered a magical entrance that led to the Enchanted Forest.",
                    "Suddenly, Luna realized that a treasure hunt was beginning, and Luna was right in the middle of it.",
                    "When the moment of truth arrived, Luna faced the challenge head-on, even though it seemed impossible at first.",
                    "The adventure came to a wonderful conclusion, and Luna felt proud of what had been accomplished."
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Image Editor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_client_initialized": bool(gemini_client is not None),
        "model": MODEL,
        "supabase_configured": bool(supabase is not None),
        "storage_bucket": STORAGE_BUCKET if supabase else None
    }

@app.post("/edit-image/", response_model=ImageResponse)
async def edit_image_endpoint(request: ImageRequest):
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
                storage_info=storage_result
            )
        else:
            # Even if upload fails, we can still return the image data
            logger.warning("Supabase upload failed, but image was processed successfully")
            return ImageResponse(
                success=True,
                message="Image edited successfully, but storage upload failed",
                storage_info=storage_result
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
        
        # Generate the story using the story library
        story_result = generate_story(
            character_name=request.character_name,
            character_type=request.character_type,
            special_ability=request.special_ability,
            age_group=request.age_group,
            story_world=request.story_world,
            adventure_type=request.adventure_type,
            occasion_theme=request.occasion_theme,
            use_api=False  # Use template-based generation
        )
        
        logger.info(f"Story generated successfully. Word count: {story_result['word_count']}")
        
        return StoryResponse(
            success=True,
            pages=story_result['pages'],
            full_story=story_result['full_story'],
            word_count=story_result['word_count'],
            page_word_counts=story_result['page_word_counts']
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
