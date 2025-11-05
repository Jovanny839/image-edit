from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import base64
import requests
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
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import similarity_compare with error handling for serverless environments
SIMILARITY_AVAILABLE = False
compare_character_similarity = None

try:
    # Try importing required dependencies first
    import torch
    import clip
    import numpy as np
    logger.info("Core dependencies (torch, clip, numpy) available")
    
    # Now try importing the similarity module
    from similarity_compare import compare_character_similarity
    SIMILARITY_AVAILABLE = True
    logger.info("Similarity comparison module loaded successfully")
except ImportError as e:
    logger.warning(f"similarity_compare module or dependencies not available: {e}")
    logger.warning("Install with: pip install torch torchvision git+https://github.com/openai/CLIP.git numpy")
    SIMILARITY_AVAILABLE = False
    compare_character_similarity = None
except Exception as e:
    logger.error(f"Error importing similarity_compare: {e}")
    import traceback
    logger.error(traceback.format_exc())
    SIMILARITY_AVAILABLE = False
    compare_character_similarity = None

# === CONFIG ===
API_KEY = os.getenv("OPENAI_API_KEY", "")
ENDPOINT = "https://api.openai.com/v1/images/edits"
MODEL = "gpt-image-1"
OUTPUT_FILE = "edited_image.png"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for storage operations
STORAGE_BUCKET = "images"

# Initialize Supabase client
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
    description="API for editing images using OpenAI's image editing capabilities",
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Additional middleware to ensure CORS headers are always present (for serverless environments)
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    """Add CORS headers to all responses for serverless compatibility"""
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    return response

# Global exception handler for better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
        }
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

# Request model for similarity comparison
class SimilarityRequest(BaseModel):
    image1_url: HttpUrl
    image2_url: HttpUrl
    
    class Config:
        schema_extra = {
            "example": {
                "image1_url": "https://your-project.supabase.co/storage/v1/object/public/images/image1.jpg",
                "image2_url": "https://your-project.supabase.co/storage/v1/object/public/images/image2.jpg"
            }
        }

# Response model for similarity comparison
class SimilarityResponse(BaseModel):
    similarity_score: float
    image1_url: str
    image2_url: str
    
    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 0.85,
                "image1_url": "https://your-project.supabase.co/storage/v1/object/public/images/image1.jpg",
                "image2_url": "https://your-project.supabase.co/storage/v1/object/public/images/image2.jpg"
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
        image = Image.open(BytesIO(image_data))
        original_size_info = f"{image.width}x{image.height}"
        
        # Convert to RGB if necessary (PNG with transparency, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = Image.new('RGB', image.size, (255, 255, 255))
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
    """Send image to OpenAI API for editing"""
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Prepare files for upload
    # Determine content type from URL if provided, otherwise default to jpeg
    if image_url:
        image_content_type = get_content_type_from_url(image_url)
    else:
        image_content_type = "image/png"  # default fallback

    files = {"image": ("image", image_data, image_content_type)}

    data = {
        "model": MODEL,
        "prompt": prompt,
        "n": "1",
        "size": "auto"
    }

    print("Sending request to OpenAI image edits endpoint...")

    start_time = time.time()
    resp = requests.post(ENDPOINT, headers=headers, files=files, data=data, timeout=120)
    elapsed = time.time() - start_time

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Error from API: {resp.status_code}, {resp.text}")

    j = resp.json()

    # try base64 field first
    image_data = j["data"][0].get("b64_json")
    if image_data:
        image_bytes = base64.b64decode(image_data)
        return image_bytes

    # fallback: URL
    url = j["data"][0].get("url")
    if url:
        download = requests.get(url)
        download.raise_for_status()
        return download.content

    raise HTTPException(status_code=500, detail="Unexpected response JSON from OpenAI API")

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
        "api_key_configured": bool(API_KEY and API_KEY.startswith("sk-")),
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

        # Send the image to OpenAI API for editing
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

        # Send the image to OpenAI API for editing
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

@app.options("/compare-similarity/")
async def compare_similarity_options():
    """Handle preflight OPTIONS request for CORS"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.post("/compare-similarity/")
async def compare_similarity_endpoint(request: SimilarityRequest):
    """
    Compare similarity between two images from Supabase URLs.
    
    Returns a similarity score between 0 and 1, where higher scores indicate
    more similar images (typically same character/object).
    """
    try:
        # Check if similarity module is available
        if not SIMILARITY_AVAILABLE or compare_character_similarity is None:
            logger.error("Similarity comparison module is not available")
            raise HTTPException(
                status_code=503,
                detail="Similarity comparison service is not available. Required dependencies (torch, clip) may be missing."
            )
        
        # Convert HttpUrl to string for processing
        image1_url_str = str(request.image1_url)
        image2_url_str = str(request.image2_url)
        
        # Download both images from Supabase URLs
        logger.info(f"Downloading image 1 from: {image1_url_str}")
        try:
            image1_data = download_image_from_url(image1_url_str)
        except Exception as e:
            logger.error(f"Failed to download image 1: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download image 1 from URL: {str(e)}")
        
        logger.info(f"Downloading image 2 from: {image2_url_str}")
        try:
            image2_data = download_image_from_url(image2_url_str)
        except Exception as e:
            logger.error(f"Failed to download image 2: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download image 2 from URL: {str(e)}")
        
        # Convert image data to PIL Images
        try:
            image1 = Image.open(BytesIO(image1_data))
            if image1.mode != 'RGB':
                image1 = image1.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to process image 1: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process image 1: {str(e)}")
        
        try:
            image2 = Image.open(BytesIO(image2_data))
            if image2.mode != 'RGB':
                image2 = image2.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to process image 2: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process image 2: {str(e)}")
        
        # Compare images using similarity_compare module
        logger.info("Comparing images using similarity_compare module...")
        try:
            # Note: First call will load the CLIP model, which may take time
            # In serverless, ensure enough memory and timeout allowance
            similarity_score = compare_character_similarity(
                image1, 
                image2, 
                verbose=False,
                device='cpu'  # Use CPU for serverless (no GPU typically available)
            )
        except RuntimeError as e:
            logger.error(f"Runtime error during similarity comparison: {e}")
            error_msg = str(e)
            # Check if it's a model loading issue
            if "CLIP" in error_msg or "model" in error_msg.lower() or "out of memory" in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail=f"Model loading failed. This may be a serverless environment issue (memory/timeout). Error: {error_msg}"
                )
            raise HTTPException(status_code=500, detail=f"Similarity comparison failed: {error_msg}")
        except OSError as e:
            # Network or file system errors (e.g., model download timeout)
            logger.error(f"OS error during similarity comparison: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model download or file system error. This may be a serverless timeout issue. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error during similarity comparison: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Similarity comparison error: {str(e)}")
        
        logger.info(f"Similarity score: {similarity_score:.4f}")
        
        response_data = SimilarityResponse(
            similarity_score=float(similarity_score),
            image1_url=image1_url_str,
            image2_url=image2_url_str
        )
        
        # Add CORS headers explicitly for serverless environments
        return JSONResponse(
            content=response_data.dict(),
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            }
        )
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in compare_similarity_endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
