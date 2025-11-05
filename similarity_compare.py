"""
Module for comparing image similarity using HuggingFace models.
"""
import requests
from PIL import Image
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Initialize model (lazy loading)
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    """Lazy load the CLIP-ViT-B-32 model"""
    global _model
    if _model is None:
        logger.info(f"Loading CLIP-ViT-B-32 model on device: {_device}")
        model_name = "sentence-transformers/clip-ViT-B-32"  # CLIP-ViT-B-32 model
        _model = SentenceTransformer(model_name, device=_device)
        logger.info("CLIP-ViT-B-32 model loaded successfully")
    return _model

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to download image from URL {url}: {e}")

def compare_similarity(image_url1: str, image_url2: str) -> float:
    """
    Compare two images and return similarity score (0-1 range) using CLIP-ViT-B-32.
    
    Args:
        image_url1: URL of first image
        image_url2: URL of second image
    
    Returns:
        float: Similarity score between 0 and 1 (1 = identical, 0 = completely different)
    """
    try:
        # Load model
        model = get_model()
        
        # Download images
        logger.info(f"Downloading image 1 from: {image_url1}")
        image1 = download_image(image_url1)
        
        logger.info(f"Downloading image 2 from: {image_url2}")
        image2 = download_image(image_url2)
        
        # Get image embeddings using CLIP-ViT-B-32
        embeddings = model.encode([image1, image2], convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarity
        # embeddings are already normalized by sentence-transformers
        similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        
        logger.info(f"Similarity score calculated: {similarity:.4f}")
        return float(similarity)
    
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        raise

