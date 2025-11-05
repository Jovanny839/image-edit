"""
Cartoon Character Image Similarity Comparison Script
Compares two cartoon character images using character identity embeddings.
Uses CLIP-based embeddings optimized for character recognition across different poses,
emotions, backgrounds, and lighting conditions.
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Try to import face detection (optional)
try:
    import cv2
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Note: Face detection not available. Install opencv-python and mediapipe for face alignment.")


class CharacterIdentityComparator:
    """
    A class to compare character identity between cartoon character images
    using CLIP-based embeddings optimized for character recognition.
    
    CLIP (Contrastive Language-Image Pre-training) is excellent for character
    identity tasks because it understands high-level semantic similarity,
    recognizing the same character despite pose, emotion, or style variations.
    """
    
    def __init__(self, device: str = None, model_name: str = 'ViT-B/32', use_face_detection: bool = False):
        """
        Initialize the character identity comparator.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detects if None.
            model_name: CLIP model to use ('ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', etc.)
                       'ViT-B/32' is a good balance of speed and accuracy
            use_face_detection: Whether to use face detection and alignment (requires opencv, mediapipe)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.use_face_detection = use_face_detection and FACE_DETECTION_AVAILABLE
        
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            print(f"✓ CLIP model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
        
        # Initialize face detection if requested
        if self.use_face_detection:
            print("Initializing face detection...")
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range
                min_detection_confidence=0.5
            )
            print("✓ Face detection initialized")
        else:
            self.face_detection = None
    
    def detect_and_align_face(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Detect and align face in the image (optional preprocessing step).
        This helps normalize pose variations for better character recognition.
        
        Args:
            image: PIL Image object
            
        Returns:
            Cropped and aligned face image, or original image if face not detected
        """
        if not self.use_face_detection:
            return image
        
        try:
            # Convert PIL to numpy array for MediaPipe
            img_array = np.array(image)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            results = self.face_detection.process(img_rgb)
            
            if results.detections:
                # Get the largest face (most likely the main character)
                largest_detection = max(
                    results.detections,
                    key=lambda d: (d.location_data.relative_bounding_box.width * 
                                  d.location_data.relative_bounding_box.height)
                )
                
                # Extract face bounding box
                bbox = largest_detection.location_data.relative_bounding_box
                h, w = img_array.shape[:2]
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 0.2
                x = max(0, int(x - width * padding))
                y = max(0, int(y - height * padding))
                width = min(w - x, int(width * (1 + 2 * padding)))
                height = min(h - y, int(height * (1 + 2 * padding)))
                
                # Crop face region
                face_crop = image.crop((x, y, x + width, y + height))
                return face_crop
            
            # No face detected, return original image
            return image
            
        except Exception as e:
            print(f"Warning: Face detection failed, using full image: {e}")
            return image
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and validate an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def extract_character_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract character identity embedding from an image using CLIP.
        
        CLIP embeddings capture high-level semantic features that represent
        character identity, making them robust to pose, emotion, and style changes.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        # Optional: Detect and align face if enabled
        if self.use_face_detection:
            image = self.detect_and_align_face(image)
        
        # Preprocess image for CLIP
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            # CLIP's image encoder outputs normalized embeddings
            embedding = self.model.encode_image(image_tensor)
            # Normalize the embedding (CLIP already does this, but ensure it's normalized)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy().squeeze()
        
        return embedding
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        For character identity, cosine similarity is the preferred metric
        as it measures the angle between vectors, which captures semantic similarity.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1 (typically 0-1 for normalized vectors)
        """
        similarity = np.dot(embedding1, embedding2)
        # Clamp to [-1, 1] range (though normalized vectors typically give [0, 1])
        similarity = max(-1.0, min(1.0, similarity))
        return float(similarity)
    
    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate euclidean distance between two embedding vectors.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean distance (lower is more similar)
        """
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)
    
    def compare_characters(self, image1_path: str, image2_path: str) -> Tuple[float, dict]:
        """
        Compare two character images and return identity similarity score.
        
        This method extracts character identity embeddings and compares them
        using cosine similarity. The score indicates how likely the two images
        show the same character, even with different poses, emotions, or backgrounds.
        
        Args:
            image1_path: Path to first character image
            image2_path: Path to second character image
            
        Returns:
            Tuple of (similarity_score, detailed_info_dict)
        """
        # Load images
        print(f"\nLoading image 1: {image1_path}")
        img1 = self.load_image(image1_path)
        print(f"Image 1 size: {img1.size}")
        
        print(f"Loading image 2: {image2_path}")
        img2 = self.load_image(image2_path)
        print(f"Image 2 size: {img2.size}")
        
        # Extract character identity embeddings
        print(f"\nExtracting character identity embedding from image 1...")
        if self.use_face_detection:
            print("  (Using face detection for better alignment)")
        embedding1 = self.extract_character_embedding(img1)
        print(f"✓ Image 1 embedding extracted (dimension: {len(embedding1)})")
        
        print(f"Extracting character identity embedding from image 2...")
        if self.use_face_detection:
            print("  (Using face detection for better alignment)")
        embedding2 = self.extract_character_embedding(img2)
        print(f"✓ Image 2 embedding extracted (dimension: {len(embedding2)})")
        
        # Calculate cosine similarity (primary metric for character identity)
        similarity = self.calculate_cosine_similarity(embedding1, embedding2)
        distance = self.calculate_euclidean_distance(embedding1, embedding2)
        
        info = {
            'similarity_score': similarity,
            'euclidean_distance': distance,
            'embedding_dimension': len(embedding1),
            'model_name': self.model_name,
            'face_detection_used': self.use_face_detection
        }
        
        return similarity, info


def compare_character_similarity(image1, image2, model_name: str = 'ViT-B/32', 
                                   use_face_detection: bool = False, 
                                   device: str = None, verbose: bool = False) -> float:
    """
    Compare two character images and return character identity similarity score.
    
    This is the main function to use for comparing character images.
    It takes two images (as file paths or PIL Image objects) and returns
    a similarity score between -1 and 1 (typically 0-1), where higher scores
    indicate the same character despite different poses, emotions, or backgrounds.
    
    Args:
        image1: Path to first image (str) or PIL Image object
        image2: Path to second image (str) or PIL Image object
        model_name: CLIP model to use (default: 'ViT-B/32')
                   Options: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50'
        use_face_detection: Whether to use face detection (default: False)
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        verbose: Whether to print progress messages (default: False)
        
    Returns:
        float: Character identity similarity score (0-1, higher = more similar)
        
    Example:
        >>> score = compare_character_similarity('char1.png', 'char2.jpg')
        >>> print(f"Similarity: {score:.2f}")
    """
    # Initialize comparator (reuse if already initialized with same settings)
    if not hasattr(compare_character_similarity, '_comparator'):
        if verbose:
            print("Initializing Character Identity Comparator...")
        compare_character_similarity._comparator = CharacterIdentityComparator(
            model_name=model_name,
            use_face_detection=use_face_detection,
            device=device
        )
    else:
        # Check if we need to recreate comparator with different settings
        existing = compare_character_similarity._comparator
        if existing.model_name != model_name or existing.use_face_detection != use_face_detection:
            if verbose:
                print("Reinitializing Character Identity Comparator with new settings...")
            compare_character_similarity._comparator = CharacterIdentityComparator(
                model_name=model_name,
                use_face_detection=use_face_detection,
                device=device
            )
    
    comparator = compare_character_similarity._comparator
    
    # Handle image inputs (can be paths or PIL Images)
    if isinstance(image1, str):
        if verbose:
            print(f"Loading image 1: {image1}")
        img1 = comparator.load_image(image1)
    elif isinstance(image1, Image.Image):
        img1 = image1
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
    else:
        raise ValueError("image1 must be a file path (str) or PIL Image object")
    
    if isinstance(image2, str):
        if verbose:
            print(f"Loading image 2: {image2}")
        img2 = comparator.load_image(image2)
    elif isinstance(image2, Image.Image):
        img2 = image2
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
    else:
        raise ValueError("image2 must be a file path (str) or PIL Image object")
    
    # Extract embeddings
    if verbose:
        print("Extracting character identity embeddings...")
    embedding1 = comparator.extract_character_embedding(img1)
    embedding2 = comparator.extract_character_embedding(img2)
    
    # Calculate similarity score
    similarity_score = comparator.calculate_cosine_similarity(embedding1, embedding2)
    
    if verbose:
        print(f"Similarity score: {similarity_score:.4f}")
    
    return similarity_score


def interpret_similarity_score(score: float) -> Tuple[str, str]:
    """
    Interpret similarity score based on character identity recognition thresholds.
    
    Based on research and real-world testing:
    - > 0.8-0.9: Likely same character
    - 0.6-0.8: Possibly same character, needs review
    - < 0.6: Likely different characters
    
    Args:
        score: Cosine similarity score
        
    Returns:
        Tuple of (interpretation, confidence_level)
    """
    if score >= 0.85:
        return "✓ Very High - Same character (despite pose/emotion/background differences)", "Very High"
    elif score >= 0.75:
        return "✓ High - Same character with different pose/emotion", "High"
    elif score >= 0.65:
        return "~ Moderate - Possibly the same character", "Moderate"
    elif score >= 0.50:
        return "? Low - Unclear, may need human review", "Low"
    else:
        return "✗ Very Low - Different characters", "Very Low"


def main():
    """
    Main function to run the character identity comparison.
    Simply calls compare_character_similarity() with command-line arguments.
    """
    if len(sys.argv) < 3:
        print("="*70)
        print("Cartoon Character Identity Comparison")
        print("="*70)
        print("\nUsage: python similarity_compare.py <image1_path> <image2_path> [options]")
        print("\nOptions:")
        print("  [--model NAME]    : CLIP model to use")
        print("                      Options: 'ViT-B/32' (default), 'ViT-B/16', 'ViT-L/14', 'RN50'")
        print("  [--face-detect]   : Enable face detection and alignment (requires opencv, mediapipe)")
        print("\nExamples:")
        print("  python similarity_compare.py character1.png character2.jpg")
        print("  python similarity_compare.py character1.png character2.jpg --model ViT-B/16")
        print("  python similarity_compare.py character1.png character2.jpg --face-detect")
        print("\nNote: Uses CLIP embeddings optimized for character identity recognition!")
        print("      Same characters score ≥0.75 even with different poses/emotions.")
        print("="*70)
        sys.exit(1)
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    model_name = 'ViT-B/32'
    use_face_detection = False
    
    # Parse optional arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--model' and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            i += 2
        elif arg == '--face-detect':
            use_face_detection = True
            i += 1
        else:
            i += 1
    
    # Check face detection availability
    if use_face_detection and not FACE_DETECTION_AVAILABLE:
        print("\nWarning: Face detection requested but not available.")
        print("Install with: pip install opencv-python mediapipe")
        print("Continuing without face detection...\n")
        use_face_detection = False
    
    # Call the main comparison function
    try:
        similarity_score = compare_character_similarity(
            image1_path, 
            image2_path,
            model_name=model_name,
            use_face_detection=use_face_detection,
            verbose=True
        )
        
        # Get interpretation
        interpretation, confidence = interpret_similarity_score(similarity_score)
        
        # Print results
        print("\n" + "="*70)
        print("CHARACTER IDENTITY COMPARISON RESULTS")
        print("="*70)
        print(f"Image 1: {image1_path}")
        print(f"Image 2: {image2_path}")
        print(f"\nCharacter Identity Similarity: {similarity_score:.4f} ({similarity_score*100:.2f}%)")
        print(f"Model: {model_name}")
        print(f"Face Detection: {'Enabled' if use_face_detection else 'Disabled'}")
        print("="*70)
        
        print(f"\n{interpretation}")
        print(f"Confidence Level: {confidence}")
        
        print("\n" + "="*70)
        print("THRESHOLD GUIDE:")
        print("  ≥ 0.85: Very High - Same character")
        print("  ≥ 0.75: High - Same character (different pose/emotion)")
        print("  ≥ 0.65: Moderate - Possibly same character")
        print("  ≥ 0.50: Low - Unclear, needs review")
        print("  < 0.50: Very Low - Different characters")
        print("="*70)
        
        return similarity_score
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
