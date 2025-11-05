"""
Character Similarity Comparison Module
A Python module for comparing cartoon character images using CLIP embeddings.
"""

from .similarity_compare import (
    compare_character_similarity,
    interpret_similarity_score,
    CharacterIdentityComparator
)

__version__ = "1.0.0"
__all__ = [
    'compare_character_similarity',
    'interpret_similarity_score',
    'CharacterIdentityComparator'
]

