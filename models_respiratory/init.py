"""
models_respiratory/__init__.py

Package initialization for sleep apnea detection models
"""

from .model import CNNTransformerHybrid
from .attention import PositionalEncoding, TransformerBlock

# Alias for backward compatibility
SleepApneaCNN = CNNTransformerHybrid

__all__ = [
    'CNNTransformerHybrid',
    'SleepApneaCNN',
    'TransformerBlock', 
    'PositionalEncoding'
]
