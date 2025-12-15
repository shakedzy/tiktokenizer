"""
TikTokenizer - Convert HuggingFace tokenizers to tiktoken format.

Example:
    from tiktokenizer import TikTokenizer

    encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")
    tokens = encoding.encode("Hello world")
"""

from importlib.metadata import version
from .core import TikTokenizer, IncompatibleTokenizerError

__all__ = ["TikTokenizer", "IncompatibleTokenizerError"]
__dist_name__ = "tiktokenizer"
__version__ = version(__dist_name__)
