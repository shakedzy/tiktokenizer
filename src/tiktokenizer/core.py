"""
Core TikTokenizer implementation.

This module provides the TikTokenizer class for converting HuggingFace
tokenizers to tiktoken format.
"""

from __future__ import annotations
import json
import base64
import tiktoken
from pathlib import Path
from typing import TYPE_CHECKING
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


__all__ = ["TikTokenizer", "IncompatibleTokenizerError"]


class IncompatibleTokenizerError(Exception):
    """Raised when a tokenizer is not compatible with tiktoken conversion."""
    pass


def _bytes_to_unicode() -> dict[int, str]:
    """
    Returns the GPT-2 byte-to-unicode mapping.

    This maps bytes (0-255) to unicode characters, with control characters and
    some bytes mapped to unicode characters in the range 256+.
    This is the standard encoding used by GPT-2, GPT-4, Qwen, Mistral, etc.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _unicode_to_bytes() -> dict[str, int]:
    """Returns the reverse mapping: unicode characters back to bytes."""
    return {v: k for k, v in _bytes_to_unicode().items()}


# Module-level cache for the byte decoder
_BYTE_DECODER: dict[str, int] | None = None


def _get_byte_decoder() -> dict[str, int]:
    """Get the cached byte decoder."""
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        _BYTE_DECODER = _unicode_to_bytes()
    return _BYTE_DECODER


class TikTokenizer:
    """
    Convert HuggingFace tokenizers to tiktoken format.

    This class provides class methods to create and load tiktoken Encoding
    objects from compatible HuggingFace tokenizers.

    Example:
        # Create from HuggingFace model (auto-caches)
        encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")

        # Load from cache
        encoding = TikTokenizer.load("Qwen/Qwen3-0.6B")

        # Check compatibility before creating
        if TikTokenizer.is_compatible("meta-llama/Llama-2-7b"):
            encoding = TikTokenizer.create("meta-llama/Llama-2-7b")
    """

    DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "tiktokenizer"

    @classmethod
    def create(
        cls,
        model_name: str,
        cache_dir: str | Path | None = None,
        *,
        override: bool = False
    ) -> tiktoken.Encoding:
        """
        Create a tiktoken Encoding from a HuggingFace model.

        Downloads the tokenizer from HuggingFace, converts it to tiktoken format,
        and caches the result for faster subsequent loads.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-0.6B").
            cache_dir: Directory to cache the converted encoding.
                       Defaults to ~/.cache/tiktokenizer/
            override: Override existing encoder and create a new one
                      Defaults to False

        Returns:
            tiktoken.Encoding object ready for use.

        Raises:
            FileExistsError: If the encoder already exists and override is set to False
            IncompatibleTokenizerError: If the model's tokenizer is not compatible.

        Example:
            encoding = TikTokenizer.create("mistralai/Mistral-7B-v0.1")
            tokens = encoding.encode("Hello, world!")
        """
        # Set up cache directory
        if cache_dir is None:
            cache_dir = cls.DEFAULT_CACHE_DIR
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename from model name
        cache_filename = cls._sanitize_model_name(model_name) + ".json"
        cache_path = cache_dir / cache_filename

        # Check if already cached
        if cache_path.exists():
            if not override:
                raise FileExistsError(f"Tokenizer for model {model_name} already exists (set override=True to create a new one)")

        # Load HuggingFace tokenizer
        tokenizer = cls._load_hf_tokenizer(model_name)

        # Verify compatibility
        if not cls._check_compatibility(tokenizer):
            raise IncompatibleTokenizerError(
                f"Tokenizer for '{model_name}' is not compatible with tiktoken conversion. "
                "Only byte-level BPE tokenizers (GPT-2/GPT-4 style) are supported. "
                "Incompatible types include: SentencePiece, WordPiece, Unigram."
            )

        # Convert to tiktoken format
        mergeable_ranks = cls._convert_vocab_to_mergeable_ranks(tokenizer)
        pat_str = cls._extract_pattern(tokenizer)
        special_tokens = cls._extract_special_tokens(tokenizer)

        # Create encoding
        encoding_name = cls._sanitize_model_name(model_name)
        encoding = tiktoken.Encoding(
            name=encoding_name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        # Cache the encoding
        cls._save_encoding(encoding, cache_path, pat_str)

        return encoding

    @classmethod
    def load(cls,        
             model_name: str,
             cache_dir: str | Path | None = None,
             ) -> tiktoken.Encoding:
        """
        Load a tiktoken Encoding from a cached file.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-0.6B").
            cache_dir: Directory to cache the converted encoding.
                       Defaults to ~/.cache/tiktokenizer/

        Returns:
            tiktoken.Encoding loaded from the file.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.

        Example:
            encoding = TikTokenizer.load("Qwen/Qwen3-0.6B")
        """
        if cache_dir is None:
            cache_dir = cls.DEFAULT_CACHE_DIR
        cache_dir = Path(cache_dir)
        cache_filename = cls._sanitize_model_name(model_name) + ".json"

        path = Path(cache_dir / cache_filename).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer for model {model_name} not found (have you created it?)")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct mergeable_ranks from base64
        mergeable_ranks: dict[bytes, int] = {}
        for b64_token, rank in data["mergeable_ranks"].items():
            token_bytes = base64.b64decode(b64_token)
            mergeable_ranks[token_bytes] = rank

        # Reconstruct special tokens
        special_tokens: dict[str, int] = data["special_tokens"]

        return tiktoken.Encoding(
            name=data["name"],
            pat_str=data["pat_str"],
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    @classmethod
    def is_compatible(cls, model_name: str) -> bool:
        """
        Check if a HuggingFace model's tokenizer is compatible with tiktoken conversion.

        Args:
            model_name: HuggingFace model name or path.

        Returns:
            True if the tokenizer can be converted, False otherwise.

        Example:
            if TikTokenizer.is_compatible("Qwen/Qwen3-0.6B"):
                encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")
        """
        try:
            tokenizer = cls._load_hf_tokenizer(model_name)
            return cls._check_compatibility(tokenizer)
        except Exception:
            return False

    @classmethod
    def get_cache_path(cls, model_name: str, cache_dir: str | Path | None = None) -> Path:
        """
        Get the cache file path for a model.

        Args:
            model_name: HuggingFace model name or path.
            cache_dir: Cache directory. Defaults to ~/.cache/tiktokenizer/

        Returns:
            Path to where the cache file would be stored.
        """
        if cache_dir is None:
            cache_dir = cls.DEFAULT_CACHE_DIR
        cache_dir = Path(cache_dir)
        cache_filename = cls._sanitize_model_name(model_name) + ".json"
        return cache_dir / cache_filename

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Convert model name to a safe filename."""
        return model_name.replace("/", "--").replace("\\", "--")

    @staticmethod
    def _load_hf_tokenizer(model_name: str) -> "PreTrainedTokenizerFast":
        """Load a HuggingFace tokenizer."""
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @staticmethod
    def _check_compatibility(tokenizer: "PreTrainedTokenizerFast") -> bool:
        """
        Check if a tokenizer uses byte-level BPE compatible with tiktoken.

        Checks for:
        1. BPE model type
        2. ByteLevel pre-tokenizer (GPT-2 style byte encoding)
        """
        try:
            backend = tokenizer.backend_tokenizer

            # Check model type is BPE
            model = backend.model
            model_type = type(model).__name__
            if "BPE" not in model_type:
                return False

            # Check for ByteLevel pre-tokenizer
            pre_tokenizer = backend.pre_tokenizer
            if pre_tokenizer is None:
                return False

            # Serialize pre_tokenizer to check for ByteLevel
            try:
                pre_tok_bytes = pre_tokenizer.__getstate__()
                pre_tok_str = (
                    pre_tok_bytes.decode("utf-8")
                    if isinstance(pre_tok_bytes, bytes)
                    else str(pre_tok_bytes)
                )
            except (AttributeError, TypeError):
                pre_tok_str = str(pre_tokenizer)

            # ByteLevel must be present in the pre-tokenizer chain
            if "ByteLevel" not in pre_tok_str:
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def _convert_vocab_to_mergeable_ranks(
        tokenizer: "PreTrainedTokenizerFast",
    ) -> dict[bytes, int]:
        """
        Convert HuggingFace vocab to tiktoken mergeable_ranks format.

        The HF tokenizer uses GPT-2 style byte-level encoding where:
        - Printable ASCII chars map to themselves
        - Control chars and other bytes map to unicode chars 256+
        - e.g., space (0x20) -> 'Ġ' (288), newline (0x0a) -> 'Ċ' (266)
        """
        vocab = tokenizer.get_vocab()
        mergeable_ranks: dict[bytes, int] = {}
        byte_decoder = _get_byte_decoder()

        # Get special tokens to exclude
        special_tokens = set(tokenizer.all_special_tokens)

        for token, rank in vocab.items():
            if token in special_tokens:
                continue

            try:
                token_bytes = bytes([byte_decoder[c] for c in token])
                mergeable_ranks[token_bytes] = rank
            except KeyError:
                # Token contains characters not in the byte decoder - skip
                continue

        return mergeable_ranks

    @staticmethod
    def _extract_pattern(tokenizer: "PreTrainedTokenizerFast") -> str:
        """Extract the pre-tokenization regex pattern from the tokenizer."""
        try:
            backend = tokenizer.backend_tokenizer
            pre_tokenizer = backend.pre_tokenizer

            if hasattr(pre_tokenizer, "pattern"):
                return pre_tokenizer.pattern

            # Extract from serialized form
            try:
                pre_tok_bytes = pre_tokenizer.__getstate__()
                pre_tok_dict = json.loads(pre_tok_bytes)
            except (AttributeError, TypeError):
                pre_tok_dict = json.loads(pre_tokenizer.to_str())

            # Handle Sequence type (most common)
            if pre_tok_dict.get("type") == "Sequence":
                for item in pre_tok_dict.get("pretokenizers", []):
                    if item.get("type") == "Split" and "pattern" in item:
                        pattern_info = item["pattern"]
                        if isinstance(pattern_info, dict) and "Regex" in pattern_info:
                            return pattern_info["Regex"]

            # Handle direct Split type
            elif pre_tok_dict.get("type") == "Split":
                pattern_info = pre_tok_dict.get("pattern", {})
                if isinstance(pattern_info, dict) and "Regex" in pattern_info:
                    return pattern_info["Regex"]

        except Exception:
            pass

        # Fallback: GPT-4 style pattern (works for most byte-level BPE models)
        return (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            r"[^\r\n\p{L}\p{N}]?\p{L}+|"
            r"\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|"
            r"\s+(?!\S)|"
            r"\s+"
        )

    @staticmethod
    def _extract_special_tokens(
        tokenizer: "PreTrainedTokenizerFast",
    ) -> dict[str, int]:
        """Extract special tokens mapping."""
        special_tokens: dict[str, int] = {}

        for token in tokenizer.all_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int):
                special_tokens[token] = token_id

        # Include additional special tokens from added_tokens
        if hasattr(tokenizer, "added_tokens_encoder"):
            for token, token_id in tokenizer.added_tokens_encoder.items():
                if token not in special_tokens:
                    special_tokens[token] = token_id

        return special_tokens

    @staticmethod
    def _save_encoding(
        encoding: tiktoken.Encoding,
        path: Path,
        pat_str: str,
    ) -> None:
        """Save encoding to a JSON file."""
        data = {
            "name": encoding.name,
            "pat_str": pat_str,
            "mergeable_ranks": {},
            "special_tokens": {},
        }

        # Store mergeable ranks as base64
        for token_bytes, rank in encoding._mergeable_ranks.items():
            b64_token = base64.b64encode(token_bytes).decode("ascii")
            data["mergeable_ranks"][b64_token] = rank

        # Store special tokens
        for token_str, token_id in encoding._special_tokens.items():
            data["special_tokens"][token_str] = token_id

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

