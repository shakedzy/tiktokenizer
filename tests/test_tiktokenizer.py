"""Tests for TikTokenizer."""

import json
import pytest
import tempfile
from pathlib import Path
from .core import TikTokenizer, IncompatibleTokenizerError


# Use a small, fast model for testing
TEST_MODEL = "openai-community/gpt2"


class TestTikTokenizer:
    """Tests for the TikTokenizer class."""

    def test_is_compatible_with_gpt2(self):
        """GPT-2 should be compatible."""
        assert TikTokenizer.is_compatible(TEST_MODEL) is True

    def test_is_compatible_with_incompatible_model(self):
        """Mistral (SentencePiece) should not be compatible."""
        assert TikTokenizer.is_compatible("mistralai/Mistral-7B-v0.1") is False

    def test_create_encoding(self):
        """Test creating an encoding from a HuggingFace model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            encoding = TikTokenizer.create(TEST_MODEL, cache_dir=tmpdir)

            assert encoding is not None
            assert encoding.name == "openai-community--gpt2"
            assert encoding.n_vocab > 0

    def test_create_caches_to_file(self):
        """Test that create() saves a cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            TikTokenizer.create(TEST_MODEL, cache_dir=tmpdir)

            cache_path = Path(tmpdir) / "openai-community--gpt2.json"
            assert cache_path.exists()

            # Verify it's valid JSON
            with open(cache_path) as f:
                data = json.load(f)
            assert "name" in data
            assert "mergeable_ranks" in data
            assert "special_tokens" in data

    def test_create_uses_cache_on_second_call(self):
        """Test that create() uses cache on subsequent calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First call creates the cache
            enc1 = TikTokenizer.create(TEST_MODEL, cache_dir=tmpdir)

            # Second call should load from cache
            enc2 = TikTokenizer.create(TEST_MODEL, cache_dir=tmpdir)

            assert enc1.name == enc2.name
            assert enc1.n_vocab == enc2.n_vocab

    def test_create_raises_for_incompatible_model(self):
        """Test that create() raises for incompatible models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(IncompatibleTokenizerError):
                TikTokenizer.create("mistralai/Mistral-7B-v0.1", cache_dir=tmpdir)

    def test_load_encoding(self):
        """Test loading an encoding from a cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First create and cache
            TikTokenizer.create(TEST_MODEL, cache_dir=tmpdir)

            # Then load from cache
            cache_path = Path(tmpdir) / "openai-community--gpt2.json"
            encoding = TikTokenizer.load(cache_path)

            assert encoding is not None
            assert encoding.name == "openai-community--gpt2"

    def test_load_raises_for_missing_file(self):
        """Test that load() raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            TikTokenizer.load("/nonexistent/path/file.json")

    def test_get_cache_path(self):
        """Test get_cache_path returns correct path."""
        path = TikTokenizer.get_cache_path("org/model-name")
        assert path.name == "org--model-name.json"

        path_custom = TikTokenizer.get_cache_path("org/model", cache_dir="/custom")
        assert path_custom == Path("/custom/org--model.json")


class TestEncoding:
    """Tests for the tiktoken encoding functionality."""

    @pytest.fixture
    def encoding(self, tmp_path):
        """Create a test encoding."""
        return TikTokenizer.create(TEST_MODEL, cache_dir=tmp_path)

    def test_encode_simple_text(self, encoding):
        """Test encoding simple text."""
        tokens = encoding.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_decode_tokens(self, encoding):
        """Test decoding tokens back to text."""
        text = "Hello, world!"
        tokens = encoding.encode(text)
        decoded = encoding.decode(tokens)
        assert decoded == text

    def test_roundtrip_unicode(self, encoding):
        """Test roundtrip with unicode text."""
        text = "Hello 你好 🎉"
        tokens = encoding.encode(text)
        decoded = encoding.decode(tokens)
        assert decoded == text

    def test_empty_string(self, encoding):
        """Test encoding empty string."""
        tokens = encoding.encode("")
        assert tokens == []

    def test_matches_huggingface(self, encoding):
        """Test that encoding matches HuggingFace tokenizer."""
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)

        test_texts = [
            "Hello, world!",
            "The quick brown fox",
            "def hello():\n    print('hi')",
            "   spaces   ",
        ]

        for text in test_texts:
            hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
            tk_tokens = list(encoding.encode(text))
            assert hf_tokens == tk_tokens, f"Mismatch for: {text!r}"


class TestCLI:
    """Tests for the CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        from tiktokenizer.cli import main

        # --help causes SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_check_compatible(self):
        """Test CLI check command with compatible model."""
        from tiktokenizer.cli import main

        result = main(["check", TEST_MODEL])
        assert result == 0

    def test_cli_check_incompatible(self):
        """Test CLI check command with incompatible model."""
        from tiktokenizer.cli import main

        result = main(["check", "mistralai/Mistral-7B-v0.1"])
        assert result == 1

    def test_cli_create(self, tmp_path):
        """Test CLI create command."""
        from tiktokenizer.cli import main

        result = main(["create", TEST_MODEL, "--cache-dir", str(tmp_path)])
        assert result == 0

        # Verify cache file was created
        cache_file = tmp_path / "openai-community--gpt2.json"
        assert cache_file.exists()

    def test_cli_load(self, tmp_path):
        """Test CLI load command."""
        from tiktokenizer.cli import main

        # First create
        main(["create", TEST_MODEL, "--cache-dir", str(tmp_path)])

        # Then load
        cache_file = tmp_path / "openai-community--gpt2.json"
        result = main(["load", str(cache_file)])
        assert result == 0

    def test_cli_load_missing_file(self):
        """Test CLI load command with missing file."""
        from tiktokenizer.cli import main

        result = main(["load", "/nonexistent/file.json"])
        assert result == 1
