"""
Command-line interface for TikTokenizer.

Usage:
    tiktokenizer create <model_name> [--cache-dir <path>]
    tiktokenizer load <cache_file> --test <text>
    tiktokenizer check <model_name>
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from . import TikTokenizer, IncompatibleTokenizerError, __version__


def cmd_create(args: argparse.Namespace) -> int:
    """Create a tiktoken encoding from a HuggingFace model."""
    model_name = args.model_name
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print(f"Creating tiktoken encoding for: {model_name}")

    if cache_dir:
        print(f"Cache directory: {cache_dir}")
    else:
        print(f"Cache directory: {TikTokenizer.DEFAULT_CACHE_DIR}")

    try:
        encoding = TikTokenizer.create(model_name, cache_dir=cache_dir)
    except IncompatibleTokenizerError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1

    cache_path = TikTokenizer.get_cache_path(model_name, cache_dir)

    print(f"\n✓ Encoding created successfully!")
    print(f"  Name: {encoding.name}")
    print(f"  Vocab size: {encoding.n_vocab:,}")
    print(f"  Cache file: {cache_path}")

    # Test encoding if requested
    if args.test:
        test_text = args.test
    else:
        test_text = "Hello, world!"

    tokens = encoding.encode(test_text)
    decoded = encoding.decode(tokens)

    print(f"\n  Test: '{test_text}'")
    print(f"  Tokens: {list(tokens)}")
    print(f"  Decoded: '{decoded}'")

    return 0


def cmd_load(args: argparse.Namespace) -> int:
    """Load a cached encoding and optionally test it."""
    model_name = args.model_name
    cache_dir = Path(args.cache_dir) if args.cache_dir else TikTokenizer.DEFAULT_CACHE_DIR
    cache_filename = TikTokenizer._sanitize_model_name(model_name) + ".json"
    cache_file = Path(cache_dir / cache_filename).expanduser()

    if not cache_file.exists():
        print(f"✗ Error: Cache file not found: {cache_file}", file=sys.stderr)
        return 1

    print(f"Loading encoding from: {cache_file}")

    try:
        encoding = TikTokenizer.load(cache_file)
    except Exception as e:
        print(f"\n✗ Error loading encoding: {e}", file=sys.stderr)
        return 1

    print(f"\n✓ Encoding loaded successfully!")
    print(f"  Name: {encoding.name}")
    print(f"  Vocab size: {encoding.n_vocab:,}")

    if args.test:
        tokens = encoding.encode(args.test)
        decoded = encoding.decode(tokens)
        print(f"\n  Test: '{args.test}'")
        print(f"  Tokens: {list(tokens)}")
        print(f"  Decoded: '{decoded}'")

    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check if a model's tokenizer is compatible."""
    model_name = args.model_name

    print(f"Checking compatibility for: {model_name}")

    is_compatible = TikTokenizer.is_compatible(model_name)

    if is_compatible:
        print(f"\n✓ {model_name} is compatible with TikTokenizer")
        return 0
    else:
        print(f"\n✗ {model_name} is NOT compatible with TikTokenizer")
        print("  Only byte-level BPE tokenizers (GPT-2/GPT-4 style) are supported.")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="tiktokenizer",
        description="Convert HuggingFace tokenizers to tiktoken format.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a tiktoken encoding from a HuggingFace model",
    )
    create_parser.add_argument(
        "model_name",
        help="HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')",
    )
    create_parser.add_argument(
        "--cache-dir",
        "-c",
        help="Directory to cache the encoding (default: ~/.cache/tiktokenizer/)",
    )
    create_parser.add_argument(
        "--test",
        "-t",
        help="Test text to encode after creation",
    )

    # Load command
    load_parser = subparsers.add_parser(
        "load",
        help="Load a cached tiktoken encoding",
    )
    create_parser.add_argument(
        "model_name",
        help="HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')",
    )
    create_parser.add_argument(
        "--cache-dir",
        "-c",
        help="Directory to cache the encoding (default: ~/.cache/tiktokenizer/)",
    )
    load_parser.add_argument(
        "--test",
        "-t",
        help="Test text to encode after loading",
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check if a model's tokenizer is compatible",
    )
    check_parser.add_argument(
        "model_name",
        help="HuggingFace model name to check",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "create":
        return cmd_create(args)
    elif args.command == "load":
        return cmd_load(args)
    elif args.command == "check":
        return cmd_check(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())