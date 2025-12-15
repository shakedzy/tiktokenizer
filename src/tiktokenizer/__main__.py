"""
Allow running the package as a module: python -m tiktokenizer
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

