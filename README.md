# 🤗→⏳ TikTokenizer

Convert HuggingFace tokenizers to [tiktoken](https://github.com/openai/tiktoken) format.

TikTokenizer allows you to use any compatible HuggingFace tokenizer with OpenAI's fast tiktoken library. It automatically handles the conversion from HuggingFace's tokenizer format to tiktoken's encoding format, with built-in caching for fast subsequent loads.

## Installation

Install from source:

```bash
git clone https://github.com/shakedzy/tiktokenizer.git
cd tiktokenizer
pip install -e .
```


## Quick Start

```python
from tiktokenizer import TikTokenizer

# Create a tiktoken encoding from any compatible HuggingFace model
encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")

# Use it like any tiktoken encoding
tokens = encoding.encode("Hello, world!")
text = encoding.decode(tokens)

print(tokens)  # [9707, 11, 1879, 0]
print(text)    # Hello, world!
```

## Usage

### Creating an Encoding

```python
from tiktokenizer import TikTokenizer

# Basic usage - caches to ~/.cache/tiktokenizer/
encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")

# Custom cache directory
encoding = TikTokenizer.create("Qwen/Qwen3-0.6B", cache_dir="./my-cache")
```

### Loading from Cache

```python
# Load a previously cached encoding (no HuggingFace download needed)
encoding = TikTokenizer.load("Qwen/Qwen3-0.6B")
```

### Checking Compatibility

```python
# Check if a model is compatible before attempting conversion
if TikTokenizer.is_compatible("some-model/name"):
    encoding = TikTokenizer.create("some-model/name")
else:
    print("Model uses incompatible tokenizer")
```

### Special Tokens

```python
# Encode text containing special tokens
encoding = TikTokenizer.create("Qwen/Qwen3-0.6B")

text = "<|im_start|>user\nHello!<|im_end|>"
tokens = encoding.encode(text, allowed_special="all")
```

## Compatible Models

TikTokenizer works with models that use **byte-level BPE** tokenizers (GPT-2/GPT-4 style):

| Model Family | Example | Compatible |
|--------------|---------|------------|
| Qwen | `Qwen/Qwen3-0.6B` | ✅ |
| GPT-2 | `openai-community/gpt2` | ✅ |
| Phi | `microsoft/phi-2` | ✅ |
| Mistral | `mistralai/Mistral-7B-v0.1` | ❌ |
| LLaMA | `meta-llama/Llama-2-7b` | ❌ |
| BERT | `bert-base-uncased` | ❌ |

### Why Some Models Are Incompatible

TikTokenizer only supports byte-level BPE tokenizers that use the GPT-2 byte encoding scheme. Models using different tokenizer architectures are not compatible:

- **SentencePiece** (Mistral, LLaMA, T5): Uses `▁` for spaces, different byte encoding
- **WordPiece** (BERT, DistilBERT): Uses `##` subword prefixes
- **Unigram** (XLNet, ALBERT): Different algorithm entirely

## API Reference

### `TikTokenizer.create(model_name, cache_dir=None)`

Create a tiktoken `Encoding` from a HuggingFace model.

**Parameters:**
- `model_name` (str): HuggingFace model name or path
- `cache_dir` (str | Path | None): Cache directory. Defaults to `~/.cache/tiktokenizer/`

**Returns:** `tiktoken.Encoding`

**Raises:** 
- `FileExistsError`: If the encoder already exists and `override=False`
- `IncompatibleTokenizerError` if the model's tokenizer is not compatible

---

### `TikTokenizer.load(model_name, cache_dir=None)`

Load a tiktoken `Encoding` from a cached file.

**Parameters:**
- `model_name` (str): HuggingFace model name or path
- `cache_dir` (str | Path | None): Cache directory. Defaults to `~/.cache/tiktokenizer/`

**Returns:** `tiktoken.Encoding`

**Raises:** `FileNotFoundError` if the cache file doesn't exist

---

### `TikTokenizer.is_compatible(model_name)`

Check if a HuggingFace model's tokenizer can be converted.

**Parameters:**
- `model_name` (str): HuggingFace model name or path

**Returns:** `bool`

## How It Works

1. **Load HuggingFace tokenizer** using `transformers.AutoTokenizer`
2. **Check compatibility** by verifying the tokenizer uses byte-level BPE with ByteLevel pre-tokenizer
3. **Convert vocabulary** from HuggingFace's string format to tiktoken's bytes format using the GPT-2 byte-to-unicode mapping
4. **Extract regex pattern** for pre-tokenization from the tokenizer config
5. **Extract special tokens** and map them to their IDs
6. **Create tiktoken.Encoding** with the converted data
7. **Cache to disk** as JSON for fast subsequent loads

## Command Line Interface

After installation, the `tiktokenizer` command is available globally:

### Create an encoding

```bash
# Create and cache a tiktoken encoding from a HuggingFace model
tiktokenizer create Qwen/Qwen3-0.6B

# Create with custom cache directory
tiktokenizer create Qwen/Qwen3-0.6B --cache-dir ./my-cache

# Create and test with custom text
tiktokenizer create Qwen/Qwen3-0.6B --test "Hello, world!"
```

### Load a cached encoding

```bash
# Load and display info about a cached encoding
tiktokenizer load Qwen/Qwen3-0.6B

# Load and test with text
tiktokenizer load Qwen/Qwen3-0.6B --test "Test text"
```

### Check compatibility

```bash
# Check if a model is compatible before creating
tiktokenizer check Qwen/Qwen3-0.6B
# ✓ Qwen/Qwen3-0.6B is compatible with TikTokenizer

tiktokenizer check mistralai/Mistral-7B-v0.1
# ✗ mistralai/Mistral-7B-v0.1 is NOT compatible with TikTokenizer
```

### Using as a module

```bash
# You can also run as a Python module
python -m tiktokenizer create Qwen/Qwen3-0.6B
```

## Why Use This?

- **Speed**: tiktoken is significantly faster than HuggingFace tokenizers
- **Simplicity**: Single-file encoding, no need for HuggingFace at runtime after caching
- **Compatibility**: Works anywhere tiktoken works
- **Offline**: Once cached, no internet connection needed

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
