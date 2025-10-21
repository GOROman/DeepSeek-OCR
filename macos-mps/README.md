# DeepSeek-OCR for MacOS (Apple Silicon)

**Run DeepSeek-OCR on your Mac without CUDA or Docker**

Simplified implementation inspired by [Simon Willison's approach](https://simonwillison.net/2025/Oct/20/deepseek-ocr-claude-code/), optimized for MacOS with Apple Silicon (M1/M2/M3).

## Features

- ✅ **Native MPS Support** - Uses Apple's Metal Performance Shaders for GPU acceleration
- ✅ **No CUDA Required** - Works on MacOS without any NVIDIA dependencies
- ✅ **Simple Setup** - One command to get started
- ✅ **CLI Tool** - Easy-to-use command-line interface
- ✅ **Batch Processing** - Process multiple images at once
- ✅ **Multiple Modes** - Support for documents, figures, and free OCR

## Quick Start

### 1. Run Setup

```bash
cd macos-mps
python3 setup.py
```

This will:
- ✓ Check your Python version and MPS availability
- ✓ Install PyTorch with MPS support
- ✓ Install required dependencies
- ✓ Download the DeepSeek-OCR model (~3GB)
- ✓ Create a test script

### 2. Process Your First Image

```bash
# Simple OCR
python ocr.py image.jpg

# Convert document to Markdown
python ocr.py --markdown document.pdf

# Parse a figure
python ocr.py --figure chart.png
```

Results will be saved to `./output/`

### ⚠️ First Run Issue?

If you see an error like `ImportError: cannot import name 'LlamaFlashAttention2'`:

```bash
# Run the fix script
python fix_model.py

# Then try again
python ocr.py image.jpg
```

See [QUICK_FIX.md](QUICK_FIX.md) for details or [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more help.

## Usage

### CLI Tool

The `ocr.py` script provides a simple command-line interface:

```bash
# Basic usage
python ocr.py <image>

# Convert to Markdown
python ocr.py --markdown document.pdf

# Process multiple files
python ocr.py *.jpg --batch

# High quality mode
python ocr.py --large important_doc.jpg

# Fast mode
python ocr.py --tiny screenshot.png

# Custom prompt
python ocr.py --prompt "Extract all text and tables" document.png

# Describe image
python ocr.py --describe photo.jpg
```

**Options:**

- `--markdown` - Convert document to Markdown
- `--figure` - Parse figure/chart
- `--describe` - Get detailed image description
- `--free` - Free OCR without layout preservation
- `--prompt TEXT` - Use custom prompt
- `--tiny` - Tiny mode (512×512, fastest)
- `--small` - Small mode (640×640)
- `--large` - Large mode (1280×1280, best quality)
- `--batch` - Process multiple files
- `-o DIR` - Output directory (default: ./output)
- `-v` - Verbose output
- `--cpu` - Force CPU mode

### Python API

```python
from ocr import DeepSeekOCR
from pathlib import Path

# Initialize
ocr = DeepSeekOCR()  # Automatically uses MPS if available

# Process image
result = ocr.process(
    image_path=Path('document.jpg'),
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    output_dir=Path('./output'),
    base_size=1024,
    image_size=640,
    crop_mode=True
)

print(result)
```

### Direct Transformers API

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
)

# Run inference
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>OCR this image.",
    image_file='image.jpg',
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True
)
```

## Resolution Modes

| Mode | Command | Size | Tokens | Use Case |
|------|---------|------|--------|----------|
| Tiny | `--tiny` | 512×512 | 64 | Screenshots, simple text |
| Small | `--small` | 640×640 | 100 | Small documents |
| Base | *(default)* | 1024×1024 | 256 | Standard documents |
| Large | `--large` | 1280×1280 | 400 | High-quality documents |
| Gundam | *(default with crop)* | Dynamic | Variable | Large documents |

## Common Prompts

```python
# Document conversion
"<image>\n<|grounding|>Convert the document to markdown."

# General OCR
"<image>\n<|grounding|>OCR this image."

# Figure parsing
"<image>\nParse the figure."

# Free OCR (no layout)
"<image>\nFree OCR."

# Image description
"<image>\nDescribe this image in detail."

# Table extraction
"<image>\n<|grounding|>Extract all tables from this document."
```

## Batch Processing Examples

### Process all PDFs in a directory

```bash
python ocr.py --markdown --batch pdfs/*.pdf
```

### Process all images with custom output

```bash
python ocr.py --batch images/*.{jpg,png} -o results/
```

### High-quality batch processing

```bash
python ocr.py --large --batch documents/*.jpg
```

## Performance

Tested on M1 Max (32GB):

| Mode | Image Size | Time | Memory |
|------|-----------|------|--------|
| Tiny | 512×512 | ~1.5s | ~2GB |
| Small | 640×640 | ~2s | ~2.5GB |
| Base | 1024×1024 | ~3s | ~3GB |
| Large | 1280×1280 | ~4.5s | ~4GB |
| Gundam | 2048×1536 | ~6s | ~5GB |

*Performance varies based on image complexity and Mac model*

## Requirements

- **MacOS**: 12.3+ (for MPS support)
- **Python**: 3.10+
- **Memory**: 8GB+ recommended (16GB for large documents)
- **Storage**: ~5GB for model and dependencies
- **Apple Silicon**: M1, M2, M3, or later

## Troubleshooting

### "MPS not available"

Make sure you have:
- MacOS 12.3 or later
- PyTorch installed with MPS support

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

If False, reinstall PyTorch:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "flash_attn" error

This means the model is trying to use Flash Attention. Solutions:

1. **Use the provided scripts** - They automatically handle this
2. **Modify model files** - See [Full MPS Guide](README_MPS.md)

### Out of memory

Try:
- Use `--tiny` or `--small` mode
- Close other applications
- Use `--cpu` mode (slower but uses less GPU memory)

### Slow performance

- Ensure MPS is enabled (not using `--cpu`)
- Check Activity Monitor for other GPU-intensive apps
- Try smaller resolution modes first

## File Structure

```
macos-mps/
├── README.md              # This file
├── setup.py              # Automated setup script
├── ocr.py                # CLI tool
├── test_ocr.py           # Simple test script (created by setup)
├── example_usage.py      # Python API examples
├── requirements_mps.txt  # Dependencies
├── QUICK_START_JP.md     # Japanese quick start
├── README_MPS.md         # Detailed technical docs
├── clip_mps.py          # CLIP vision encoder
├── sam_vary_mps.py      # SAM vision encoder
└── projector.py         # Vision-language projector
```

## Advanced Usage

### Custom Model Modifications

If you need to modify the model implementation:

1. Download model locally:
```bash
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

2. Modify files as needed (e.g., remove flash_attn)

3. Load from local directory:
```python
model = AutoModel.from_pretrained(
    './DeepSeek-OCR',
    trust_remote_code=True
)
```

### Integration with Other Tools

```python
# Example: Process images from a web scraper
from ocr import DeepSeekOCR
from pathlib import Path
import requests

ocr = DeepSeekOCR()

def process_url(image_url):
    # Download image
    response = requests.get(image_url)
    image_path = Path('temp.jpg')
    image_path.write_bytes(response.content)

    # Process
    result = ocr.process(
        image_path=image_path,
        prompt="<image>\n<|grounding|>OCR this image.",
        output_dir=Path('./output')
    )

    return result
```

## Resources

- 📖 [DeepSeek-OCR Paper](../DeepSeek_OCR_paper.pdf)
- 🤗 [Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- 💬 [GitHub Discussions](https://github.com/deepseek-ai/DeepSeek-OCR/discussions)
- 🌐 [DeepSeek Website](https://www.deepseek.com/)

## Acknowledgments

- DeepSeek AI for the amazing OCR model
- Simon Willison for implementation inspiration
- Apple for MPS support in PyTorch

## License

Follows the DeepSeek-OCR license. See main repository for details.
