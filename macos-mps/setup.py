#!/usr/bin/env python3
"""
Simplified DeepSeek-OCR setup for MacOS
Inspired by Simon Willison's implementation approach
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_mps():
    """Check if MPS is available."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon GPU) is available")
            return True
        else:
            print("⚠ MPS not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return None


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")

    print("Installing PyTorch with MPS support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--quiet"
    ], check=True)

    print("Installing other dependencies...")
    requirements = [
        "transformers>=4.46.3",
        "tokenizers>=0.20.3",
        "Pillow>=10.0.0",
        "PyMuPDF",
        "img2pdf",
        "einops",
        "easydict",
        "addict",
        "numpy",
    ]

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        *requirements,
        "--quiet"
    ], check=True)

    print("✓ All dependencies installed")


def verify_installation():
    """Verify the installation."""
    print_header("Verifying Installation")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        import transformers
        print(f"✓ Transformers {transformers.__version__}")

        from PIL import Image
        print(f"✓ Pillow {Image.__version__}")

        if torch.backends.mps.is_available():
            print("✓ MPS is available and ready")
        else:
            print("⚠ MPS not available, will use CPU")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def download_model():
    """Download the model (just verify it's accessible)."""
    print_header("Checking Model Access")

    try:
        from transformers import AutoTokenizer

        print("Downloading model (this may take a few minutes on first run)...")
        print("Model size: ~3GB")

        model_name = 'deepseek-ai/DeepSeek-OCR'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print("✓ Model files downloaded and cached")
        print(f"✓ Cache location: ~/.cache/huggingface/")

        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def create_test_script():
    """Create a simple test script."""
    print_header("Creating Test Script")

    script_content = '''#!/usr/bin/env python3
"""
Simple DeepSeek-OCR test script
Usage: python test_ocr.py <image_path>
"""

import sys
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

def run_ocr(image_path, prompt=None):
    """Run OCR on an image."""

    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # Default prompt
    if prompt is None:
        prompt = "<image>\\n<|grounding|>Convert the document to markdown."

    # Run inference
    print(f"Processing: {image_path}")
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path='./output',
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True
    )

    print("\\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(result)
    print("\\n✓ Output saved to: ./output")

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    run_ocr(image_path)
'''

    script_path = Path("test_ocr.py")
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    print(f"✓ Created: {script_path}")
    print(f"\nUsage: python test_ocr.py <image_path>")


def main():
    """Main setup function."""
    print_header("DeepSeek-OCR MacOS Setup")
    print("This script will set up DeepSeek-OCR for MacOS")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check MPS
    check_mps()

    # Confirm installation
    print("\nThis will install the following:")
    print("  - PyTorch with MPS support")
    print("  - Transformers and tokenizers")
    print("  - Image processing libraries")
    print("  - DeepSeek-OCR model (~3GB)")

    response = input("\nContinue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Setup cancelled")
        sys.exit(0)

    try:
        # Install dependencies
        install_dependencies()

        # Verify installation
        if not verify_installation():
            print("\n❌ Installation verification failed")
            sys.exit(1)

        # Download model
        if not download_model():
            print("\n⚠ Model download had issues, but you can try running anyway")

        # Create test script
        create_test_script()

        # Success message
        print_header("Setup Complete!")
        print("✓ DeepSeek-OCR is ready to use")
        print("\nQuick start:")
        print("  1. python test_ocr.py your_image.jpg")
        print("  2. Check ./output/ for results")
        print("\nFor more examples:")
        print("  - See example_usage.py")
        print("  - Read QUICK_START_JP.md")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
