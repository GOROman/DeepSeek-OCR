#!/usr/bin/env python3
"""
Integrated Setup and Fix Script for DeepSeek-OCR on MacOS MPS

This script handles the chicken-and-egg problem:
1. Downloads the model code (tokenizer triggers code download)
2. Fixes the model code for MPS compatibility
3. Downloads the model weights
4. Verifies everything works

Usage:
    python setup_and_fix.py
"""

import os
import sys
from pathlib import Path
import shutil


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def find_model_cache():
    """Find the Hugging Face cache directory for DeepSeek-OCR."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"

    candidates = []
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if "deepseek" in item.name.lower() and "ocr" in item.name.lower():
                candidates.append(item)

    return candidates


def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = Path(str(file_path) + ".backup")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up: {file_path.name}")
    return backup_path


def fix_modeling_deepseekv2(file_path):
    """Fix modeling_deepseekv2.py to remove Flash Attention imports."""

    print(f"\n  Fixing: {file_path.name}")

    # Backup original
    backup_file(file_path)

    # Read file
    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # Fix: Remove LlamaFlashAttention2 import and references
    if "LlamaFlashAttention2" in content:
        print("    - Removing LlamaFlashAttention2...")

        # Remove from imports
        content = content.replace(
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)"
        )

        # Handle other import variations
        content = content.replace("LlamaFlashAttention2,", "")
        content = content.replace("    LlamaFlashAttention2", "")

        # Replace usage
        content = content.replace(
            '"flash_attention_2": LlamaFlashAttention2,',
            '"flash_attention_2": LlamaSdpaAttention,  # Use SDPA for MPS'
        )
        content = content.replace('LlamaFlashAttention2', 'LlamaSdpaAttention')

    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"    ✓ Fixed!")
        return True
    else:
        print(f"    ℹ Already compatible")
        return False


def download_tokenizer_only():
    """Download just the tokenizer to trigger code download."""
    print_header("Step 1: Downloading Model Code")

    try:
        from transformers import AutoTokenizer

        print("Downloading tokenizer and model code...")
        print("(This triggers the model code download)")

        model_name = 'deepseek-ai/DeepSeek-OCR'
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("✓ Model code downloaded")
        return True

    except Exception as e:
        print(f"⚠ Error during download: {e}")
        print("This is expected if Flash Attention code is already cached")
        return True  # Continue anyway


def fix_cached_code():
    """Fix the cached model code."""
    print_header("Step 2: Fixing Model Code for MPS")

    cache_dirs = find_model_cache()

    if not cache_dirs:
        print("❌ Could not find cached model code")
        return False

    print(f"Found {len(cache_dirs)} cached location(s)")

    fixed_any = False

    for cache_dir in cache_dirs:
        print(f"\nProcessing: {cache_dir.name}")

        for version_dir in cache_dir.iterdir():
            if not version_dir.is_dir():
                continue

            # Fix modeling_deepseekv2.py
            v2_file = version_dir / "modeling_deepseekv2.py"
            if v2_file.exists():
                if fix_modeling_deepseekv2(v2_file):
                    fixed_any = True
            else:
                # Check if there's a different file structure
                print(f"  Looking in: {version_dir}")
                for py_file in version_dir.glob("*.py"):
                    print(f"    Found: {py_file.name}")

    if fixed_any:
        print("\n✓ Model code has been fixed for MPS compatibility")
    else:
        print("\n✓ Model code is already compatible")

    return True


def download_full_model():
    """Download the full model."""
    print_header("Step 3: Downloading Full Model")

    try:
        from transformers import AutoModel
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Target device: {device}")

        print("\nDownloading model weights...")
        print("(This may take a few minutes, ~3GB)")

        model_name = 'deepseek-ai/DeepSeek-OCR'
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',  # Force SDPA instead of flash attention
        )

        print("✓ Model downloaded successfully")

        # Try to move to device
        print(f"\nTesting device compatibility ({device})...")
        model = model.to(device)
        print("✓ Model loaded to device successfully!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf the error persists, try:")
        print("  1. Clear cache: rm -rf ~/.cache/huggingface/hub/models--deepseek-ai*")
        print("  2. Run this script again")
        return False


def verify_installation():
    """Verify the installation works."""
    print_header("Step 4: Verification")

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading model on {device}...")

        model_name = 'deepseek-ai/DeepSeek-OCR'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',
        ).to(device)

        print("✓ Everything working!")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main function."""

    print("=" * 60)
    print("  DeepSeek-OCR for MacOS MPS")
    print("  Integrated Setup and Fix")
    print("=" * 60)

    print("\nThis script will:")
    print("  1. Download model code")
    print("  2. Fix Flash Attention compatibility")
    print("  3. Download model weights (~3GB)")
    print("  4. Verify everything works")

    response = input("\nContinue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Setup cancelled")
        sys.exit(0)

    # Step 1: Download tokenizer (triggers code download)
    if not download_tokenizer_only():
        print("\n❌ Setup failed at step 1")
        sys.exit(1)

    # Step 2: Fix the code
    if not fix_cached_code():
        print("\n❌ Setup failed at step 2")
        sys.exit(1)

    # Step 3: Download full model
    if not download_full_model():
        print("\n❌ Setup failed at step 3")
        print("\nTry running fix_model.py manually and then retry")
        sys.exit(1)

    # Step 4: Verify
    if not verify_installation():
        print("\n❌ Setup failed at step 4")
        sys.exit(1)

    # Success!
    print_header("Setup Complete!")
    print("✓ DeepSeek-OCR is ready to use on MacOS MPS")
    print("\nNext steps:")
    print("  1. Try the test script:")
    print("     python test_ocr.py your_image.png")
    print("\n  2. Or use the CLI tool:")
    print("     python ocr.py image.jpg")
    print("     python ocr.py --markdown document.pdf")
    print("\n  3. Or try the interactive demo:")
    print("     python demo.py")
    print("\nFor more examples, see:")
    print("  - README.md")
    print("  - QUICK_START_JP.md")
    print("  - example_usage.py")


if __name__ == "__main__":
    main()
