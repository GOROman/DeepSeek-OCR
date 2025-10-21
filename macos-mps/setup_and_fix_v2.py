#!/usr/bin/env python3
"""
Improved Integrated Setup and Fix Script for DeepSeek-OCR on MacOS MPS
More robust cache detection and better error handling
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


def find_all_deepseek_files():
    """Find all DeepSeek-related files in cache."""
    base_cache = Path.home() / ".cache" / "huggingface"

    deepseek_files = {
        'modeling_files': [],
        'config_files': [],
        'all_dirs': []
    }

    if not base_cache.exists():
        return deepseek_files

    # Search recursively for deepseek files
    for path in base_cache.rglob("*deepseek*"):
        if path.is_dir():
            deepseek_files['all_dirs'].append(path)
        elif path.name.endswith('.py'):
            if 'modeling' in path.name:
                deepseek_files['modeling_files'].append(path)
            elif 'config' in path.name:
                deepseek_files['config_files'].append(path)

    return deepseek_files


def find_modeling_files():
    """Find modeling_deepseekv2.py files specifically."""
    base_cache = Path.home() / ".cache" / "huggingface"
    modeling_files = []

    if not base_cache.exists():
        return modeling_files

    # Search for modeling_deepseekv2.py
    for path in base_cache.rglob("modeling_deepseekv2.py"):
        modeling_files.append(path)

    # Also search for any modeling file in deepseek directories
    for path in base_cache.rglob("*"):
        if path.is_file() and path.name.startswith("modeling_") and "deepseek" in str(path).lower():
            if path not in modeling_files:
                modeling_files.append(path)

    return modeling_files


def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = Path(str(file_path) + ".backup")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up: {file_path.name}")
    return backup_path


def fix_modeling_file(file_path):
    """Fix a modeling file to remove Flash Attention imports."""

    print(f"\n  Fixing: {file_path}")
    print(f"  Location: {file_path.parent}")

    try:
        # Backup original
        backup_file(file_path)

        # Read file
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Check if file needs fixing
        if "LlamaFlashAttention2" not in content:
            print("    ℹ Already compatible (no LlamaFlashAttention2 found)")
            return False

        print("    - Found LlamaFlashAttention2 references")

        # Fix: Remove LlamaFlashAttention2 import
        content = content.replace(
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)"
        )

        # Handle variations
        content = content.replace("LlamaFlashAttention2,\n", "")
        content = content.replace(",\n    LlamaFlashAttention2", "")
        content = content.replace("    LlamaFlashAttention2,", "")

        # Replace usage
        content = content.replace(
            '"flash_attention_2": LlamaFlashAttention2,',
            '"flash_attention_2": LlamaSdpaAttention,  # MPS compatible'
        )
        content = content.replace('LlamaFlashAttention2', 'LlamaSdpaAttention')

        # Write back
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"    ✓ Fixed and saved")
            return True
        else:
            print(f"    ℹ No changes made")
            return False

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def download_tokenizer_only():
    """Download just the tokenizer to trigger code download."""
    print_header("Step 1: Downloading Model Code")

    try:
        # Set environment to force download
        os.environ['TRANSFORMERS_OFFLINE'] = '0'

        from transformers import AutoTokenizer

        print("Downloading tokenizer and model code...")

        model_name = 'deepseek-ai/DeepSeek-OCR'
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=False,  # Use cache if available
        )

        print("✓ Tokenizer loaded")

        # Show where files are cached
        cache_dir = Path.home() / ".cache" / "huggingface"
        print(f"\nCache directory: {cache_dir}")

        return True

    except Exception as e:
        print(f"⚠ Note: {e}")
        print("Continuing anyway (code may already be cached)...")
        return True


def fix_cached_code():
    """Fix the cached model code."""
    print_header("Step 2: Fixing Model Code for MPS")

    print("Searching for modeling files...")

    # First, show what we found
    all_files = find_all_deepseek_files()

    if all_files['all_dirs']:
        print(f"\nFound {len(all_files['all_dirs'])} DeepSeek directories:")
        for d in all_files['all_dirs'][:5]:
            print(f"  - {d}")

    if all_files['modeling_files']:
        print(f"\nFound {len(all_files['modeling_files'])} modeling files:")
        for f in all_files['modeling_files']:
            print(f"  - {f}")

    # Find specific modeling files
    modeling_files = find_modeling_files()

    if not modeling_files:
        print("\n❌ Could not find modeling_deepseekv2.py")
        print("\nDebug: Let's check what's in the cache...")

        base_cache = Path.home() / ".cache" / "huggingface"
        if base_cache.exists():
            print(f"\nCache exists at: {base_cache}")

            # Check modules directory
            modules_dir = base_cache / "modules" / "transformers_modules"
            if modules_dir.exists():
                print(f"\nTransformers modules directory exists:")
                for item in modules_dir.iterdir():
                    print(f"  - {item.name}")
                    if item.is_dir():
                        print(f"    Contains:")
                        for subitem in list(item.iterdir())[:10]:
                            print(f"      - {subitem.name}")

        print("\n💡 Tip: Run 'python debug_cache.py' to see full cache structure")
        return False

    print(f"\n✓ Found {len(modeling_files)} modeling file(s) to fix")

    fixed_any = False
    for file_path in modeling_files:
        if fix_modeling_file(file_path):
            fixed_any = True

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

        # Try with explicit attn_implementation
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',
            )
            print("✓ Model loaded with SDPA attention")
        except Exception as e:
            print(f"⚠ SDPA approach failed: {e}")
            print("Trying without explicit attention...")

            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            print("✓ Model loaded")

        # Try to move to device
        print(f"\nTesting device compatibility ({device})...")
        model = model.to(device)
        print("✓ Model loaded to device successfully!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis might be because:")
        print("  1. The model code still has Flash Attention references")
        print("  2. Network issue downloading model weights")
        print("\nTry:")
        print("  1. Run 'python debug_cache.py' to check cache")
        print("  2. Delete cache: rm -rf ~/.cache/huggingface/hub/models--deepseek-ai*")
        print("  3. Run this script again")
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

        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',
            ).to(device)
        except:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
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
    print("  Integrated Setup and Fix (v2)")
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

    # Step 1: Download tokenizer
    if not download_tokenizer_only():
        print("\n❌ Setup failed at step 1")
        sys.exit(1)

    # Step 2: Fix the code
    if not fix_cached_code():
        print("\n❌ Setup failed at step 2")
        print("\n💡 Debug tip: Run 'python debug_cache.py' to see cache structure")
        sys.exit(1)

    # Step 3: Download full model
    if not download_full_model():
        print("\n❌ Setup failed at step 3")
        sys.exit(1)

    # Step 4: Verify
    if not verify_installation():
        print("\n❌ Setup failed at step 4")
        sys.exit(1)

    # Success!
    print_header("Setup Complete!")
    print("✓ DeepSeek-OCR is ready to use on MacOS MPS")
    print("\nNext steps:")
    print("  python ocr.py image.jpg")
    print("  python ocr.py --markdown document.pdf")
    print("  python demo.py")


if __name__ == "__main__":
    main()
