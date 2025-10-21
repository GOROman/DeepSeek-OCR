#!/usr/bin/env python3
"""
Fix DeepSeek-OCR for MPS compatibility
This script patches the cached model files to remove Flash Attention dependencies
"""

import os
import sys
from pathlib import Path
import shutil


def find_model_cache():
    """Find the Hugging Face cache directory for DeepSeek-OCR."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"

    # Look for DeepSeek-OCR in cache
    pattern = "deepseek*ai*DeepSeek*OCR"

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

    print(f"\nFixing: {file_path}")

    # Backup original
    backup_file(file_path)

    # Read file
    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # Fix 1: Remove LlamaFlashAttention2 import
    if "from transformers.models.llama.modeling_llama import" in content:
        print("  - Removing LlamaFlashAttention2 import...")

        # Replace the import line
        content = content.replace(
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)"
        )

        # Also handle variations
        content = content.replace(
            "LlamaFlashAttention2,",
            ""
        )
        content = content.replace(
            "    LlamaFlashAttention2",
            ""
        )

    # Fix 2: Replace attention layer mappings
    if "LlamaFlashAttention2" in content:
        print("  - Replacing LlamaFlashAttention2 references...")
        content = content.replace(
            '"flash_attention_2": LlamaFlashAttention2,',
            '"flash_attention_2": LlamaSdpaAttention,  # Use SDPA instead of Flash'
        )
        content = content.replace(
            'LlamaFlashAttention2',
            'LlamaSdpaAttention'
        )

    # Fix 3: Handle _attn_implementation
    if '"flash_attention_2"' in content and 'self._attn_implementation' in content:
        print("  - Fixing attention implementation checks...")
        content = content.replace(
            'if self._attn_implementation == "flash_attention_2"',
            'if self._attn_implementation == "sdpa"'
        )

    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✓ Fixed: {file_path.name}")
        return True
    else:
        print(f"  ℹ No changes needed")
        return False


def fix_modeling_deepseekocr(file_path):
    """Fix modeling_deepseekocr.py if needed."""

    if not file_path.exists():
        return False

    print(f"\nChecking: {file_path}")

    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # Check if it imports flash_attn
    if 'flash_attn' in content.lower():
        print("  - Removing flash_attn imports...")

        backup_file(file_path)

        # Remove flash_attn imports
        lines = content.split('\n')
        new_lines = []

        for line in lines:
            if 'from flash_attn' in line or 'import flash_attn' in line:
                new_lines.append('# ' + line + '  # Disabled for MPS compatibility')
            else:
                new_lines.append(line)

        content = '\n'.join(new_lines)

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"  ✓ Fixed: {file_path.name}")
            return True

    print(f"  ℹ No changes needed")
    return False


def main():
    """Main fix function."""

    print("="*60)
    print("DeepSeek-OCR MPS Compatibility Fix")
    print("="*60)

    # Find cached model files
    print("\nSearching for cached model files...")
    cache_dirs = find_model_cache()

    if not cache_dirs:
        print("❌ No cached DeepSeek-OCR model found")
        print("\nThe model needs to be downloaded first. Try running:")
        print("  python test_ocr.py your_image.png")
        print("\nThis will download the model, then run this fix script again.")
        sys.exit(1)

    print(f"✓ Found {len(cache_dirs)} cached model location(s)")

    # Fix each cached version
    fixed_any = False

    for cache_dir in cache_dirs:
        print(f"\nProcessing: {cache_dir}")

        # Find version subdirectories
        for version_dir in cache_dir.iterdir():
            if not version_dir.is_dir():
                continue

            print(f"\n  Version: {version_dir.name}")

            # Fix modeling_deepseekv2.py
            v2_file = version_dir / "modeling_deepseekv2.py"
            if v2_file.exists():
                if fix_modeling_deepseekv2(v2_file):
                    fixed_any = True
            else:
                print(f"  ⚠ Not found: modeling_deepseekv2.py")

            # Fix modeling_deepseekocr.py
            ocr_file = version_dir / "modeling_deepseekocr.py"
            if ocr_file.exists():
                if fix_modeling_deepseekocr(ocr_file):
                    fixed_any = True

    # Summary
    print("\n" + "="*60)
    if fixed_any:
        print("✓ Model files have been patched for MPS compatibility")
        print("\nYou can now run:")
        print("  python test_ocr.py your_image.png")
        print("  python ocr.py your_image.png")
    else:
        print("ℹ All files are already compatible or no changes were needed")
    print("="*60)

    # Show backup location
    print("\nNote: Original files have been backed up with .backup extension")
    print("If something goes wrong, you can restore them or delete the cache:")
    print(f"  rm -rf {Path.home()}/.cache/huggingface/modules/transformers_modules/deepseek*")


if __name__ == "__main__":
    main()
