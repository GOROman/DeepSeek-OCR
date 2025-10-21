#!/usr/bin/env python3
"""
Fix DeepSeek-OCR cached files based on actual cache structure
"""

import shutil
from pathlib import Path


def backup_and_fix(file_path):
    """Backup and fix a modeling file."""

    if not file_path.exists():
        return False

    print(f"\nFixing: {file_path}")

    # Backup
    backup_path = Path(str(file_path) + ".backup")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up to: {backup_path.name}")

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    # Check if needs fixing
    if "LlamaFlashAttention2" not in content:
        print("  ℹ Already fixed (no LlamaFlashAttention2)")
        return False

    print("  - Removing LlamaFlashAttention2...")

    # Fix imports
    content = content.replace(
        "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
        "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)"
    )

    # Remove from other import formats
    content = content.replace("LlamaFlashAttention2,\n", "")
    content = content.replace(",\n    LlamaFlashAttention2", "")
    content = content.replace("    LlamaFlashAttention2,", "")
    content = content.replace("LlamaFlashAttention2", "LlamaSdpaAttention")

    # Save
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print("  ✓ Fixed and saved!")

        # Delete compiled Python cache
        pycache = file_path.parent / "__pycache__" / f"{file_path.stem}.cpython-312.pyc"
        if pycache.exists():
            pycache.unlink()
            print("  ✓ Cleared Python cache")

        return True
    else:
        print("  ℹ No changes needed")
        return False


def main():
    print("=" * 60)
    print("DeepSeek-OCR Fix (Based on Your Cache Structure)")
    print("=" * 60)

    base_cache = Path.home() / ".cache" / "huggingface"

    # Location 1: transformers_modules (this is what gets imported)
    file1 = base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"

    # Location 2: hub snapshots (original source)
    file2 = base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"

    files_to_fix = [file1, file2]

    fixed_count = 0

    for file_path in files_to_fix:
        if backup_and_fix(file_path):
            fixed_count += 1

    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} file(s)")
        print("\nNow you can run:")
        print("  python test_ocr.py your_image.png")
        print("  python ocr.py your_image.png")
    else:
        print("✓ All files already compatible")
        print("\nYou should be able to run:")
        print("  python test_ocr.py your_image.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
