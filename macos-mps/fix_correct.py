#!/usr/bin/env python3
"""
Correct fix for DeepSeek-OCR - removes Flash Attention references completely
Instead of replacing with LlamaSdpaAttention, just use LlamaAttention
"""

import shutil
from pathlib import Path


def backup_and_fix(file_path):
    """Backup and fix a modeling file."""

    if not file_path.exists():
        return False

    print(f"\nFixing: {file_path}")

    # Backup
    backup_path = Path(str(file_path) + ".backup2")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up to: {backup_path.name}")

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    # Check if needs fixing
    if "LlamaFlashAttention2" not in content:
        print("  ℹ No LlamaFlashAttention2 found")
        # But check if LlamaSdpaAttention is there (from previous fix)
        if "LlamaSdpaAttention" in content:
            print("  ⚠ Found LlamaSdpaAttention from previous fix, will correct...")
        else:
            print("  ✓ Already correctly fixed")
            return False

    print("  - Applying correct fix...")

    # Strategy: Just use LlamaAttention for everything
    # Remove LlamaFlashAttention2 and LlamaSdpaAttention from imports

    # Fix 1: Clean import statement - just keep LlamaAttention
    import_patterns = [
        # Original with Flash
        (
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import LlamaAttention"
        ),
        # After first fix attempt
        (
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import LlamaAttention"
        ),
    ]

    for old, new in import_patterns:
        if old in content:
            content = content.replace(old, new)
            print(f"    - Fixed import statement")

    # Fix 2: Remove any remaining Flash/Sdpa references
    content = content.replace("LlamaFlashAttention2", "LlamaAttention")
    content = content.replace("LlamaSdpaAttention", "LlamaAttention")

    # Fix 3: Update attention layer mapping dictionary
    # Look for patterns like: "flash_attention_2": LlamaFlashAttention2,
    lines = content.split('\n')
    new_lines = []
    in_attention_dict = False

    for line in lines:
        # Check if we're in the attention classes mapping
        if 'LLAMA_ATTENTION_CLASSES' in line or '"eager": LlamaAttention' in line:
            in_attention_dict = True

        # Skip or replace flash_attention_2 line
        if '"flash_attention_2"' in line and in_attention_dict:
            # Comment it out instead of removing
            new_lines.append('    # ' + line.strip() + '  # Disabled for MPS compatibility')
            continue

        # Skip sdpa if it references non-existent class
        if '"sdpa"' in line and 'LlamaSdpaAttention' in line:
            new_lines.append('    # ' + line.strip() + '  # Disabled for MPS compatibility')
            continue

        new_lines.append(line)

    content = '\n'.join(new_lines)

    # Save
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print("  ✓ Fixed and saved!")

        # Delete compiled Python cache
        pycache = file_path.parent / "__pycache__"
        if pycache.exists():
            for pyc_file in pycache.glob("*.pyc"):
                pyc_file.unlink()
            print("  ✓ Cleared all Python cache files")

        return True
    else:
        print("  ℹ No changes needed")
        return False


def main():
    print("=" * 60)
    print("DeepSeek-OCR Correct Fix")
    print("=" * 60)

    base_cache = Path.home() / ".cache" / "huggingface"

    # Location 1: transformers_modules (this is what gets imported)
    file1 = base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"

    # Location 2: hub snapshots (original source)
    file2 = base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"

    files_to_fix = [file1, file2]

    print("\nThis fix will:")
    print("  1. Remove all Flash Attention references")
    print("  2. Use only standard LlamaAttention")
    print("  3. Clear Python cache")

    fixed_count = 0

    for file_path in files_to_fix:
        if backup_and_fix(file_path):
            fixed_count += 1

    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} file(s)")
        print("\nNow try:")
        print("  python ocr.py image.png")
    else:
        print("✓ All files already compatible")
    print("=" * 60)


if __name__ == "__main__":
    main()
