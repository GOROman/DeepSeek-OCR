#!/usr/bin/env python3
"""
Fix DynamicCache compatibility issue in modeling_deepseekv2.py
The error 'DynamicCache' object has no attribute 'seen_tokens'
occurs with newer transformers versions
"""

import shutil
from pathlib import Path
import re


def fix_dynamiccache_issue(file_path):
    """Fix DynamicCache compatibility."""

    if not file_path.exists():
        return False

    print(f"\nFixing: {file_path.name}")

    # Backup
    backup_path = Path(str(file_path) + ".backup5")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up to: {backup_path.name}")

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    fixed = []

    # Fix 1: Replace .seen_tokens with .get_seq_length()
    if '.seen_tokens' in content:
        print("  - Found .seen_tokens usage")

        # Replace cache.seen_tokens with cache.get_seq_length(layer_idx)
        # This is a common pattern
        content = re.sub(
            r'(\w+)\.seen_tokens',
            r'\1.get_seq_length(0)',  # Use layer 0 as default
            content
        )
        fixed.append("Replaced .seen_tokens with .get_seq_length(0)")

    # Fix 2: Replace past_key_values.seen_tokens
    if 'past_key_values.seen_tokens' in content:
        content = content.replace(
            'past_key_values.seen_tokens',
            'past_key_values.get_seq_length(0)'
        )
        fixed.append("Fixed past_key_values.seen_tokens")

    # Fix 3: Look for other cache-related issues
    if 'cache.seen_tokens' in content:
        content = content.replace(
            'cache.seen_tokens',
            'cache.get_seq_length(0) if hasattr(cache, "get_seq_length") else 0'
        )
        fixed.append("Added fallback for cache.seen_tokens")

    if fixed:
        print(f"  Applied fixes:")
        for fix in fixed:
            print(f"    - {fix}")

        # Save
        file_path.write_text(content, encoding='utf-8')
        print("  ✓ Saved")

        # Clear cache
        pycache = file_path.parent / "__pycache__"
        if pycache.exists():
            for pyc_file in pycache.glob("*.pyc"):
                pyc_file.unlink()
            print("  ✓ Cleared Python cache")

        return True
    else:
        print("  ℹ No .seen_tokens found")
        return False


def main():
    print("=" * 60)
    print("Fix DynamicCache Compatibility")
    print("=" * 60)
    print("\nThis fixes the error:")
    print("  'DynamicCache' object has no attribute 'seen_tokens'")
    print()

    base_cache = Path.home() / ".cache" / "huggingface"

    files_to_fix = [
        base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py",
        base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py",
        base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py",
        base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py",
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if fix_dynamiccache_issue(file_path):
            fixed_count += 1

    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} file(s)")
        print("\n🎉 Now try:")
        print("  python ocr.py image.png")
    else:
        print("ℹ No fixes needed (or different attribute name)")
        print("\nIf the error persists, try:")
        print("  pip install transformers==4.46.3")
    print("=" * 60)


if __name__ == "__main__":
    main()
