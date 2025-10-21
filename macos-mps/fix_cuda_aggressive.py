#!/usr/bin/env python3
"""
Aggressive fix for .cuda() calls in DeepSeek-OCR model code
Replaces .cuda() with device-agnostic code
"""

import shutil
from pathlib import Path
import re


def fix_cuda_calls(file_path):
    """Fix .cuda() calls in the file."""

    if not file_path.exists():
        return False

    print(f"\nFixing: {file_path.name}")

    # Backup
    backup_path = Path(str(file_path) + ".backup4")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up to: {backup_path.name}")

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    fixed = []

    # Fix 1: .cuda() -> comment out or replace
    # Find all .cuda() calls
    cuda_calls = re.findall(r'\.cuda\(\)', content)
    if cuda_calls:
        print(f"  - Found {len(cuda_calls)} .cuda() calls")

        # Strategy: Remove .cuda() calls entirely
        # Most of the time, the device is already set via .to(device)
        content = re.sub(r'\.cuda\(\)', '', content)
        fixed.append(f"Removed {len(cuda_calls)} .cuda() calls")

    # Fix 2: torch.cuda.amp.autocast -> torch.autocast or remove
    if 'torch.cuda.amp.autocast' in content:
        print("  - Found torch.cuda.amp.autocast")
        content = content.replace('torch.cuda.amp.autocast', 'torch.autocast')
        fixed.append("Replaced torch.cuda.amp.autocast with torch.autocast")

    # Fix 3: device_type='cuda' in autocast
    if "device_type='cuda'" in content or 'device_type="cuda"' in content:
        print("  - Found device_type='cuda'")
        content = content.replace("device_type='cuda'", "device_type='cpu'")
        content = content.replace('device_type="cuda"', 'device_type="cpu"')
        fixed.append("Changed device_type from 'cuda' to 'cpu'")

    # Fix 4: with torch.autocast('cuda') -> with torch.autocast('cpu')
    if re.search(r"torch\.autocast\(['\"]cuda['\"]", content):
        print("  - Found torch.autocast('cuda')")
        content = re.sub(r"torch\.autocast\(['\"]cuda['\"]", "torch.autocast('cpu'", content)
        fixed.append("Changed torch.autocast('cuda') to torch.autocast('cpu')")

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
        print("  ℹ No fixes needed")
        return False


def main():
    print("=" * 60)
    print("Aggressive CUDA Fix for DeepSeek-OCR")
    print("=" * 60)
    print("\nThis will:")
    print("  - Remove all .cuda() calls")
    print("  - Replace torch.cuda.amp.autocast with torch.autocast")
    print("  - Change device_type='cuda' to device_type='cpu'")
    print()

    base_cache = Path.home() / ".cache" / "huggingface"

    files_to_fix = [
        base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py",
        base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py",
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if fix_cuda_calls(file_path):
            fixed_count += 1

    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} file(s)")
        print("\n🎉 Now try:")
        print("  python ocr.py image.png")
    else:
        print("ℹ No fixes needed")
    print("=" * 60)


if __name__ == "__main__":
    main()
