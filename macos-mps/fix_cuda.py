#!/usr/bin/env python3
"""
Fix CUDA references in DeepSeek-OCR model code
The model tries to use torch.autocast with device_type='cuda'
"""

import shutil
from pathlib import Path
import re


def fix_cuda_references(file_path):
    """Fix CUDA device_type references in model files."""

    if not file_path.exists():
        return False

    print(f"\nChecking: {file_path.name}")

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    changes = []

    # Fix 1: autocast with device_type='cuda'
    if "device_type='cuda'" in content or 'device_type="cuda"' in content:
        print("  - Found CUDA device_type in autocast")
        # Replace with cpu or remove device_type
        content = content.replace("device_type='cuda'", "device_type='cpu'")
        content = content.replace('device_type="cuda"', 'device_type="cpu"')
        changes.append("autocast device_type")

    # Fix 2: .cuda() calls
    if re.search(r'\.cuda\(\)', content):
        print("  - Found .cuda() calls")
        # These are usually followed by dtype conversion
        # Replace .cuda() with .to(self.device) or similar
        # But be careful not to break things
        changes.append("Found .cuda() calls (not auto-fixed, may need manual review)")

    # Fix 3: torch.cuda specific calls
    if 'torch.cuda' in content:
        print("  - Found torch.cuda references")
        changes.append("Found torch.cuda references")

    if changes:
        print(f"  Changes needed: {', '.join(changes)}")

        # Backup
        backup_path = Path(str(file_path) + ".backup3")
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"  ✓ Backed up to: {backup_path.name}")

        # Save
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            print("  ✓ Fixed and saved")

            # Clear cache
            pycache = file_path.parent / "__pycache__"
            if pycache.exists():
                for pyc_file in pycache.glob("*.pyc"):
                    pyc_file.unlink()
                print("  ✓ Cleared Python cache")

            return True
        else:
            print("  ℹ No automatic fixes applied")
            return False
    else:
        print("  ✓ No CUDA references found")
        return False


def main():
    print("=" * 60)
    print("Fix CUDA References in DeepSeek-OCR")
    print("=" * 60)

    base_cache = Path.home() / ".cache" / "huggingface"

    # Check both modeling files
    files_to_check = [
        (
            "transformers_modules",
            base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py"
        ),
        (
            "transformers_modules",
            base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"
        ),
        (
            "hub snapshots",
            base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekocr.py"
        ),
        (
            "hub snapshots",
            base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py"
        ),
    ]

    fixed_count = 0

    for location, file_path in files_to_check:
        print(f"\n[{location}]")
        if fix_cuda_references(file_path):
            fixed_count += 1

    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} file(s)")
    else:
        print("ℹ No automatic fixes applied")

    print("\nNow try:")
    print("  python ocr.py image.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
