#!/usr/bin/env python3
"""
Comprehensive fix script for DeepSeek-OCR on MacOS MPS
Applies all necessary patches in the correct order
"""

import shutil
from pathlib import Path
import re
import sys


def backup_file(file_path):
    """Create backup of file."""
    backup_path = Path(str(file_path) + ".backup_original")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backed up to: {backup_path.name}")
        return True
    return False


def clear_cache(file_path):
    """Clear Python bytecode cache."""
    pycache = file_path.parent / "__pycache__"
    if pycache.exists():
        count = 0
        for pyc_file in pycache.glob("*.pyc"):
            pyc_file.unlink()
            count += 1
        if count > 0:
            print(f"  ✓ Cleared {count} cache file(s)")
        return True
    return False


def fix_flash_attention(content):
    """Remove Flash Attention and use standard LlamaAttention."""
    fixes = []

    # Fix imports - use only standard LlamaAttention
    import_patterns = [
        (
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import LlamaAttention"
        ),
        (
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaSdpaAttention,\n)",
            "from transformers.models.llama.modeling_llama import LlamaAttention"
        ),
        (
            "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2,\n)",
            "from transformers.models.llama.modeling_llama import LlamaAttention"
        ),
    ]

    for old, new in import_patterns:
        if old in content:
            content = content.replace(old, new)
            fixes.append("Fixed import statement")
            break

    # Replace all Flash/Sdpa references with standard attention
    if "LlamaFlashAttention2" in content:
        content = content.replace("LlamaFlashAttention2", "LlamaAttention")
        fixes.append("Replaced LlamaFlashAttention2 with LlamaAttention")

    if "LlamaSdpaAttention" in content:
        content = content.replace("LlamaSdpaAttention", "LlamaAttention")
        fixes.append("Replaced LlamaSdpaAttention with LlamaAttention")

    # Comment out flash_attention_2 in attention classes mapping
    lines = content.split('\n')
    new_lines = []
    in_attention_dict = False

    for line in lines:
        if 'LLAMA_ATTENTION_CLASSES' in line or '"eager": LlamaAttention' in line:
            in_attention_dict = True

        if '"flash_attention_2"' in line and in_attention_dict:
            new_lines.append('    # ' + line.strip() + '  # Disabled for MPS compatibility')
            fixes.append("Disabled flash_attention_2 mapping")
            continue

        if '"sdpa"' in line and 'LlamaSdpaAttention' in line:
            new_lines.append('    # ' + line.strip() + '  # Disabled for MPS compatibility')
            fixes.append("Disabled sdpa mapping")
            continue

        new_lines.append(line)

    content = '\n'.join(new_lines)

    return content, fixes


def fix_cuda_references(content):
    """Remove CUDA-specific code."""
    fixes = []

    # Remove .cuda() calls
    cuda_calls = re.findall(r'\.cuda\(\)', content)
    if cuda_calls:
        content = re.sub(r'\.cuda\(\)', '', content)
        fixes.append(f"Removed {len(cuda_calls)} .cuda() call(s)")

    # Replace torch.cuda.amp.autocast with torch.autocast
    if 'torch.cuda.amp.autocast' in content:
        content = content.replace('torch.cuda.amp.autocast', 'torch.autocast')
        fixes.append("Replaced torch.cuda.amp.autocast with torch.autocast")

    # Change device_type='cuda' to device_type='cpu'
    if "device_type='cuda'" in content or 'device_type="cuda"' in content:
        content = content.replace("device_type='cuda'", "device_type='cpu'")
        content = content.replace('device_type="cuda"', 'device_type="cpu"')
        fixes.append("Changed device_type from 'cuda' to 'cpu'")

    # Change torch.autocast('cuda') to torch.autocast('cpu')
    if re.search(r"torch\.autocast\(['\"]cuda['\"]", content):
        content = re.sub(r"torch\.autocast\(['\"]cuda['\"]", "torch.autocast('cpu'", content)
        fixes.append("Changed torch.autocast('cuda') to torch.autocast('cpu')")

    return content, fixes


def fix_model_file(file_path):
    """Apply all fixes to a model file."""

    if not file_path.exists():
        return False, []

    print(f"\nProcessing: {file_path.name}")
    print(f"Path: {file_path}")

    # Backup
    backup_file(file_path)

    # Read
    content = file_path.read_text(encoding='utf-8')
    original = content

    all_fixes = []

    # Apply fixes
    content, flash_fixes = fix_flash_attention(content)
    all_fixes.extend(flash_fixes)

    content, cuda_fixes = fix_cuda_references(content)
    all_fixes.extend(cuda_fixes)

    # Save if changed
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✓ Applied {len(all_fixes)} fix(es):")
        for fix in all_fixes:
            print(f"    - {fix}")

        # Clear cache
        clear_cache(file_path)

        return True, all_fixes
    else:
        print("  ℹ No changes needed")
        return False, []


def main():
    print("=" * 70)
    print("DeepSeek-OCR Comprehensive Fix for MacOS MPS")
    print("=" * 70)
    print()
    print("This script will apply all necessary fixes:")
    print("  1. Remove Flash Attention (CUDA-only)")
    print("  2. Replace with standard LlamaAttention")
    print("  3. Remove .cuda() calls")
    print("  4. Fix autocast device_type references")
    print("  5. Clear Python bytecode cache")
    print()

    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")

        major, minor = version.split('.')[:2]
        if int(minor) < 46:
            print(f"⚠ Warning: transformers {version} may be too old")
            print("  Recommended: pip install transformers==4.46.3")
            print()
        elif int(minor) > 49:
            print(f"⚠ Warning: transformers {version} may be too new")
            print("  Recommended: pip install transformers==4.46.3")
            print()
        else:
            print(f"✓ Version looks good")
            print()
    except ImportError:
        print("⚠ Warning: transformers not installed")
        print()

    # Find cache directory
    base_cache = Path.home() / ".cache" / "huggingface"

    # Files to fix
    files_to_fix = [
        base_cache / "modules" / "transformers_modules" / "deepseek_hyphen_ai" / "DeepSeek_hyphen_OCR" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py",
        base_cache / "hub" / "models--deepseek-ai--DeepSeek-OCR" / "snapshots" / "59512895521e669adb87064b37e5749fe9b9f5d2" / "modeling_deepseekv2.py",
    ]

    # Check if files exist
    existing_files = [f for f in files_to_fix if f.exists()]

    if not existing_files:
        print("❌ No cached model files found!")
        print()
        print("Please run setup first:")
        print("  python setup.py")
        print()
        print("Or download the model:")
        print("  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)\"")
        print()
        sys.exit(1)

    print(f"Found {len(existing_files)} file(s) to fix")
    print()

    # Apply fixes
    fixed_count = 0
    total_fixes = []

    for file_path in existing_files:
        success, fixes = fix_model_file(file_path)
        if success:
            fixed_count += 1
            total_fixes.extend(fixes)

    # Summary
    print()
    print("=" * 70)
    if fixed_count > 0:
        print(f"✓ Successfully fixed {fixed_count} file(s)")
        print(f"  Total {len(total_fixes)} individual fix(es) applied")
        print()
        print("Next steps:")
        print("  1. Try running OCR:")
        print("     python ocr.py image.png")
        print()
        print("  2. If you get MPS device errors, use CPU mode:")
        print("     python ocr.py --cpu image.png")
        print()
        print("  3. For batch processing:")
        print("     python ocr.py --batch image1.png image2.png")
    else:
        print("✓ All files already fixed or no changes needed")
        print()
        print("You can now run:")
        print("  python ocr.py image.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
