#!/usr/bin/env python3
"""
Check transformers version and provide compatibility information
"""

import sys

print("Checking environment...")
print("=" * 60)

# Python version
print(f"Python: {sys.version}")

# Transformers version
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")

# PyTorch version
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

# Check DynamicCache
try:
    from transformers import DynamicCache
    cache = DynamicCache()

    print("\nDynamicCache attributes:")
    attrs = [attr for attr in dir(cache) if not attr.startswith('_')]
    for attr in attrs[:10]:
        print(f"  - {attr}")

    # Check for specific attributes
    has_seen_tokens = hasattr(cache, 'seen_tokens')
    has_get_seq_length = hasattr(cache, 'get_seq_length')

    print(f"\nHas 'seen_tokens': {has_seen_tokens}")
    print(f"Has 'get_seq_length': {has_get_seq_length}")

except Exception as e:
    print(f"\nError checking DynamicCache: {e}")

print("\n" + "=" * 60)
print("Recommendation:")

# Check version
try:
    import transformers
    version = transformers.__version__
    major, minor = version.split('.')[:2]

    if int(minor) < 46:
        print(f"⚠ Your transformers version ({version}) may be too old")
        print("Try updating:")
        print("  pip install --upgrade transformers")
    elif int(minor) >= 50:
        print(f"✓ Your transformers version ({version}) should work")
        print("The issue might be in the model code compatibility")
    else:
        print(f"Your transformers version: {version}")
        print("Recommended: 4.46.x - 4.49.x")
except:
    pass

print("=" * 60)
