#!/usr/bin/env python3
"""
Debug script to find where the model cache actually is
"""

from pathlib import Path
import os

print("Checking cache locations...\n")

# Check Hugging Face cache
hf_home = os.environ.get('HF_HOME', Path.home() / ".cache" / "huggingface")
print(f"HF_HOME: {hf_home}")

cache_locations = [
    Path.home() / ".cache" / "huggingface",
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "huggingface" / "modules",
    Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
]

for loc in cache_locations:
    print(f"\n{loc}")
    if loc.exists():
        print("  ✓ EXISTS")
        if loc.is_dir():
            items = list(loc.iterdir())
            print(f"  Contains {len(items)} items:")
            for item in items[:10]:  # Show first 10
                print(f"    - {item.name}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
    else:
        print("  ✗ Does not exist")

# Search for deepseek
print("\n" + "="*60)
print("Searching for 'deepseek' in cache...")
print("="*60)

base_cache = Path.home() / ".cache" / "huggingface"
if base_cache.exists():
    for path in base_cache.rglob("*"):
        if "deepseek" in path.name.lower():
            print(f"\n✓ Found: {path}")
            if path.is_dir():
                print(f"  Type: Directory")
                items = list(path.iterdir())
                print(f"  Contains: {len(items)} items")
                for item in list(path.iterdir())[:5]:
                    print(f"    - {item.name}")
            else:
                print(f"  Type: File")
                print(f"  Size: {path.stat().st_size} bytes")

print("\n" + "="*60)
print("Search complete")
print("="*60)
