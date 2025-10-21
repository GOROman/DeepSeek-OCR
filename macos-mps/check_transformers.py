#!/usr/bin/env python3
"""
Check what's actually available in transformers.models.llama.modeling_llama
"""

import transformers

print(f"Transformers version: {transformers.__version__}")
print("\nAvailable in transformers.models.llama.modeling_llama:")

try:
    from transformers.models.llama import modeling_llama

    # List all classes
    classes = [name for name in dir(modeling_llama) if not name.startswith('_')]

    attention_classes = [name for name in classes if 'Attention' in name]

    print("\nAttention classes found:")
    for name in attention_classes:
        print(f"  - {name}")

    # Check specific ones
    print("\nChecking specific classes:")

    names_to_check = [
        'LlamaAttention',
        'LlamaSdpaAttention',
        'LlamaFlashAttention2',
        'LlamaSdpAttention',
    ]

    for name in names_to_check:
        if hasattr(modeling_llama, name):
            print(f"  ✓ {name} - Available")
        else:
            print(f"  ✗ {name} - Not available")

except Exception as e:
    print(f"Error: {e}")
