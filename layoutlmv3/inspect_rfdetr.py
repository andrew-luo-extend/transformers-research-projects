#!/usr/bin/env python
"""Quick script to inspect RF-DETR model structure and methods."""

from rfdetr import RFDETRMedium

model = RFDETRMedium()

print("RF-DETR Model Attributes:")
print("=" * 60)
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\n" + "=" * 60)
print("Checking for common loading methods:")
print("=" * 60)
print(f"  hasattr(model, 'load_state_dict'): {hasattr(model, 'load_state_dict')}")
print(f"  hasattr(model, 'model'): {hasattr(model, 'model')}")
print(f"  hasattr(model, 'load'): {hasattr(model, 'load')}")
print(f"  hasattr(model, 'load_weights'): {hasattr(model, 'load_weights')}")
print(f"  hasattr(model, 'from_pretrained'): {hasattr(model, 'from_pretrained')}")

if hasattr(model, 'model'):
    print(f"\n  model.model type: {type(model.model)}")
    print(f"  hasattr(model.model, 'load_state_dict'): {hasattr(model.model, 'load_state_dict')}")

print("\n" + "=" * 60)
print("Model type and structure:")
print("=" * 60)
print(f"  Type: {type(model)}")
print(f"  MRO: {[c.__name__ for c in type(model).__mro__]}")
