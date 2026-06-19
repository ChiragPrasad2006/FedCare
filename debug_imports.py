#!/usr/bin/env python3
import sys

print("Python path:")
for p in sys.path[:3]:
    print(f"  {p}")

print("\nTesting TensorFlow imports:")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported: {type(tf)}")
    print(f"   Has keras: {hasattr(tf, 'keras')}")
    print(f"   Has __version__: {hasattr(tf, '__version__')}")
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    from tensorflow import keras
    print(f"✅ Keras imported from TensorFlow")
except Exception as e:
    print(f"❌ Keras import failed: {e}")
    try:
        import keras
        print(f"✅ Keras imported standalone")
    except Exception as e2:
        print(f"❌ Standalone keras also failed: {e2}")

try:
    from shared import create_simple_model
    print(f"✅ shared.models imported successfully")
except Exception as e:
    print(f"❌ shared.models import failed: {e}")
