#!/usr/bin/env python3
"""
Test script to verify the ML strategy fix.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.ml.strategy_ml import MLStrategy

def test_model_loading():
    """Test loading a model file to verify the fix."""
    # Try to load one of the existing model files
    model_files = list(Path("ml_models").glob("*.pkl"))
    
    if not model_files:
        print("❌ No model files found in ml_models directory")
        return False
    
    # Try loading the first model file
    model_path = model_files[0]
    print(f"Testing model loading for: {model_path.name}")
    
    try:
        # This should now provide better error messages if something is wrong
        strategy = MLStrategy(
            model_path=str(model_path),
            confidence_threshold=0.5,
            min_signal_strength="слабое",
            stability_filter=True
        )
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(strategy.model)}")
        print(f"   Feature names count: {len(strategy.feature_names)}")
        return True
    except KeyError as e:
        print(f"❌ KeyError during model loading: {e}")
        print("   This indicates the model file is missing required data.")
        return False
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError during model loading: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ML Strategy Fix...")
    print("=" * 50)
    
    success = test_model_loading()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)