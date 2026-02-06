#!/usr/bin/env python3
"""Test if SAM3 and MoGe2 can be imported correctly."""

import sys

# Add SAM3D Body path
sys.path.insert(0, "C:/Sam3dBody/sam-3d-body")

# Add SAM3 path (nested structure: sam3/sam3/sam3)
sys.path.insert(0, "C:/Sam3dBody/sam3/sam3")

print("Testing imports...")
print("=" * 60)

# Test SAM3
print("\n1. Testing SAM3 import...")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("   [OK] SAM3 imports successful")
except Exception as e:
    print(f"   [FAILED] SAM3 import error: {e}")

# Test MoGe2
print("\n2. Testing MoGe2 import...")
try:
    from moge.model.v2 import MoGeModel
    print("   [OK] MoGe2 imports successful")
except Exception as e:
    print(f"   [FAILED] MoGe2 import error: {e}")

# Test FOVEstimator
print("\n3. Testing FOVEstimator from SAM3D Body...")
try:
    from tools.build_fov_estimator import FOVEstimator
    print("   [OK] FOVEstimator import successful")
except Exception as e:
    print(f"   [FAILED] FOVEstimator import error: {e}")

# Test HumanDetector with sam3
print("\n4. Testing HumanDetector from SAM3D Body...")
try:
    from tools.build_detector import HumanDetector
    print("   [OK] HumanDetector import successful")
except Exception as e:
    print(f"   [FAILED] HumanDetector import error: {e}")

print("\n" + "=" * 60)
print("Import test complete.")
print("\nIf all tests pass, you can run the pipeline.")
print("If any test fails, that component needs to be installed.")
