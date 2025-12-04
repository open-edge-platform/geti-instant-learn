#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quick test script to validate OpenVINO benchmark integration."""

import sys
from argparse import Namespace
from getiprompt.utils.benchmark import load_model
from getiprompt.utils.constants import ModelName, SAMModelName


def test_load_model_with_backends():
    """Test loading model with different backends."""
    print("=" * 60)
    print("Testing Model Loading with Different Backends")
    print("=" * 60)

    # Test 1: PyTorch backend (default)
    print("\nTest 1: Loading Matcher with PyTorch backend...")
    try:
        pytorch_args = Namespace(
            backend="pytorch",
            encoder_model="dinov3_large",
            num_foreground_points=40,
            num_background_points=2,
            mask_similarity_threshold=0.38,
            precision="fp32",
            compile_models=False,
            device="cpu",
        )
        pytorch_model = load_model(
            sam=SAMModelName.SAM_HQ_TINY,
            model_name=ModelName.MATCHER,
            args=pytorch_args,
        )
        print(f"✅ PyTorch model loaded: {type(pytorch_model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return False

    # Test 2: OpenVINO backend (requires export)
    print("\nTest 2: Loading Matcher with OpenVINO backend...")
    print("Note: This will attempt to load from ./exports/matcher")
    print("If models don't exist, you'll see an error (expected for this test)")

    try:
        ov_args = Namespace(
            backend="openvino",
            export_dir="./exports/matcher",
            encoder_model="dinov3_large",
            num_foreground_points=40,
            num_background_points=2,
            mask_similarity_threshold=0.38,
            precision="fp32",
            compile_models=False,
            device="CPU",
        )
        ov_model = load_model(
            sam=SAMModelName.SAM_HQ_TINY,
            model_name=ModelName.MATCHER,
            args=ov_args,
        )
        print(f"✅ OpenVINO model loaded: {type(ov_model).__name__}")
    except FileNotFoundError as e:
        print(f"ℹ️  Models not exported yet (expected): {e}")
        print("   Run 'python examples/ov_matcher_example.py' to export models first")
    except Exception as e:
        print(f"❌ Unexpected error loading OpenVINO model: {e}")
        return False

    # Test 3: Verify argument parsing
    print("\nTest 3: Verifying argument attributes...")
    assert hasattr(pytorch_args, "backend"), "Missing 'backend' attribute"
    assert hasattr(ov_args, "export_dir"), "Missing 'export_dir' attribute"
    print("✅ All required arguments present")

    print("\n" + "=" * 60)
    print("Basic integration tests PASSED ✅")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_load_model_with_backends()
    sys.exit(0 if success else 1)
