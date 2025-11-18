#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Test script to verify ReDimNet integration
"""

import torch
import sys

def test_redimnet_import():
    """Test if ReDimNet model can be imported"""
    try:
        from models.ReDimNet import MainModel
        print("✓ ReDimNet module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import ReDimNet: {e}")
        return False

def test_redimnet_model_creation():
    """Test if ReDimNet model can be instantiated"""
    try:
        from models.ReDimNet import MainModel

        print("\nTesting ReDimNet model creation...")
        model = MainModel(
            redimnet_model='b2',
            redimnet_train_type='ptn',
            redimnet_dataset='vox2',
            redimnet_pretrained=True,
            num_out=192
        )
        print(f"✓ ReDimNet model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params / 1e6:.2f}M")

        return True
    except Exception as e:
        print(f"✗ Failed to create ReDimNet model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_redimnet_forward():
    """Test if ReDimNet forward pass works"""
    try:
        from models.ReDimNet import MainModel

        print("\nTesting ReDimNet forward pass...")
        model = MainModel(
            redimnet_model='b2',
            redimnet_train_type='ptn',
            redimnet_dataset='vox2',
            redimnet_pretrained=True,
            num_out=192
        )

        # Create dummy input (batch_size=2, 3 seconds of audio at 16kHz)
        dummy_input = torch.randn(2, 48000)

        # Forward pass
        model.eval()
        with torch.no_grad():
            embeddings = model(dummy_input, aug=False)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected shape: [2, 192]")

        # Verify output shape
        assert embeddings.shape == (2, 192), f"Unexpected output shape: {embeddings.shape}"
        print("✓ Output shape verified")

        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("ReDimNet Integration Test")
    print("=" * 60)

    results = []

    # Test 1: Import
    results.append(("Import", test_redimnet_import()))

    # Test 2: Model creation
    if results[0][1]:
        results.append(("Model Creation", test_redimnet_model_creation()))

    # Test 3: Forward pass
    if len(results) > 1 and results[1][1]:
        results.append(("Forward Pass", test_redimnet_forward()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("All tests passed! ✓")
        print("ReDimNet is ready to use with SASV baseline.")
    else:
        print("Some tests failed. ✗")
        print("Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
