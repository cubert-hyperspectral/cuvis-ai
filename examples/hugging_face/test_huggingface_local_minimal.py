"""Minimal test for HuggingFace Local Model (Phase 2)

This script tests core functionality without requiring dataset files:
- Model loading and lazy initialization
- Gradient passthrough verification
- Freeze/unfreeze functionality
- Performance measurement

Run:
    uv run python examples/test_huggingface_local_minimal.py
"""

import time

import torch
from loguru import logger

from cuvis_ai.node.adaclip import AdaCLIPLocalNode


def test_model_loading() -> AdaCLIPLocalNode:
    """Test that the model loads correctly."""
    logger.info("=== Test 1: Model Loading ===")

    # Use a small, publicly available CLIP model for testing
    node = AdaCLIPLocalNode(
        model_name="openai/clip-vit-base-patch32",
        cache_dir=None,
        default_text_prompt="stones",
        name="test_adaclip",
    )

    logger.info(f"Node created: {node.name}")
    logger.info(f"Model name: {node.model_name}")
    logger.info("Model not yet loaded (lazy loading)...")

    # Trigger model loading by accessing the model property
    logger.info("Accessing model to trigger loading...")
    model = node.model
    logger.success(f"✓ Model loaded successfully: {type(model).__name__}")

    # Check that model is frozen
    trainable_params = sum(p.requires_grad for p in model.parameters())
    total_params = sum(1 for _ in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    if trainable_params == 0:
        logger.success("✓ Model is correctly frozen (0 trainable parameters)")
    else:
        logger.warning(f"✗ Model has {trainable_params} trainable parameters (expected 0)")

    return node


def test_gradient_passthrough(node: AdaCLIPLocalNode) -> bool:
    """Test that gradients pass through frozen model to inputs."""
    logger.info("\n=== Test 2: Gradient Passthrough ===")

    # Create synthetic RGB image with gradient tracking
    batch_size = 2
    height, width = 224, 224

    rgb_image = torch.rand(batch_size, height, width, 3, requires_grad=True)
    logger.info(f"Created synthetic RGB image: {rgb_image.shape}")
    logger.info(f"Image requires_grad: {rgb_image.requires_grad}")

    # Forward pass
    logger.info("Running forward pass...")
    try:
        outputs = node.forward(image=rgb_image, text_prompt="stones")
        logger.success("✓ Forward pass successful")

        logger.info(f"Output keys: {list(outputs.keys())}")

        # Check outputs
        if "anomaly_mask" in outputs:
            mask = outputs["anomaly_mask"]
            logger.info(f"Anomaly mask shape: {mask.shape}, dtype: {mask.dtype}")

        if "anomaly_scores" in outputs:
            scores = outputs["anomaly_scores"]
            logger.info(f"Anomaly scores shape: {scores.shape}, dtype: {scores.dtype}")

            # Test gradient computation
            logger.info("Computing gradients...")
            loss = scores.mean()
            loss.backward()

            # Check if gradients reached the input
            input_has_grad = rgb_image.grad is not None
            if input_has_grad:
                logger.success("✓ Gradients successfully passed through to input!")
                logger.info(f"Input gradient shape: {rgb_image.grad.shape}")
                logger.info(f"Input gradient norm: {rgb_image.grad.norm():.6f}")
            else:
                logger.error("✗ No gradients on input (expected gradients)")

            # Check model parameters don't have gradients
            model_has_grad = any(
                p.grad is not None for p in node.model.parameters() if p.requires_grad
            )
            if not model_has_grad:
                logger.success("✓ Model parameters remain frozen (no gradients)")
            else:
                logger.warning("✗ Model parameters have gradients (should be frozen)")

            # Overall verdict
            if input_has_grad and not model_has_grad:
                logger.success("✓✓✓ GRADIENT PASSTHROUGH VERIFIED ✓✓✓")
                return True
            else:
                logger.error("✗✗✗ GRADIENT PASSTHROUGH FAILED ✗✗✗")
                return False

    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_freeze_unfreeze(node: AdaCLIPLocalNode) -> bool:
    """Test freeze/unfreeze functionality."""
    logger.info("\n=== Test 3: Freeze/Unfreeze ===")

    # Model should start frozen
    trainable = sum(p.requires_grad for p in node.model.parameters())
    logger.info(f"Initial trainable parameters: {trainable}")

    if trainable == 0:
        logger.success("✓ Model starts frozen")
    else:
        logger.warning(f"✗ Model has {trainable} trainable parameters")

    # Test unfreeze
    logger.info("Unfreezing model...")
    node.unfreeze()
    trainable_after_unfreeze = sum(p.requires_grad for p in node.model.parameters())
    logger.info(f"Trainable parameters after unfreeze: {trainable_after_unfreeze}")

    if trainable_after_unfreeze > 0:
        logger.success(f"✓ Model unfrozen ({trainable_after_unfreeze} trainable parameters)")
    else:
        logger.error("✗ Model still frozen after unfreeze()")

    # Test freeze again
    logger.info("Freezing model again...")
    node.freeze()
    trainable_after_refreeze = sum(p.requires_grad for p in node.model.parameters())
    logger.info(f"Trainable parameters after re-freeze: {trainable_after_refreeze}")

    if trainable_after_refreeze == 0:
        logger.success("✓ Model re-frozen successfully")
    else:
        logger.warning(f"✗ Model has {trainable_after_refreeze} trainable parameters")

    return trainable_after_refreeze == 0


def test_performance() -> tuple[float, float]:
    """Test inference performance."""
    logger.info("\n=== Test 4: Performance Measurement ===")

    node = AdaCLIPLocalNode(
        model_name="openai/clip-vit-base-patch32",
        cache_dir=None,
        default_text_prompt="stones",
        name="perf_test",
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        node.cuda()

    # Warm-up run (triggers model loading)
    logger.info("Warm-up run (loads model)...")
    dummy_img = torch.rand(1, 224, 224, 3).to(device)
    _ = node.forward(image=dummy_img)
    logger.success("✓ Warm-up complete")

    # Performance test
    batch_size = 4
    num_iterations = 10

    logger.info(f"Running {num_iterations} iterations with batch_size={batch_size}...")

    total_time = 0.0
    for i in range(num_iterations):
        rgb_image = torch.rand(batch_size, 224, 224, 3).to(device)

        start_time = time.time()
        _ = node.forward(image=rgb_image)
        elapsed = time.time() - start_time

        total_time += elapsed

        if i == 0:
            logger.info(f"  Iteration 1: {elapsed:.4f}s")

    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    logger.success("✓ Performance test complete")
    logger.info(f"Average time per batch: {avg_time:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} images/sec")
    logger.info(f"Time per image: {avg_time / batch_size * 1000:.2f}ms")

    return avg_time, throughput


def main() -> None:
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("HuggingFace Local Node - Minimal Test Suite")
    logger.info("=" * 60)

    results = {
        "model_loading": False,
        "gradient_passthrough": False,
        "freeze_unfreeze": False,
        "performance": False,
    }

    try:
        # Test 1: Model loading
        node = test_model_loading()
        results["model_loading"] = True

        # Test 2: Gradient passthrough
        results["gradient_passthrough"] = test_gradient_passthrough(node)

        # Test 3: Freeze/unfreeze
        results["freeze_unfreeze"] = test_freeze_unfreeze(node)

        # Test 4: Performance
        avg_time, throughput = test_performance()
        results["performance"] = True

    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:25s}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    logger.info("=" * 60)
    logger.info(f"Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.success("✓✓✓ ALL TESTS PASSED ✓✓✓")
        logger.success("Phase 2 implementation is COMPLETE and VERIFIED!")
    else:
        logger.warning(f"Some tests failed ({total_tests - passed_tests} failures)")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
