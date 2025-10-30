import torch
import sys
sys.path.insert(0, 'cuvis_ai')

from anomoly.rx_detector import RXDetectorTorch

# Create test data
torch.manual_seed(42)
B, H, W, C = 2, 10, 10, 5
data = torch.rand(B, H, W, C)

print("Testing RX Detector Score Normalization\n")
print("=" * 60)

# Test each normalization method
methods = ["identity", "minmax", "sigmoid"]

for method in methods:
    print(f"\nTesting method: {method}")
    print("-" * 60)
    
    detector = RXDetectorTorch(score_normalization=method)
    detector.fit(data)
    scores = detector.score(data)
    
    print(f"Shape: {scores.shape}")
    print(f"Min:   {scores.min().item():.6f}")
    print(f"Max:   {scores.max().item():.6f}")
    print(f"Mean:  {scores.mean().item():.6f}")
    print(f"Std:   {scores.std().item():.6f}")
    
    # Check per-batch statistics
    for i in range(B):
        print(f"  Batch {i} - Min: {scores[i].min().item():.6f}, Max: {scores[i].max().item():.6f}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
