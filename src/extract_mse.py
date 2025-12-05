import os
import re
from pathlib import Path

# Define the base path
base_path = "/dtu/blackhole/0a/224426/runs"

# Define the subdirectories to search
subdirs = [
    "original",
    "original_norm",
    "original_norm_skip",
    "original_norm_skip_7layer",
]

# Dictionary to store results
mse_results = {}

# Search for mse_history.txt in each directory
for subdir in subdirs:
    search_path = os.path.join(base_path, subdir)
    
    if not os.path.exists(search_path):
        print(f"Warning: {search_path} not found")
        continue
    
    print(f"\nSearching in {search_path}...")
    
    # Look for results folders recursively
    for root, dirs, files in os.walk(search_path):
        if "mse_history.txt" in files:
            mse_file = os.path.join(root, "mse_history.txt")
            
            try:
                with open(mse_file, 'r') as f:
                    lines = f.readlines()
                
                mse_values = []
                for line in lines:
                    # Parse lines like "Epoch 10: MSE = 0.273625, Loss = 0.280315"
                    match = re.search(r'MSE = ([\d.]+)', line)
                    if match:
                        mse_values.append(float(match.group(1)))
                
                if mse_values:
                    best_mse = min(mse_values)
                    # Store with relative path for clarity
                    rel_path = os.path.relpath(mse_file, search_path)
                    key = f"{subdir}/{rel_path}"
                    mse_results[key] = best_mse
                    print(f"  ✓ {rel_path}: {best_mse:.6f} (from {len(mse_values)} epochs)")
                else:
                    print(f"  ✗ No MSE values found in {rel_path}")
            except Exception as e:
                print(f"  ✗ Error reading {mse_file}: {e}")

# Print summary
print("\n" + "="*70)
print("SUMMARY - Best MSE for each experiment:")
print("="*70)

for path, mse in sorted(mse_results.items()):
    print(f"{path:<60} {mse:>12.6f}")

# Find overall best
if mse_results:
    best_overall = min(mse_results.values())
    best_path = [k for k, v in mse_results.items() if v == best_overall][0]
    print("\n" + "="*70)
    print(f"Overall best MSE: {best_overall:.6f}")
    print(f"Location: {best_path}")
    print("="*70)
else:
    print("\nNo results found!")