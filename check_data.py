#!/usr/bin/env python3
"""
Script to check the exact size of your QM9 dataset
"""

import os
import torch
from torch_geometric.datasets import QM9

# Update this path to match your data directory
data_dir = "./data"  # or wherever your QM9 data is stored

# Load the dataset
dataset = QM9(root=data_dir)

# Print exact size
print(f"QM9 Dataset exact size: {len(dataset):,} molecules")

# Calculate splits with your configuration
splits = [0.72, 0.08, 0.1, 0.1]
total = len(dataset)

unlabeled_train = int(total * splits[0])
labeled_train = int(total * splits[1])
val = int(total * splits[2])
test = int(total * splits[3])

print(f"\nWith splits {splits}:")
print(f"  Unlabeled train (72%): {unlabeled_train:,} samples")
print(f"  Labeled train (8%):    {labeled_train:,} samples")
print(f"  Validation (10%):      {val:,} samples")
print(f"  Test (10%):            {test:,} samples")
print(f"\nTotal: {unlabeled_train + labeled_train + val + test:,}")