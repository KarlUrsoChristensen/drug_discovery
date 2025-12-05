# analyze_sweep.py
import json
from pathlib import Path

def find_best_hyperparameters(
    sweep_results_dir="/dtu/blackhole/0a/224426/runs/sweep_results"
):
    """
    Find the best hyperparameters from the sweep
    """
    
    results_dir = Path(sweep_results_dir)
    
    # Read all_results.json from the trainer
    all_results_file = results_dir / "all_results.json"
    
    if not all_results_file.exists():
        print(f"Error: {all_results_file} not found!")
        print("Make sure the sweep has completed.")
        return None
    
    # Load all results
    with open(all_results_file, "r") as f:
        all_results = json.load(f)
    
    # Find best (lowest MSE) without pandas
    best_result = None
    best_mse = float('inf')
    
    for result in all_results:
        if result['val_MSE'] < best_mse:
            best_mse = result['val_MSE']
            best_result = result
    
    if best_result:
        print("\n" + "="*80)
        print("BEST HYPERPARAMETERS FROM SWEEP")
        print("="*80)
        print(f"\nBest Validation MSE: {best_result['val_MSE']:.6f}")
        print(f"Epoch: {int(best_result['epoch'])}")
        print("="*80)
        
        print("\n" + "="*80)
        print("BEST HYPERPARAMETERS (FROM all_results.json)")
        print("="*80)
        print(f"Validation MSE: {best_result['val_MSE']:.6f}")
        print(f"Learning Rate: {best_result['lr']:.6f}")
        print(f"Weight Decay: {best_result['weight_decay']:.6f}")
        print(f"Scheduler Gamma: {best_result['scheduler_gamma']:.6f}")
        print(f"Hidden Channels: {best_result['hidden_channels']}")
        print(f"Epoch: {int(best_result['epoch'])}")
        print("="*80)
        
        # Save best hyperparameters
        summary_file = Path("/dtu/blackhole/0a/224426/runs") / "best_hyperparameters.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, "w") as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\n✓ Best hyperparameters saved to: {summary_file}")
        
        # Also create a config snippet you can use
        config_snippet = f"""
# Best hyperparameters from sweep
trainer:
  init:
    optimizer:
      lr: {best_result['lr']}
      weight_decay: {best_result['weight_decay']}
    scheduler:
      gamma: {best_result['scheduler_gamma']}

model:
  init:
    hidden_channels: {best_result['hidden_channels']}
"""
        config_file = Path("/dtu/blackhole/0a/224426/runs") / "best_config.yaml"
        with open(config_file, "w") as f:
            f.write(config_snippet)
        
        print(f"✓ Config snippet saved to: {config_file}")
        print("\nYou can use this config with: python run.py --config-path configs --config-name best_config")
        
        return best_result
    else:
        print("No results found in all_results.json!")
        return None


if __name__ == "__main__":
    best = find_best_hyperparameters()