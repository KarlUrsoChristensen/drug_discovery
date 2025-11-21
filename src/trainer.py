from functools import partial
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path  # ADD THIS
import json  # ADD THIS

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.unsupervised_train_dataloader = datamodule.unsupervised_train_dataloader()
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        # Create results directory
        results_dir = Path("outputs/mse_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results list for saving all epochs
        all_results = []
        final_val_mse = None  # Track final validation MSE for Optuna
        
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
                loss = supervised_loss
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                
                # Update final validation MSE
                final_val_mse = val_metrics['val_MSE']
                
                # Create result dict for this epoch
                epoch_result = {
                    "epoch": epoch,
                    "val_MSE": float(val_metrics['val_MSE']),
                    "supervised_loss": float(supervised_losses_logged)
                }
                all_results.append(epoch_result)
                
                # Save to text file (append mode)
                with open(results_dir / "mse_history.txt", "a") as f:
                    f.write(f"Epoch {epoch}: MSE = {val_metrics['val_MSE']:.6f}, Loss = {supervised_losses_logged:.6f}\n")
                
                # Save all results as JSON (overwrites with complete history)
                with open(results_dir / "all_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)
                
                # Also save latest separately for quick access
                with open(results_dir / "latest_mse.json", "w") as f:
                    json.dump(epoch_result, f, indent=2)
                    
            self.logger.log_dict(summary_dict, step=epoch)
        
        # Return final validation MSE for Optuna
        return final_val_mse