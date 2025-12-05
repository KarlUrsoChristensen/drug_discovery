from functools import partial
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path  # ADD THIS
import json  # ADD THIS
import copy
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader

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
        result_dir="outputs/mse_results",
    ):
        self.device = device
        self.models = models
        self.result_dir = result_dir 

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

    def train(self, total_epochs, validation_interval, trial_hyperparams=None):
        """
        Train the ensemble model
        
        Args:
            total_epochs: Number of epochs to train
            validation_interval: How often to validate
            trial_hyperparams: Dict with hyperparameters for this trial
                {
                    'lr': float,
                    'weight_decay': float,
                    'scheduler_gamma': float,
                    'hidden_channels': int
                }
        """
        # Create results directory
        results_dir = Path(self.result_dir) 
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
                
                # Add trial hyperparameters if provided
                if trial_hyperparams is not None:
                    epoch_result.update(trial_hyperparams)
                
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


class MeanTeacherEnsemble:
    def __init__(
        self,
        supervised_criterion,
        unsupervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        result_dir="outputs/mse_results",
    ):
        self.device = device
        self.student_models = models
        self.teacher_models = copy.deepcopy(self.student_models)
        self.result_dir = result_dir

        # Optim related things
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        student_params = [p for m in self.student_models for p in m.parameters()]
        self.optimizer = optimizer(params=student_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        old_loader = datamodule.unsupervised_train_dataloader() # length 942 batches
        self.unlabeled_dataloader = DataLoader(
            old_loader.dataset,
            batch_size=old_loader.batch_size,
            sampler=RandomSampler(old_loader.dataset, replacement = True)
        )

        self.train_dataloader = datamodule.train_dataloader() # length 105 batches
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def ema_update(self, alpha):
        for student_model, teacher_model in zip(self.student_models, self.teacher_models):
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

    def validate(self):
        for teacher_model in self.teacher_models:
            teacher_model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [teacher_model(x) for teacher_model in self.teacher_models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval, alpha, unlabeled_per_labeled, trial_hyperparams=None):
        # Create results directory
        results_dir = Path(self.result_dir) 
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results list for saving all epochs
        all_results = []
        final_val_mse = None  # Track final validation MSE for Optuna

        # Add trial hyperparameters if provided
        
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for student_model, teacher_model in zip(self.student_models, self.teacher_models):
                student_model.train()
                teacher_model.train()

            unlabeled_dataloader_iter = iter(self.unlabeled_dataloader)

            supervised_losses_logged = []
            unsupervised_losses_logged = []
            losses_logged = []
            
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(student_model(x), targets) for student_model in self.student_models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.student_models))

                unsupervised_loss = torch.zeros(1, device=self.device)
                for _ in range(unlabeled_per_labeled):
                    xu = next(unlabeled_dataloader_iter)
                    xu = xu.to(self.device)
                    with torch.no_grad():
                        preds_teachers = [teacher_model(xu) for teacher_model in self.teacher_models]
                    preds_students = [student_model(xu) for student_model in self.student_models]
                    unsupervised_losses = [self.unsupervised_criterion(ps, pt) for ps, pt in zip(preds_students, preds_teachers)]
                    unsupervised_loss += sum(unsupervised_losses)
                if unlabeled_per_labeled != 0:
                    unsupervised_loss /= unlabeled_per_labeled
                unsupervised_losses_logged.append(unsupervised_loss.detach().item() / len(self.teacher_models))

                loss = supervised_loss + unsupervised_loss
                losses_logged.append(loss.detach().item())
                loss.backward()
                self.optimizer.step()
                # update teacher weights
                self.ema_update(alpha)

            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            unsupervised_losses_logged = np.mean(unsupervised_losses_logged)
            losses_logged = np.mean(losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "unsupervised_loss": unsupervised_losses_logged,
                "total_loss": losses_logged,
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
                    "supervised_loss": float(supervised_losses_logged),
                    "alpha": alpha,
                    "unlabeled_per_labeled": unlabeled_per_labeled
                }
                if trial_hyperparams is not None:   
                    epoch_result.update(trial_hyperparams)
                
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