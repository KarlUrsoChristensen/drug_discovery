from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf
import json
from pathlib import Path

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device, result_dir=cfg.result_dir)

    # Capture the exact parameters being swept
    trial_hyperparams = {
        'lr': float(cfg.trainer.init.optimizer.lr),
        #'weight_decay': float(cfg.trainer.init.optimizer.weight_decay),
        #'scheduler_gamma': float(cfg.trainer.init.scheduler.gamma),
        #'hidden_channels': int(cfg.model.init.hidden_channels),
        #'alpha': float(cfg.trainer.train.alpha),
        #'unlabeled_per_labeled': int(cfg.trainer.train.unlabeled_per_labeled),
    }
    
    # Train and get final validation MSE (pass hyperparams)
    final_val_mse = trainer.train(
        **cfg.trainer.train,
        trial_hyperparams=trial_hyperparams
    )
    
    # Save sweep parameters and final MSE locally
    sweep_results = {
        'hyperparameters': trial_hyperparams,
        'final_val_mse': float(final_val_mse),
    }
    
    # Create results directory
    results_dir = Path(cfg.result_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(results_dir / "sweep_params.json", "a") as f:
        f.write(json.dumps(sweep_results) + "\n")
    
    # Return MSE for Optuna sweeper
    return final_val_mse


if __name__ == "__main__":
    main()