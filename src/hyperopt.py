import torch.nn as nn
import optuna
from optuna import create_study, Trial, TrialPruned
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from utils.config import CFG
from typing import Callable
from Trainer import EarlyStopping
from utils.training_utils import init_trainer, init_loaders
import joblib
from contextlib import redirect_stdout
import os


def objective_MLP(model: nn.Module, model_config: dict) -> Callable[[Trial], float]:
    """
    Objective function builder to optimize the hyperparameters of the MLP model.
    Args:
        model (nn.Module): The model to be optimized.
        model_config (dict): The model configuration dictionary. e.g. CFG['ProtTransMLP']
    Returns:
        objective (Callable): The objective function for Optuna.
    """
    cfg = model_config  # NOTE: the CFG object will be modified during optimization

    ES = None
    if "early_stopping" in cfg:
        es_cfg = cfg["early_stopping"]
        ES = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            restore_best_weights=True,
        )

    # Closure needed for passing different models to the objective function
    def objective(trial: Trial) -> float:
        layer_sizes = [2**i for i in range(6, 12)]  # powers of 2: [64, 128, ..., 4096]
        # NAS
        cfg["n_hidden"] = trial.suggest_int("n_hidden", 1, 8)

        hidden_dims = []
        for i in range(cfg["n_hidden"]):
            hidden_dims.append(trial.suggest_categorical(f"hidden_dim_{i}", layer_sizes))
        cfg["hidden_dims"] = hidden_dims

        cfg["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])
        cfg["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5) if not cfg["batch_norm"] else 0.0

        # Training hyperparameters
        train_cfg = cfg["training"]
        train_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        train_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        # Create a new model instance
        m = model.__class__(cfg)
        trainer = init_trainer(m, cfg)
        train_loader, val_loader = init_loaders(cfg)

        print(f"Starting trial n. {trial.number}")
        for epoch in range(train_cfg["epochs"]):
            with open(os.devnull, "w") as f:  ## Output redirection to avoid spamming progress bars
                with redirect_stdout(f):
                    trainer.train(train_loader, val_loader, epochs=1, early_stopping=ES)

            val_loss = trainer.history["validation_loss"][-1]
            trial.report(val_loss, step=epoch)

            # Pruning / early stopping
            if ES and ES.was_triggered:
                val_loss = trainer.evaluate(val_loader)  # rerun evaluation on best weights
                break

            if trial.should_prune():
                raise TrialPruned()

        return trainer.history["validation_loss"][-1]

    return objective


def optimize_hyperparameters(
    model: nn.Module, model_config: dict, obj_builder: Callable, n_trials: int = CFG["hyperopt"]["num_trials"]
) -> optuna.Study:
    """
    Optimize the hyperparameters of the model then save
    Args:
        model (nn.Module): The model to be trained.
        model_config (dict): The model configuration dictionary. e.g. CFG['ProtTransMLP']
        obj_builder (Callable): The objective function builder for Optuna.
        n_trials (int): The number of trials to run.
    """
    name = f"{model.__class__.__name__}_{CFG.num_classes}classes_study"

    study = create_study(
        study_name=name,
        direction="minimize",
        sampler=TPESampler(),
        pruner=HyperbandPruner(
            min_resource=CFG["hyperopt"]["min_resource"],
            max_resource=model_config["training"]["epochs"],
        ),
    )
    objective = obj_builder(model, model_config)
    study.optimize(objective, n_trials=n_trials)

    path = CFG.root_dir / CFG["paths"]["save_dir"] / "optuna"
    with open(path / f"{name}.pkl", 'wb') as f:
        joblib.dump(study, f)

    return study


def apply_best_config(study: optuna.Study, model_cfg: dict) -> None:
    train_cfg = model_cfg["training"]
    hidden_dims = {}
    for param in study.best_params:
        current_param = study.best_params[param]
        if param in model_cfg:
            model_cfg[param] = current_param
        if param in train_cfg:
            train_cfg[param] = current_param
        if param.startswith('hidden_dim'):
            hidden_dims[param] = current_param
    
    if hidden_dims:
        model_cfg['hidden_dims'] = [hidden_dims[layer] for layer in sorted(list(hidden_dims.keys()))]
