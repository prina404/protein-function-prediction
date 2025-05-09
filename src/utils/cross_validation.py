from Trainer import Trainer, EarlyStopping
from ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from utils.config import CFG
import numpy as np
import torch


def kfoldCV(
    model_trainer: Trainer,
    dataset: ProteinDataset,
    num_folds: int,
    epochs: int,
    ES: EarlyStopping = None,
) -> float:
    """
    Perform k-fold cross-validation on the given dataset.

    Args:
        model_trainer (Trainer): The model trainer instance.
        dataset (ProteinDataset): The dataset to be used for training.
        num_folds (int): The number of folds for cross-validation.
        epochs (int): The number of epochs for training.
        ES (EarlyStopping): Early stopping instance to monitor validation loss.
    """
    fold_losses = 0
    fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=CFG["project"]["seed"])

    x, y = np.zeros(len(dataset)), dataset.data["label"]  # use only the labels for stratified kfold
    for i, (train_ids, val_ids) in enumerate(fold.split(x, y)): # Iter over fold indices
        print(f"Fold {i + 1}/{num_folds}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=CFG["data"]["batch_size"], num_workers=8, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=CFG["data"]["batch_size"], num_workers=8, sampler=val_subsampler)

        model_trainer.reset_training()
        model_trainer.train(train_loader, val_loader, epochs, ES)  # train on current fold

        val_loss = (
            model_trainer.evaluate(val_loader)  # If ES restored weights, re-run evaluation
            if ES and ES.was_triggered
            else model_trainer.history["val_loss"][-1]
        )

        fold_losses += val_loss

    print(f"{num_folds}-fold CV loss: {fold_losses / num_folds:.4f}")
    return fold_losses / num_folds
