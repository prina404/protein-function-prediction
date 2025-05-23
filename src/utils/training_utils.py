import torch
import torch.nn as nn
import torchmetrics
from ProteinDataset import ProteinDataset, ProtTransDataset
from utils.dataset_utils import load_csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from Trainer import Trainer
from collections.abc import Callable
from utils.config import CFG

## utility functions to create the trainer and load a pre-trained model

optimizer_dict = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


metrics_dict = {
    "accuracy": torchmetrics.Accuracy("multiclass", num_classes=CFG["data"]["num_classes"] + 1, average="macro"),
    "f1_score": torchmetrics.F1Score("multiclass", num_classes=CFG["data"]["num_classes"] + 1, average="macro"),
    "precision": torchmetrics.Precision("multiclass", num_classes=CFG["data"]["num_classes"] + 1, average="macro"),
    "recall": torchmetrics.Recall("multiclass", num_classes=CFG["data"]["num_classes"] + 1, average="macro"),
}


dataset_dict = {
    "ProtVec": ProteinDataset,
    "ProtTrans": ProtTransDataset,
}


def init_trainer(
    model: nn.Module,
    model_conf: dict,  # sub-dictionary e.g. CFG['model_name']
    loss_fn: Callable = nn.CrossEntropyLoss(),
) -> Trainer:
    '''
    Initialize the trainer with the model, optimizer, loss function, and metrics.
    Args:
        model (nn.Module): The model to be trained.
        model_conf (dict): The model configuration dictionary.
        loss_fn (Callable): The loss function to be used.
    Returns:
        Trainer: The trainer instance.
    '''
    train_cfg = model_conf["training"]

    # create optimizer
    opt_name = train_cfg["optimizer"].lower()
    optimizer = optimizer_dict[opt_name](
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    loss_fn = nn.CrossEntropyLoss()
    metrics = {name: metric for name, metric in metrics_dict.items() if name in CFG["evaluation"]["metrics"]}

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
    )


def init_loaders(model_cfg: dict): 
    '''
    Initialize the data loaders for training and validation.
    Args:
        model_cfg (dict): The model configuration dictionary.
    Returns:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
    '''
    df = load_csv(CFG.train_data)
    train_df, val_df = train_test_split(
        df,
        test_size=CFG["data"]["validation_split"],
        stratify=df["label"],
        random_state=CFG["project"]["seed"],
    )

    dataset_class = dataset_dict[model_cfg["dataset_type"]]
    # create datasets
    train_data = dataset_class(train_df, CFG.train_data)
    val_data = dataset_class(val_df, CFG.train_data)

    # upsamplers
    train_sampler = WeightedRandomSampler(train_data.get_sample_weights(), len(train_data), replacement=True)
    val_sampler = WeightedRandomSampler(val_data.get_sample_weights(), len(val_data), replacement=True)

    # create loaders
    train_loader = DataLoader(train_data, batch_size=CFG["data"]["batch_size"], sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=CFG["data"]["batch_size"], sampler=val_sampler, num_workers=8)

    return train_loader, val_loader
