import copy
import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from collections.abc import Callable
from tqdm import tqdm
from utils.config import CFG


class EarlyStopping:
    """
    Early stopping to stop training when the validation loss does not improve for a given number of epochs.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change to qualify as an improvement.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
    """

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.min_delta = min_delta

    def step(self, model: nn.Module, val_loss: float):
        # when I observe an improvement greater than delta, reset counter & backup model parameters
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience  # returns false if earlystopping policy is triggered

    def restore(self, model: nn.Module):
        if self.best_weights:
            model.load_state_dict(self.best_weights)


class Trainer:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, loss_fn: Callable, metrics: dict = None, scheduler=None
    ):
        self.model = model.to(CFG.device)
        if CFG.device == 'cuda':
            self.model.compile()
            
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics else {}
        self.scheduler = scheduler

    def train(self, train_data: DataLoader, val_data: DataLoader, epochs=20, early_stopping: EarlyStopping = None):
        history = {"train_loss": [], "validation_loss": [], "metrics": {m: [] for m in self.metrics}}

        for epoch in range(epochs):
            self.model.train(True)  # set the model to train mode
            total_loss = 0
            progress_bar = tqdm(train_data, desc=f"\nEpoch {epoch+1}/{epochs}", leave=True)

            for X, Y in train_data:
                X, Y = X.to(CFG.device), Y.to(CFG.device)       # send batch to device
                self.optim.zero_grad()                          # reset optimizer gradients from previous step
                Y_pred = self.model(X)                          # compute prediction
                loss = self.loss_fn(Y_pred, Y)                  # compute loss
                loss.backward()                                 # compute loss gradient for each parameter
                self.optim.step()                               # update parameters accordingly
                total_loss += loss.item()

                progress_bar.update(1)                          # update progress bar
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()
            train_loss = total_loss / len(train_data)   # avg batch loss
            history["train_loss"] = train_loss          # record training loss
            print(f"[{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}")

            val_loss, metrics = self.evaluate(val_data)
            history["validation_loss"] = val_loss       # record validation loss & metrics
            print(f"[{epoch + 1}/{epochs}] Val Loss: {val_loss:.4f} ")

            for name, val in metrics.items():
                history["metrics"][name].append(val)
                print(f"\t{name}: {val:.4f}")

            # update the scheduler (e.g. if using ReduceLRonPlateau trigger the desired side effects)
            if self.scheduler:
                self.scheduler.step(val_loss)

            if early_stopping and early_stopping.step(self.model, val_loss):
                print("Stopping training due to EarlyStopping")
                if early_stopping.restore_best_weights:
                    early_stopping.restore(self.model)      # restore weights with lowest observed validation error
                break

    def evaluate(self, val_data: DataLoader):
        self.model.eval()  # set the model to evaluation mode
        total_loss = 0
        all_pred = []
        all_labels = []
        progress_bar = tqdm(val_data, desc=f"Evaluation", leave=True)
        with torch.no_grad():
            for X, Y in val_data:
                X, Y = X.to(CFG.device), Y.to(CFG.device)  # send batches to device
                Y_pred = self.model(X)  # get predictions
                total_loss += self.loss_fn(Y_pred, Y).item()  # compute loss on the prediction
                all_pred.append(Y_pred.detach().cpu())  # detach tensors and send them back to cpu
                all_labels.append(Y.detach().cpu())
                
                progress_bar.update(1)                          # update progress bar

        progress_bar.close()
        val_loss = total_loss / len(val_data)  # avg batch loss
        all_pred = torch.concatenate(all_pred)  # concat all tensors in a single tensor
        all_labels = torch.concatenate(all_labels)

        metrics_result = {name: metric_fn(all_pred, all_labels) for name, metric_fn in self.metrics.items()}
        return val_loss, metrics_result
