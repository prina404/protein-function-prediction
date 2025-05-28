import copy
import torch.nn as nn
import torch
import os
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from collections.abc import Callable
from tqdm.notebook import tqdm
from utils.config import CFG
from pathlib import Path
from datetime import datetime

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
        self.min_delta = min_delta
        self.reset_state()

    def reset_state(self):
        self.counter = 0
        self.best_loss = float("inf")
        self.best_weights = None
        self.was_triggered = False

    def step(self, model: nn.Module, val_loss: float):
        # when I observe an improvement greater than delta, reset counter & backup model parameters
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        self.was_triggered = self.counter >= self.patience  
        return self.was_triggered   # returns true if earlystopping policy is triggered

    def restore_weights(self, model: nn.Module):
        if self.best_weights:
            model.load_state_dict(self.best_weights)


class Trainer:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, loss_fn: Callable, metrics: dict = None, scheduler=None
    ):
        self.model = model.to(CFG.device)
        if CFG.device == 'cuda':
            torch.compiler.reset()
            self.model.compile(fullgraph=True)

        self.optim = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics else {}
        self.scheduler = scheduler
        self.history = {
            "train_loss": [], 
            "validation_loss": [], 
            "metrics": {m: [] for m in metrics} if metrics else {}
            }

        # store the initial weights of the model for resetting the training process
        self.model_init = copy.deepcopy(model.state_dict())
        self.optim_init = copy.deepcopy(optimizer.state_dict())
        self.scheduler_init = copy.deepcopy(scheduler.state_dict()) if scheduler else None

    def train(self, train_data: DataLoader, val_data: DataLoader, epochs=20, early_stopping: EarlyStopping = None):
        self.training_id = self.model.__class__.__name__ + f"_{CFG.num_classes}class"

        if early_stopping: # reset early stopping in case multiple training runs are performed
            early_stopping.reset_state()

        for epoch in range(epochs):
            self.model.train(True)  # set the model to train mode
            total_loss = 0

            progress_bar = tqdm(train_data, desc=f"\nEpoch {epoch+1}/{epochs}", leave=False)
            for X, Y in progress_bar:
                X, Y = X.to(CFG.device), Y.to(CFG.device)           # send batch to device
                self.optim.zero_grad()                              # reset optimizer gradients from previous step
                Y_pred = self.model(X)                              # compute prediction
                loss = self.loss_fn(Y_pred, Y)                      # compute loss
                loss.backward()                                     # compute loss gradient for each parameter
                self.optim.step()                                   # update parameters accordingly
                total_loss += loss.item()

                progress_bar.update()                               # update progress bar
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            train_loss = total_loss / len(train_data)           # avg batch loss
            self.history["train_loss"].append(train_loss)       # record training loss
            print(f"[{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}")

            val_loss = self.evaluate(val_data)
            self.history["validation_loss"].append(val_loss)    # record validation loss 
            print(f"[{epoch + 1}/{epochs}] Val Loss: {val_loss:.4f} ")

            # update the scheduler (e.g. if using ReduceLRonPlateau trigger the desired side effects)
            if self.scheduler:
                self.scheduler.step(val_loss)

            if early_stopping and early_stopping.step(self.model, val_loss):
                print("Stopping training due to EarlyStopping")
                if early_stopping.restore_best_weights:
                    early_stopping.restore_weights(self.model)          # restore weights with lowest observed validation error
                break

    def evaluate(self, val_data: DataLoader) -> float:
        """
        Evaluate the model on the validation data and compute metrics.
        Args:
            val_data (DataLoader): DataLoader for the validation data.
        Returns:
            float: Validation loss.
        """

        all_pred, all_labels, val_loss = self.predict(val_data)  # get predictions and loss

        # compute metrics
        metrics_result = {name: metric_fn(all_pred, all_labels) for name, metric_fn in self.metrics.items()}

        for name, value in metrics_result.items():  # record metrics
            self.history["metrics"][name].append(value)
            print(f"\t{name}: {value:.4f}")

        return val_loss

    def predict(self, data: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Make predictions on the given data and return the predicted labels.
        Args:
            data (DataLoader): DataLoader for the data to be predicted.
        Returns:
            tuple: Tuple containing the predicted labels, true labels, and loss.
        """
        self.model.eval()  # set the model to evaluation mode
        total_loss = 0
        all_pred = []
        all_labels = []
        progress_bar = tqdm(data, desc=f"Evaluation", leave=False)
        with torch.no_grad():
            for X, Y in data:
                X, Y = X.to(CFG.device), Y.to(CFG.device)       # send batches to device
                Y_pred = self.model(X)                          # get predictions
                total_loss += self.loss_fn(Y_pred, Y).item()    # compute loss on the prediction
                all_pred.append(Y_pred.detach().cpu())          # detach tensors and send them back to cpu
                all_labels.append(Y.detach().cpu())

                progress_bar.update()  # update progress bar

        progress_bar.close()
        loss = total_loss / len(data)  # avg batch loss
        all_pred = torch.concatenate(all_pred).argmax(dim=1)  # concat all tensors in a single tensor
        all_labels = torch.concatenate(all_labels).argmax(dim=1)

        return all_pred, all_labels, loss

    def reset_training(self):
        """
        Reset the model weights and training history to the initial state.
        """
        self.model.load_state_dict(self.model_init)     # reset model weights 
        self.optim.load_state_dict(self.optim_init)     # reset optimizer state
        self.scheduler.load_state_dict(self.scheduler_init) if self.scheduler else None
        self.history = {"train_loss": [], "validation_loss": [], "metrics": {m: [] for m in self.metrics}}

    def save_model(self, save_dir: str | Path):
        """
        Save the model state dictionary to a file.
        Args:
            save_dir (str): directory where the model will be saved.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        fName = self.training_id + ".pt"
        torch.save(self.model.state_dict(), os.path.join(save_dir, fName))
        print(f"Model {fName} saved to {save_dir}")

    def save_history(self, save_dir: str | Path):
        """
        Save the training history to a file.
        Args:
            save_dir (str): directory where the history will be saved.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        fName = self.training_id + "_history.pt"
        torch.save(self.history, os.path.join(save_dir, fName))
        print(f"History {fName} saved to {save_dir}")
