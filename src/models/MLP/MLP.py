from utils.config import CFG
import utils.dataset_utils as du
from sklearn.model_selection import train_test_split
from ProteinDataset import ProteinDataset
from models.MLP.ProteinMLP import ProteinMLP
from Trainer import Trainer, EarlyStopping
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics import Accuracy, Precision, Recall, F1Score



def prepare_data(batch_size):
    '''
    Prepares the training, validation, and test datasets.
    '''
    
    du.make_test_train_folders()
    df_train = du.load_csv(CFG.train_data)
    df_test = du.load_csv(CFG.test_data)

    train_dataset = ProteinDataset(df_train)
    test_dataset = ProteinDataset(df_test)

    df_minority = df_train[df_train['label'] != 'other']
    df_majority = df_train[df_train['label'] == 'other']

    num_minority_samples = len(df_minority)

    df_majority_downsampled = df_majority.sample(n=num_minority_samples, random_state=42)

    # Combine downsampled majority with minority data
    df_train_balanced = pd.concat([df_minority, df_majority_downsampled])

    
    train, val = train_test_split(df_train_balanced, test_size = 0.2, stratify=df_train_balanced["label"])
    train_data = ProteinDataset(train)
    val_data = ProteinDataset(val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, df_train, df_test

def trainMLP(model: ProteinMLP, trainer: Trainer, train_loader: DataLoader, val_loader: DataLoader, es, num_epochs = 10):
    '''
    Trains the given MLP model.
    '''

    trainer.train(train_loader,val_loader, num_epochs, es)
    print(trainer.history)

    torch.save(model.state_dict(), "protein_model.pth")

def save_predictions_to_csv(df_test: pd.DataFrame, y_pred: list):
    '''
    Saves test predictions to CSV file.
    '''
    sequences = df_test["sequence"].tolist()
    df_predictions = pd.DataFrame({"Sequence": sequences[:len(y_pred)], "Prediction": y_pred})

    df_predictions.to_csv("test_predictions.csv", index=False)
    print(f"Predictions saved to test_predictions.csv")

def plot_train_history(history):
    '''
    Plots training and validation loss curves.
    '''
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["validation_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig("training_loss.png")
    plt.close()

if __name__ == '__main__':

    device = CFG.device

    #preprocess the data into ProteinDataset instances
    train_loader, val_loader, test_loader, df_train, df_test = prepare_data(batch_size = 32)

    output_size = len(df_train["label"].unique())

    #initialize MLP with input size, hidden size, output size and dropout
    model = ProteinMLP(input_size=100, hidden_size = 50, output_size = output_size, dropout = 0.3)

    #define optimizer, loss function and metrics
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    metrics = { 
        "accuracy": Accuracy("multiclass", num_classes=output_size, average = "macro"),
        "precision": Precision("multiclass", num_classes=output_size, average = "macro"),
        "recall": Recall("multiclass", num_classes=output_size, average = "macro"),
        "f1": F1Score("multiclass", num_classes=output_size, average = "macro")
        }

    #create a Trainer instance
    trainer = Trainer(model, optimizer, loss_fn, metrics)

    #create an EarlyStopping instance
    es = EarlyStopping(patience = 10, min_delta = 0.001, restore_best_weights=True)

    #train the MLP and plot the training history
    trainMLP(model, trainer, train_loader, val_loader, es, num_epochs=50)

    plot_train_history(trainer.history)

    # store predictions and labels of the validation dataset to plot a confusion matrix
    y_pred_val, y_true_val = [], []
    for X, Y in val_loader:
        preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
        labels = Y.argmax(dim=1).cpu().numpy()
        y_pred_val.extend(preds)
        y_true_val.extend(labels)

    #load the model
    model.load_state_dict(torch.load("protein_model.pth"))
    model.to(device)
    model.eval()


    # Evaluate loss on test set
    test_loss = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Compute metrics manually
    y_pred, y_true = [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1).cpu()
            
            if Y.ndim > 1 and Y.shape[1] > 1:  # one-hot encoded
                labels = Y.argmax(dim=1).cpu()
            else:
                labels = Y.cpu()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    # Compute metrics
    output_size = len(set(y_true))
    metrics = {
        "accuracy": Accuracy(task="multiclass", num_classes=output_size),
        "precision": Precision(task="multiclass", num_classes=output_size, average="macro"),
        "recall": Recall(task="multiclass", num_classes=output_size, average="macro"),
        "f1_score": F1Score(task="multiclass", num_classes=output_size, average="macro"),
    }

    y_pred_tensor = torch.tensor(y_pred)
    y_true_tensor = torch.tensor(y_true)

    print("Test Metrics:")
    for name, metric_fn in metrics.items():
        score = metric_fn(y_pred_tensor, y_true_tensor).item()
        print(f"{name.capitalize()}: {score:.4f}")

    # save predictions to CSV
    save_predictions_to_csv(df_test, y_pred)




