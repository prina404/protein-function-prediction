from models.transformer import ProteinTransformer
from ProteinDataset import TransformerDataset, dynamic_batch_wrapper
from utils.config import CFG
from torch.utils.data import DataLoader
from Trainer import Trainer
import torchmetrics
import torch



if __name__ == '__main__':
    ds = TransformerDataset(CFG.train_data)
    train_data, val_data = torch.utils.data.random_split(TransformerDataset(CFG.train_data), [0.8, 0.2])
    
    train_loader = dynamic_batch_wrapper(DataLoader(train_data, batch_size=1,  shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=10))
    val_loader = dynamic_batch_wrapper(DataLoader(val_data, batch_size=1,  shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=10))

    t = ProteinTransformer(num_heads=10, num_layers=5)
    metrics = {'accuracy':torchmetrics.Accuracy(task="multiclass", num_classes=CFG["data"]["num_classes"]+1)}
    trainer = Trainer(t, torch.optim.AdamW(t.parameters(), lr=0.0001), torch.nn.CrossEntropyLoss(),metrics=metrics)
    trainer.train(train_loader, val_loader)


