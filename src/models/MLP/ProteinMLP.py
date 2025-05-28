import torch
import torch.nn as nn

class ProteinMLP(nn.Module):
    """Multilayer perceptron class"""
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(ProteinMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

    


 



