import torch
import torch.nn as nn

from utils.config import CFG
from typing import Iterator
from torch.utils.checkpoint import checkpoint  


class ProteinTransformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        num_classes: int = CFG["data"]["num_classes"],
        embed_dim: int = CFG["data"]["embedding_dim"],
    ):
        super().__init__()
        self.max_seq_len = CFG["transformer"]["max_seq_len"]
        self.embed_dim = embed_dim

        self.positional_embedding = nn.Embedding(self.max_seq_len, embed_dim, device=CFG.device)

        self.CLS_token = nn.Embedding(1, embed_dim, device=CFG.device)  # CLS token for classification

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes + 1), # +1 for the negative class
            nn.Softmax()
        )
        print(f"Transformer model with {num_heads} heads, {num_layers} layers and {sum(p.numel() for p in self.parameters())} created.")

    
    def _transformer_encoder(self, sequence, pos_embedding) -> torch.Tensor:
        """Process a single sequence and return the CLS token output."""

        # Add positional embeddings, sequence.shape = (L, E)
        sequence = sequence + pos_embedding

        # Prepend CLS token
        CLS = self.CLS_token.weight  # (1, E)
        sequence = torch.cat((CLS, sequence), dim=0)  # (L+1, E)

        sequence = self.transformer(sequence)

        # Return the CLS token output
        return sequence[0]  # (1, E)

    # Receive a nested tensor of shape (B=batch_size, L=sequence_length, E=embed_dim)
    def forward(self, x: torch.Tensor):
        B, E = x.size(0), x.size(2)  # (batch_size, embed_dim)
        assert E == self.embed_dim


        # Process each chunk separately to save memory
        sequence_embeddings = []
        for i in range(B):          
            seq_len = x[i].size(0)  # = L 
            pos = torch.arange(x[i].size(0), device=CFG.device)
            if len(pos) < self.max_seq_len:
                pos_embedding = self.positional_embedding(pos) # (L, E)
            else:
                pos_embedding = self.positional_embedding(pos[:self.max_seq_len]) # (Max, E)
                overflow_embedding = pos_embedding[-1].repeat(seq_len - self.max_seq_len, 1) # (L - Max, E)
                # pad the remaining potions embedding with the overflow embedding, adding seq_len - max_seq_len embeddings
                pos_embedding = torch.cat((pos_embedding, overflow_embedding), dim=0) # (L, E)
            # Process this seq
            CLS_out = self._transformer_encoder(x[i], pos_embedding)  # (1, E)
            sequence_embeddings.append(CLS_out)  # store the CLS token output

        # Stack the sequence outputs
        x = torch.stack(sequence_embeddings)  # (B, E)

        # Final classification
        logits = self.classifier(x) # (B, num_classes)
        return logits


