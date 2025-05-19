import os
import torch
import utils.dataset_utils as util
import pandas as pd
import numpy as np

from typing import Iterator
from utils.config import CFG
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
from collections import defaultdict


class ProteinDataset(Dataset):
    def __init__(self, data_file: str | Path):
        if not os.path.exists(CFG.data_dir):
            raise FileNotFoundError(f"Data directory {CFG.data_dir} does not exist.")

        # check if train and test folders exist, otherwise create them
        data_split_ok = os.path.exists(CFG.train_data) and os.path.exists(CFG.test_data)
        if not data_split_ok:
            util.make_test_train_folders()

        # Read data from disk and initialize the label encoder
        self.data = pd.read_csv(data_file)
        self.label_encoder = LabelBinarizer().fit(self.data["label"].sort_values())

        # Create the embedding dictionary
        df_3grams = pd.read_csv(CFG.data_dir / "protVec_100d_3grams.csv", sep="\t")

        keys = df_3grams["words"].to_numpy()
        values = df_3grams.iloc[:, 1:].to_numpy(dtype=np.float32)

        embeddings = dict(zip(keys, values))
        default_embedding = embeddings["<unk>"]
        # if I encounter a 3gram not in the dictionary, I will use the embedding of <unk>
        self.embedding_dict = defaultdict(lambda: default_embedding, embeddings)
        self.embedding_size = len(default_embedding)


    def __len__(self) -> int:
        return len(self.data)

    # Use a sliding window to create the embedding matrix
    def _full_embeddings(self, seq: str) -> torch.Tensor:
        matrix = np.zeros((len(seq)-2, self.embedding_size), dtype=np.float32)
        for i in range(len(seq)-2):
            matrix[i] = self.embedding_dict[seq[i:i+3]]

        return torch.tensor(matrix, dtype=torch.float32)
    
    def _create_embedding(self, seq: str) -> torch.Tensor:
        embedding_matrix = self._full_embeddings(seq)
        # sum matrix rows together to get a single embedding
        return torch.sum(embedding_matrix, dim=0)  

    def __getitem__(self, idx: int) -> tuple:
        seq, label = self.data.iloc[idx]
        seq_embedding = self._create_embedding(seq)
        oneHot_label = self.label_encoder.transform([label])[0]

        return seq_embedding, torch.tensor(oneHot_label, dtype=torch.float32)
    
    def get_label_weights(self) -> torch.Tensor:
        """
        Returns the weights of the labels in the dataset.
        """
        label_weights = torch.zeros(len(self.label_encoder.classes_), device=CFG.device)
        for label in self.data["label"]:
            index = np.where(self.label_encoder.classes_ == label)[0][0]
            label_weights[index] += 1

        return label_weights / len(self.data)


class TransformerDataset(ProteinDataset):
    '''
    Instead of outputting a single embedding for an entire protein, returns a list of embeddings for each amino acid in the protein.
    '''
    def __init__(self, data_file: str | Path):
        super().__init__(data_file)


    def _create_embedding(self, seq) -> torch.Tensor:
        return self._full_embeddings(seq)


def dynamic_batch_wrapper(loader: torch.utils.data.DataLoader) -> Iterator[torch.Tensor]:
    """
    Wrapper unction to create batches of variable size, so that no batch exceeds the maximum number of embeddings
    specified in the config file.
    """
    max_embeddings = CFG["transformer"]["max_embeddings_per_batch"]
    embedding_size = CFG["data"]["embedding_dim"]
    current_batch = []
    batch_labels = []
    current_batch_size = 0
    for X, Y in loader:
        X = torch.squeeze(X, 0)  # remove the first dimension (batch size)
        Y = torch.squeeze(Y, 0)  
        # X has shape (C=num_chunks, L=chunk_len, E=embedding_dim)
        # infer the number of embeddings by the total number of elements in the tensor
        num_embeddings = X.numel() // embedding_size

        # if batch size does not exceed the max size, add the tensor
        if current_batch_size + num_embeddings <= max_embeddings or len(current_batch) == 0:
            current_batch.append(X)
            batch_labels.append(Y)
            current_batch_size += num_embeddings
        else:  # max size exceeded -> yield the current batch and start a new one
            yield torch.nested.nested_tensor(current_batch, layout=torch.jagged, requires_grad=False), torch.stack(batch_labels)
            current_batch = [X]
            batch_labels = [Y]
            current_batch_size = num_embeddings

    if len(current_batch) > 0:  # yield leftover tensors, if any
        yield torch.nested.nested_tensor(current_batch, layout=torch.jagged, requires_grad=False), torch.stack(batch_labels)
