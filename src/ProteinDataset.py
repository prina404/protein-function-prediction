import os
import torch
import utils.dataset_utils as util
import pandas as pd
import numpy as np

from utils.config import CFG
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
from collections import defaultdict
from functools import cache


class ProteinDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, *args, **kwargs):
        self.data = data_df.copy().sort_index()

        max_len = CFG['data']['max_seq_len']
        # Remove sequences longer than max_seq_len
        self.data = self.data[self.data["sequence"].apply(lambda x: len(x) < max_len)]

        self.label_encoder = LabelBinarizer().fit(self.data["label"])

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
    
    @cache
    def _create_embedding(self, seq: str) -> torch.Tensor:
        embedding_matrix = self._full_embeddings(seq)
        # sum matrix rows together to get a single embedding
        return torch.sum(embedding_matrix, dim=0)  

    def __getitem__(self, idx: int) -> tuple:
        seq, label = self.data.iloc[idx]
        seq_embedding = self._create_embedding(seq)
        oneHot_label = self.label_encoder.transform([label])[0]

        return seq_embedding, torch.tensor(oneHot_label, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        '''
        Returns a tensor of sample weights for the dataset. The sample weights are computed as the inverse of the
        class frequencies in the dataset. The sample weights are used to balance the dataset during training.
        '''
        class_counts = self.data["label"].value_counts()
        class_weights = (1.0 / class_counts) 
        
        sample_weights = class_weights[self.data["label"]].values
        return torch.tensor(sample_weights, dtype=torch.float32)


