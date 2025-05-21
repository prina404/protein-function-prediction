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
    
    def _create_embedding(self, seq: str) -> torch.Tensor:
        embedding_matrix = self._full_embeddings(seq)
        # sum matrix rows together to get a single embedding
        return torch.sum(embedding_matrix, dim=0)  

    def __getitem__(self, idx: int) -> tuple:
        seq, label = self.data.iloc[idx]
        seq_embedding = self._create_embedding(seq)
        oneHot_label = self.label_encoder.transform([label])[0]

        return seq_embedding, torch.tensor(oneHot_label, dtype=torch.float32)
