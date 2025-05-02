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
        self.label_encoder = LabelBinarizer().fit(self.data["label"])

        # Create the embedding dictionary
        df_3grams = pd.read_csv(CFG.data_dir / "protVec_100d_3grams.csv", sep="\t")

        keys = df_3grams["words"].to_numpy()
        values = df_3grams.iloc[:, 1:].to_numpy(dtype=np.float32)

        embeddings = dict(zip(keys, values))
        default_embedding = embeddings["<unk>"]
        # if I encounter a 3gram not in the dictionary, I will use the embedding of <unk>
        self.embedding_dict = defaultdict(lambda: default_embedding, embeddings)


    def __len__(self) -> int:
        return len(self.data)

    # We apply the same sequence overlapping technique used in the paper:
    # e.g. AGFOYLEK... =
    # [AGF, OYL, EK.] -> offset 0
    # [GFO, YLE, K..] -> offset 1
    # [FOY, LEK, ...] -> offset 2
    def _create_embedding(self, seq: str) -> torch.Tensor:
        splits = []
        for offset in range(3):
            splits.append([seq[offset + i : offset + i + 3] for i in range(0, len(seq) - 2, 3)])
            if len(splits[-1][-1]) < 3:  # if last split is not a 3gram, remove it
                del splits[-1][-1]

        # sum the embeddings of the 3grams
        embedded_splits = [sum(map(lambda x: self.embedding_dict[x], s)) for s in splits]
        embedding = sum(embedded_splits)  # sum the three splits

        return torch.tensor(embedding, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple:
        seq, label = self.data.iloc[idx]
        seq_embedding = self._create_embedding(seq)
        oneHot_label = self.label_encoder.transform([label])[0]

        return seq_embedding, torch.tensor(oneHot_label, dtype=torch.float32)
