import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from pathlib import Path
from config.config import CFG


class ProteinDataset(Dataset):
    def __init__(self, data_file: str | Path):
        if not os.path.exists(CFG.data_dir):
            raise FileNotFoundError(f"Data directory {CFG.data_dir} does not exist.")

        data_split_ok = os.path.exists(CFG.train_data) and os.path.exists(CFG.test_data)
        if not data_split_ok:  # check if train and test folders exist, otherwise create them
            make_test_train_folders()

        # Read data and create the label encoder
        self.data = pd.read_csv(data_file)
        self.label_encoder = LabelBinarizer().fit(self.data["label"])

        # Create the embedding dictionary
        df_3grams = pd.read_csv(CFG.data_dir / "protVec_100d_3grams.csv", sep="\t")

        keys = df_3grams['words'].to_numpy()
        values = df_3grams.iloc[:, 1:].to_numpy(dtype=np.float32)

        self.embedding_dict = dict(zip(keys, values))

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
        # sum the three splits
        embedding = sum(embedded_splits) 
        return torch.tensor(embedding, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple:
        seq, label = self.data.iloc[idx]
        seq_embedding = self._create_embedding(seq)
        oneHot_label = self.label_encoder.transform([label])[0]
        return seq_embedding, oneHot_label


def make_test_train_folders():
    df = rawdata_to_df()
    # df = preprocess_data(df)

    # TODO: should stratify by label, but we need to perform some preprocessing first:
    # (there are labels that appear very few times, and we need to group them together)
    df_train, df_test = train_test_split(
        df,
        test_size=CFG["data"]["validation_split"],
        random_state=CFG["project"]["seed"],
        shuffle=True,
    )
    print("Saving train and test data in the appropriate folders...")
    os.makedirs(CFG.train_data.parent, exist_ok=True)
    os.makedirs(CFG.test_data.parent, exist_ok=True)
    df_train.to_csv(CFG.train_data, index=False)
    df_test.to_csv(CFG.test_data, index=False)


# Reads the raw data files, and merges them into a single DataFrame with columns: ["sequence", "label"]
def rawdata_to_df() -> pd.DataFrame:
    # Check if the dataset is already unzipped
    assert os.path.exists(CFG.data_dir / "family_classification_metadata.xlsx")
    assert os.path.exists(CFG.data_dir / "family_classification_sequences.csv")
    assert os.path.exists(CFG.data_dir / "protVec_100d_3grams.csv")

    print("Reading raw dataset files...")
    metadata_file = os.path.join(CFG.data_dir, "family_classification_metadata.xlsx")
    sequence_file = os.path.join(CFG.data_dir, "family_classification_sequences.csv")
    df_meta = pd.read_excel(metadata_file)
    df_seq = pd.read_csv(sequence_file)

    print("Merging data files...")
    df_final = pd.concat([df_seq, df_meta["Family ID"]], axis=1).drop_duplicates()
    df_final.columns = ["sequence", "label"]  # final df columns

    return df_final


# Keep the N most frequent labels, and group the rest into "other"
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    pass
