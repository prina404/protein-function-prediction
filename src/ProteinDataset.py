import os
import torch
import utils.dataset_utils as util
import pandas as pd
import numpy as np

from utils.config import CFG
from utils.dataset_utils import load_csv
from torch.utils.data import Dataset, TensorDataset
from models.ProtTransClassifier import ProtTransEncoder
from tqdm import tqdm
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



class ProtTransDataset(ProteinDataset):
    def __init__(self, data_df: pd.DataFrame, source_dataset: str | Path):
        '''
        Args:
            data_df (pd.DataFrame): subset of the entire dataframe containing the (seq, label) tuples.
            source_dataset (str | Path): Path to the CSV file containing the protein sequences and labels.
        '''
        super().__init__(data_df)

        embedding_file = Path(source_dataset).with_suffix('')       # remove .csv
        embedding_file = str(embedding_file) + "_ProtTrans.pt"    # add _ProtTrans.pt suffix

        if not os.path.exists(embedding_file):         # if the embedding file does not exist, create it
            self._create_embeddings(embedding_file)
    
        # filter the dataset to only include the sequences in data_df
        full_df = load_csv(source_dataset)
        indices = full_df[full_df["sequence"].isin(data_df["sequence"])].index

        # dataset is a tuple, where the first element is the data tensor of shape (num_seq, 1024), and the 
        # second element is the label tensor of shape (num_seq, num_classes)
        full_dataset = torch.load(embedding_file, weights_only=False)
        self.dataset = TensorDataset(full_dataset.tensors[0][indices], full_dataset.tensors[1][indices])


    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.dataset.__getitem__(idx)
    
    def to_pandas(self) -> pd.DataFrame:
        '''
        Returns a dataframe with shape (len(dataset), 1025), where the first column ('sequence') contains the
        full sequence, and the remaining 1024 columns contain the corresponding protein embedding 
        '''
        df = pd.DataFrame(self.dataset.tensors[0])
        df = pd.concat((self.data['sequence'].reset_index(drop=True), df), axis=1, ignore_index=True)
        old_col = df.columns.copy()
        df.columns = ['sequence'] + list(map(str, old_col[:-1]))
        return df

    def _create_embeddings(self, embedding_file: str) -> None:
        encoder = ProtTransEncoder()
        embeddings = torch.zeros((len(self.data), 1024), dtype=torch.float32)
        labels = torch.zeros((len(self.data), CFG['data']['num_classes'] + 1), dtype=torch.float32)
        
        offset = 0
        data_to_encode = self.data.values
        if self._has_checkpoint():
            checkpoint = torch.load("emb_checkpoint.pt", weights_only=False)
            emb, lbl = checkpoint.tensors
            offset = len(emb)
            embeddings[:offset] = emb
            labels[:offset] = lbl
            data_to_encode = self.data.values[offset:]

        # I create the protein embeddings one by one to avoid dealing with variable length sequences and
        # the padding addend by the tokenizer. Also when using an 8GB GPU the maximum batch size is ~2
        print(f"Creating tensor file {embedding_file}, it may take some time ...")
        progress_bar = tqdm(enumerate(data_to_encode), total=len(data_to_encode), desc="Encoding sequences")
        for i, (seq, label) in progress_bar:
            oneHot_label = self.label_encoder.transform([label])[0]
            label = torch.tensor(oneHot_label, dtype=torch.float32)
            labels[offset + i] = label

            try:
                seq_embedding = encoder.encode([seq])[0] # (1, 1024) -> (1024,)
            except torch.OutOfMemoryError:
                old_dev = CFG.device
                encoder.to("cpu")
                CFG.device = "cpu"
                seq_embedding = encoder.encode([seq])[0]
                CFG.device = old_dev
                encoder.to(old_dev)

            embeddings[offset + i] = seq_embedding
        
            if i % 2000 == 0 and i > 0:
                self._save_checkpoint(TensorDataset(embeddings[:offset + i], labels[:offset + i]))
                torch.cuda.empty_cache()
                print(f"Saved checkpoint at {i} sequences")
                
        
        dataset = TensorDataset(embeddings, labels)
        torch.save(dataset, embedding_file)

        if self._has_checkpoint():
            os.remove("emb_checkpoint.pt")

    def _save_checkpoint(self, tensor: TensorDataset) -> None:
        if self._has_checkpoint():
            os.remove("emb_checkpoint.pt")
        torch.save(tensor, "emb_checkpoint.pt")
    
    def _has_checkpoint(self) -> bool:
        return 'emb_checkpoint.pt' in os.listdir()
