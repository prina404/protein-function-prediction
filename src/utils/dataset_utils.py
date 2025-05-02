import os
import pandas as pd

from sklearn.model_selection import train_test_split
from utils.config import CFG


def make_test_train_folders():
    """
    Splits the raw dataset into training and test sets, preprocesses the data,
    and saves them into respective directories as CSV files.
    """

    df = rawdata_to_df()
    df = preprocess_data(df, CFG["data"]["num_classes"])

    df_train, df_test = train_test_split(
        df,
        test_size=CFG["data"]["validation_split"],
        random_state=CFG["project"]["seed"],
        shuffle=True,
        stratify=df["label"],
    )
    print("Saving train and test data in the respective directories...")
    os.makedirs(CFG.train_data.parent, exist_ok=True)
    os.makedirs(CFG.test_data.parent, exist_ok=True)
    df_train.to_csv(CFG.train_data, index=False)
    df_test.to_csv(CFG.test_data, index=False)


def rawdata_to_df() -> pd.DataFrame:
    """
    Reads raw data files, merges them into a single DataFrame with columns
    ["sequence", "label"], and removes duplicates.

    Returns:
        pd.DataFrame: The merged and cleaned dataset.
    """
    assert os.path.exists(CFG.data_dir / "family_classification_metadata.xlsx")
    assert os.path.exists(CFG.data_dir / "family_classification_sequences.csv")
    assert os.path.exists(CFG.data_dir / "protVec_100d_3grams.csv")

    print("Reading raw dataset files...")
    metadata_file = os.path.join(CFG.data_dir, "family_classification_metadata.xlsx")
    sequence_file = os.path.join(CFG.data_dir, "family_classification_sequences.csv")
    df_meta = pd.read_excel(metadata_file)
    df_seq = pd.read_csv(sequence_file)

    # Merge together the sequence column and the family ID column
    df_final = pd.concat([df_seq, df_meta["Family ID"]], axis=1).drop_duplicates()
    df_final.columns = ["sequence", "label"]  # final df columns names

    return df_final


def preprocess_data(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Preprocesses the dataset by keeping the N most frequent labels and grouping
    the rest into an "other" category.

    Args:
        df: The input dataframe with columns ["sequence", "label"].
        N:  The number of most frequent labels to retain.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    print("Preprocessing data...")
    df = multilabel_to_singlelabel(df)  # convert multilabel to single label
    top_labels = set(df["label"].value_counts().index[:N])  # these are the N most common labels

    # set all labels that are not in top_labels to "other"
    df.loc[~df["label"].isin(top_labels), "label"] = "other"
    return df


def multilabel_to_singlelabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a multilabel DataFrame to a single-label DataFrame by retaining
    the least frequent label for each sequence.

    Args:
        df (pd.DataFrame): The input dataset with potential multilabel entries.

    Returns:
        pd.DataFrame: The dataset with single labels per sequence.
    """
    label_frequency = df.groupby("label").count()  # count the number of sequences for each label
    sorted_labels = list(label_frequency.sort_values(by="sequence").index)  # sort by frequency

    label_ranking = {label: idx for idx, label in enumerate(sorted_labels)}  # create a ranking dict
    df["ranking"] = df["label"].map(label_ranking)  # add a column for sorting the entire df

    df = df.sort_values(by=["sequence", "ranking"])  # primary key: seq, secondary key: label
    df = df.drop(columns=["ranking"])
    df = df.drop_duplicates(subset=["sequence"], keep="first")  # keep only the less frequent label

    return df
