import numpy as np
import pandas as pd
from src.utils.config import CFG
from src.utils.dataset_utils import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from focal_loss import SparseCategoricalFocalLoss
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(CFG['project']['seed'])
tf.random.set_seed(CFG['project']['seed'])

# Convert protein sequences to embedding vectors using 3-gram dictionary
def create_embeddings_sequences(df, df_3grams):
    """
    Creates embeddings sequences for each row of the dataset

    Parameters:
    df: dataframe
    df_3grams: dataframe with embeddings of 3grams

    Returns:
    df: dataframe with sequence of embeddings per row
    """

    # dict with the embedding of each 3gram
    embedding_dict = {row['words']: row[1:].tolist() for _, row in df_3grams.iterrows()}
    # add the column with the list of the 3grams embeddings
    df['Sequence_embeddings'] = None
    for i, row in df.iterrows():
        sequence_embedding = []
        protein_seq = row['sequence']

        # Create the list of 3grams embeddings of the protein sequence
        for j in range(len(protein_seq) - 2):
            trigram = protein_seq[j:j + 3]

            if trigram in embedding_dict:
                embedding = embedding_dict[trigram]
            else:
                embedding = embedding_dict['<unk>']

            sequence_embedding.append(embedding)

        # Add the sequence to the dataframe
        df.at[i, 'Sequence_embeddings'] = np.array(sequence_embedding)

    return df

# Balance dataset
def balance_df(df):
    """
    Balance dataframe with the same number of 'other' classes as the number of classes to classify

    Parameters:
    df: dataframe to be balanced

    Returns:
    df: balanced dataframe
    """
    n = CFG['data']['num_classes']
    # df_final = preprocess_data(df_final, n)
    top_families = list(df['label'].value_counts()[:n + 1].index)
    top_families.remove('other')

    mask = df['label'].isin(top_families)
    df_topfamilies = df[mask]
    df_others = df[~mask]

    df_others = df_others.sample(n=len(df_topfamilies), random_state=CFG['project']['seed'])

    df = pd.concat([df_topfamilies, df_others], axis=0)
    df.reset_index(drop=True, inplace=True)

    return df

# Load protein data and prepare embedding sequences for LSTM training
def load_data(file_3grams):
    """
    Load protein sequences and metadata, convert to embeddings and encode labels.

    Parameters:
    file_path_seq (str): Path to CSV file with protein sequences
    file_path_meta (str): Path to Excel file with protein metadata
    file_path_3grams (str): Path to CSV file with 3-gram embeddings

    Returns:
    X_train (list): Training embedding sequences
    y_train_encoded (array): Encoded training labels
    X_test (list): Test embedding sequences
    y_test_encoded (array): Encoded test labels
    label_encoder: LabelEncoder for converting family names to integers
    """
    make_test_train_folders()
    df_train = pd.read_csv(CFG['paths']['train_data'])
    df_test = pd.read_csv(CFG['paths']['test_data'])
    file_3grams = os.path.join(CFG.data_dir, file_3grams)
    df_3grams = pd.read_csv(file_3grams, sep='\t')

    df_train = balance_df(df_train)
    df_test = balance_df(df_test)

    df_train = create_embeddings_sequences(df_train, df_3grams)
    df_test = create_embeddings_sequences(df_test, df_3grams)

    # Process sequences and embeddings
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for _, row in df_train.iterrows():
        embedding_matrix = row['Sequence_embeddings']
        X_train.append(embedding_matrix)
        y_train.append(row['label'])

    for _, row in df_test.iterrows():
        embedding_matrix = row['Sequence_embeddings']
        X_test.append(embedding_matrix)
        y_test.append(row['label'])

    # Encode the family labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    print(f"Loaded {len(X_train)+len(X_test)} protein sequences")
    print(f"Number of unique families: {len(label_encoder.classes_)}")
    print(f"Family train distribution: {np.bincount(y_train_encoded)}")
    print(f"Family test distribution: {np.bincount(y_test_encoded)}")

    return X_train, y_train_encoded, X_test, y_test_encoded, label_encoder

# Pad embedding sequences to uniform length for batch processing
def pad_embedding_sequences(X, max_length=None):
    """
    Pad or truncate sequences to consistent length using 95th percentile.

    Parameters:
    X (list): List of embedding matrices for each protein
    max_length (int, optional): Maximum sequence length. If None, calculated from data

    Returns:
    padded_X (array): Padded sequences with uniform length
    """
    print(f"Padding {len(X)} sequences")
    if max_length is None:
        # Calculate the sequence length at 95th percentile to avoid excessive padding
        lengths = [len(seq) for seq in X]
        max_length = int(np.percentile(lengths, 95))
        print(f"Using max_length of {max_length} (95th percentile of sequence lengths)")

    # Create a padding mask - sequences shorter than max_length will be padded with zeros
    # Sequences longer than max_length will be truncated
    padded_X = []
    for seq in X:
        if len(seq) > max_length:
            # Trim
            padded_X.append(seq[:max_length])
        else:
            # Pad with zeros
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            padded_X.append(np.vstack([seq, padding]))

    return np.array(padded_X)

# Build bidirectional LSTM model for protein family classification
def build_lstm_model(input_shape, num_classes):
    """
    Create sequential model with bidirectional LSTM layers and regularization.

    Parameters:
    input_shape (tuple): Shape of input data (sequence_length, embedding_dim)
    num_classes (int): Number of protein families to classify

    Returns:
    model: Compiled Keras model with focal loss and Adam optimizer
    """
    model = Sequential([
        # Bidirectional LSTM layer to capture context from both directions
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=SparseCategoricalFocalLoss(gamma=2.0),
        metrics=['accuracy']
    )

    return model

# Train LSTM model with callbacks for early stopping and learning rate reduction
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train model with validation monitoring and automatic optimization callbacks.

    Parameters:
    model: Compiled Keras model to train
    X_train, y_train: Training data and labels
    X_val, y_val: Validation data and labels
    batch_size (int): Batch size for training
    epochs (int): Maximum number of epochs

    Returns:
    history: Training history with loss and accuracy metrics
    """
    # Stop training when val_loss does not improve after 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # Model checkpoint to save best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return history

# Evaluate model performance and generate classification report with plots
def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Test model accuracy and create visualization plots for performance analysis.

    Parameters:
    model: Trained Keras model to evaluate
    X_test, y_test: Test data and labels
    label_encoder: LabelEncoder used for converting labels to class names

    Returns:
    None (prints classification report and saves confusion matrix and training plots)
    """
    # Predict on test data
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('LSTM_confusion_matrix.png')
    plt.close()

    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('LSTM_training_history.png')
    plt.close()


if __name__ == '__main__':
    file_3grams = "protVec_100d_3grams.csv"

    # Load and preprocess data
    X_train_val, y_train_val, X_test, y_test, label_encoder = load_data(file_3grams)

    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=CFG["data"]["validation_split"],
        random_state=CFG["project"]["seed"],
        stratify=y_train_val,
        shuffle=CFG["data"]["shuffle"]
    )

    # Determine suitable max_length for padding
    sequence_lengths = [len(seq) for seq in X_train]
    max_length = int(np.percentile(sequence_lengths, 95))  # 95th percentile
    print(f"Max sequence length (95th percentile): {max_length}")

    # Pad sequences
    X_train_padded = pad_embedding_sequences(X_train, max_length)
    X_val_padded = pad_embedding_sequences(X_val, max_length)
    X_test_padded = pad_embedding_sequences(X_test, max_length)
    print(f"Padded training data shape: {X_train_padded.shape}")

    # Build the model
    embedding_dim = X_train[0].shape[1]  # Get embedding dimension from the data
    num_classes = len(np.unique(y_train))

    model = build_lstm_model(
        input_shape=(max_length, embedding_dim),
        num_classes=num_classes
    )

    model.summary()

    # Train the model
    history = train_model(
        model, X_train_padded, y_train, X_val_padded, y_val,
        batch_size=32,
        epochs=50
    )

    # Evaluate the model
    model.load_weights("best_model.keras")
    evaluate_model(model, X_test_padded, y_test, label_encoder)
