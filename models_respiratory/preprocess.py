import numpy as np
import os

## Used the functions in this file to run process_and_train.ipynb in colab

FEATURE_INDICES = [8, 12, 13, 14] # indices of raw data features
FEATURE_NAMES = ['SAO2', 'FLOW', 'THORAX', 'ABDOMEN'] #names of raw data features
LABEL_COLUMN = 9 #professinally annotated label of apnea events or normal events
LABEL_THRESHOLD = 0.1 #epoch marked as apnea if more than 10% of time steps are marked apnea events

def preprocess_subject_data(data):
    """
    processing subject data:
        input: 3d .np array of shape (n_epochs, sequence_length, total_channels)
        outputs:
            - 4 selected respiratory features in np array (n_epochs, 3000, 4)
            - labels for each epoch (n_epochs)
    """
    n_epochs = data.shape[0]
    X = np.zeros((n_epochs, 3000, 4))
    y = np.zeros(n_epochs, dtype=np.int32)
    
    for i in range(n_epochs):
        # fills epoch data
        epoch = data[i]
        X[i] = epoch[:, FEATURE_INDICES]
        # finds labels of apnea events for each epoch and computes if >10% of total time to label apnea event
        apnea_labels = epoch[:, LABEL_COLUMN]
        apnea_fraction = np.sum(apnea_labels == 1.0) / len(apnea_labels)
        y[i] = 1 if apnea_fraction >= LABEL_THRESHOLD else 0
    
    return X, y

def normalize_data(X_train, X_val, X_test):
    """
    Normalizing data
    inputs: unnormalized data
    outputs: normalized data removing outliers of more than 10sd
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)
    
    X_train_norm = np.clip(np.nan_to_num((X_train - mean) / std), -10, 10)
    X_val_norm = np.clip(np.nan_to_num((X_val - mean) / std), -10, 10)
    X_test_norm = np.clip(np.nan_to_num((X_test - mean) / std), -10, 10)
    
    return X_train_norm, X_val_norm, X_test_norm

def get_subject_indices(data_shape, epochs_per_subject=350):
    """
    Creates an array of subject IDS for patient
    """
    n_epochs = data_shape[0]
    n_subjects = n_epochs // epochs_per_subject
    subject_ids = np.repeat(np.arange(n_subjects), epochs_per_subject)
    if len(subject_ids) < n_epochs:
        subject_ids = np.concatenate([subject_ids, np.full(n_epochs - len(subject_ids), n_subjects)])
    return subject_ids

def main():
    """
    Main execution. Extracts and normalizes data using above functions
    Saves data to /processed_data directory
    """
    print("Loading data...")
    train_raw = np.load('data/train_data_XL.npy')
    val_raw = np.load('data/val_data_XL.npy')
    test_raw = np.load('data/test_data_XL.npy')
    
    print("Extracting features and labels...")
    X_train, y_train = preprocess_subject_data(train_raw)
    X_val, y_val = preprocess_subject_data(val_raw)
    X_test, y_test = preprocess_subject_data(test_raw)
    
    print("Normalizing...")
    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train, X_val, X_test)
    
    train_subjects = get_subject_indices(train_raw.shape)
    val_subjects = get_subject_indices(val_raw.shape)
    test_subjects = get_subject_indices(test_raw.shape)
    
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_train.npy', X_train_norm)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/train_subjects.npy', train_subjects)
    np.save('processed_data/X_val.npy', X_val_norm)
    np.save('processed_data/y_val.npy', y_val)
    np.save('processed_data/val_subjects.npy', val_subjects)
    np.save('processed_data/X_test.npy', X_test_norm)
    np.save('processed_data/y_test.npy', y_test)
    np.save('processed_data/test_subjects.npy', test_subjects)
    
    print(f"Done. Train: {X_train_norm.shape}, Val: {X_val_norm.shape}, Test: {X_test_norm.shape}")
    print(f"Positive: Train={np.sum(y_train==1)}/{len(y_train)}, Val={np.sum(y_val==1)}/{len(y_val)}, Test={np.sum(y_test==1)}/{len(y_test)}")

if __name__ == '__main__':
    main()