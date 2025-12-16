# -*- coding: utf-8 -*-
"""
Sleep apnea balanced dataset generator for EEG to prevent majority bias
The general pipeline is the following: Load -> Filter Sleep -> Epoch -> Threshold (33%) -> Balance (50/50) -> Save

Created on Sun Dec  7 16:44:58 2025

@author: Quint + Gemini 3 Pro
"""
import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
fs = 100
seconds = 30
epoch_size = fs * seconds  # 3000 samples

# Select your subject range
all_people = range(100)  

folder_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/datasets/dreamt-dataset/dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0/data_100Hz/'
output_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data'
output_filename = os.path.join(output_path, 'balanced_training_data.npy')

if not os.path.exists(output_path): #Makes path if its not there
    os.makedirs(output_path)

def filepath(folder_base, i):
    # Adjust filename pattern if necessary
    return folder_base + f"S{i:03d}_PSG_df_updated.csv"

# SUBJECT PROCESSER
def process_subject(subject_id):
    path = filepath(folder_path, subject_id)

    # Define desired columns
    cols = ['Sleep_Stage','TIMESTAMP','C4-M1','F4-M1','O2-M1','Fp1-O2',
            'T3 - CZ','CZ - T4', 'ECG','Central_Apnea'] 
    
    #Read in df
    df = pd.read_csv(path, usecols=cols)
    
    
    #Fill NaNs in Apnea Column with zeros
    df['Central_Apnea'] = df['Central_Apnea'].fillna(0)
    
    #DISCARD DATA THAT IS NOT ASLEEP
    #We keep only rows where Sleep_Stage is NOT 'P', 'W', or 'Missing'
    valid_sleep = ~df['Sleep_Stage'].isin(['P', 'W', 'Missing'])
    df = df[valid_sleep].copy()
    
    #We drop the Sleepstage column its a headache because of th strings
    df.drop('Sleep_Stage', axis=1, inplace=True)
    
    # Create a new list of columns excluding Sleep_Stage
    final_cols = [c for c in cols if c != 'Sleep_Stage']
    
    # Use the new list to order columns
    df = df[final_cols]

    
    #Divy into 30 sec epochs
    n_samples = len(df)
    n_epochs = n_samples // epoch_size
    
    if n_epochs == 0:
        return None, None
        
    cutoff = n_epochs * epoch_size
    # Convert to float32 immediately to save memory
    data_values = df.iloc[:cutoff].values.astype('float32')
    
    # Reshape to (n_epochs, 3000, n_channels)
    epochs = data_values.reshape(n_epochs, epoch_size, df.shape[1])
    
    
    # If any epoch contains a NaN, we discard that specific epoch
    nan_mask = np.isnan(epochs).any(axis=(1, 2))
    epochs = epochs[~nan_mask]
    
    if len(epochs) == 0:
        return None, None

    
    # Find the index of the apnea column
    apnea_idx = df.columns.get_loc("Central_Apnea")
    
    # Calculate fraction of '1's in the apnea channel for each epoch
    apnea_channel = epochs[:, :, apnea_idx]
    apnea_fraction = np.mean(apnea_channel, axis=1)
    
    #Apnea Criterum (> 33%) 
    apnea_indices = np.where(apnea_fraction > 0.33)[0]
    
    
    # Set the entire apnea column to 1.0 for these epochs
    if len(apnea_indices) > 0:
        apnea_indices[:, :, apnea_idx] = 1.0


    # Collect epochs with 0 apnea to a separate list
    normal_indices = np.where(apnea_fraction == 0.0)[0]
    
    
    # Make label homogenous (already 0, but ensures consistency)
    if len(normal_indices) > 0:
        normal_indices[:, :, apnea_idx] = 0.0

    return apnea_indices, normal_indices

# MAIN EXECUTION
def main():
    # SPLIT PATIENTS 
    # We use the first 80/20 split
    all_people = list(range(100))
    train_people = all_people[:80]
    test_people  = all_people[80:]
    
    print(f"Splitting subjects: {len(train_people)} Train / {len(test_people)} Test")

    
    def collect_dataset(people_list, mode_name):
        print(f"\nProcessing {mode_name} set...")
        apnea_epochs = []
        normal_epochs = []
        
        for i in people_list:
            a_chunk, n_chunk = process_subject(i)
            if a_chunk is not None and len(a_chunk) > 0:
                apnea_epochs.append(a_chunk)
            if n_chunk is not None and len(n_chunk) > 0:
                normal_epochs.append(n_chunk)
            print(f"  Processed S{i:03d}")
            
        # Check if empty
        if not apnea_epochs or not normal_epochs:
            print(f"  Warning: Not enough data for {mode_name}.")
            return None

        # Concatenate
        apnea_data = np.concatenate(apnea_epochs, axis=0)
        normal_data = np.concatenate(normal_epochs, axis=0)
        
        # Balance 50/50 (Undersample Normal)
        n_apnea = len(apnea_data)
        n_normal = len(normal_data)
        
        if n_normal > n_apnea:
            indices = np.random.choice(n_normal, n_apnea, replace=False)
            normal_balanced = normal_data[indices]
        else:
            normal_balanced = normal_data
            
        # Combine
        combined = np.concatenate([apnea_data, normal_balanced], axis=0)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(combined))
        final_data = combined[shuffle_idx]
        
        return final_data

    #GENERATE TRAIN DATA
    train_data = collect_dataset(train_people, "TRAIN")
    path = os.path.join(output_path, 'train_data.npy')
    np.save(path, train_data)
    print(f"Saved TRAIN data: {len(train_data)} epochs -> {path}")

    #GENERATE TEST DATA
    test_data = collect_dataset(test_people, "TEST")
    path = os.path.join(output_path, 'test_data.npy')
    np.save(path, test_data)
    print(f"Saved TEST data: {len(test_data)} epochs -> {path}")

if __name__ == "__main__":
    main()