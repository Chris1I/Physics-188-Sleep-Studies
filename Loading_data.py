

# -*- coding: utf-8 -*-
"""
Makes the datasets for all the other further CNN MLP sytems with some adjustments between group size and stuff like that.
It is very unbalanced due to the nature of sleep apnea though
Created on Sun Dec  7 16:44:58 2025

@author: Quint + Gemini 3 Pro
"""


import numpy as np
import pandas as pd
import os

# --- CONFIGURATION --- 
#These are all people with sleep apnea in their medical history
trainers = [2,3,5,6,7,17,21]
validators = [22,23]
testers = [24,28]

# trainers = [2,3,5,6,7,17,21, 22, 23, 24, 28,30, 34, 37, 38 ,39, 40, 46, 47, 48, 51, 53, 54, 55, 57, 59] 
# validators = [61, 62, 66, 74, 75, 86 ]
# testers = [89, 90, 97, 99, 100, 102]

fs = 100 #frequency is 100 Hz
epoch_time = 30
epoch_size = fs * epoch_time # 3000 timestamps
epochs_per_person = 350   # The number of epochs you want per person. This only excludes one subject 

# Paths
folder_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/datasets/dreamt-dataset/dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0/data_100Hz/'
output_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data'

output_x_path_train = os.path.join(output_path, 'train_data_XL.npy')
output_x_path_val = os.path.join(output_path, 'val_data_XL.npy')
output_x_path_test = os.path.join(output_path, 'test_data_XL.npy')

def filepath(folder_base, i):
    return folder_base + f"S{i:03d}_PSG_df_updated.csv"

# Subject Loader
def load_subject_df(subject_id):
    
    path = filepath(folder_path, subject_id)
    if not os.path.exists(path): #Makes sure that the whole protocol doesnt break if one path does not exist
        return None

    # Columns we want to extract from the larger file
    cols = ['Sleep_Stage','TIMESTAMP','C4-M1','F4-M1','O2-M1','Fp1-O2',
            'T3 - CZ','CZ - T4','SAO2','Central_Apnea', 'Obstructive_Apnea', 
            'PTAF','FLOW', 'THORAX', 'ABDOMEN'] 
    
    # Load Data
    
    df = pd.read_csv(path, usecols=cols)
    
    #Clean Apnea Columns (Fill NaNs with 0)
    df['Central_Apnea'] = df['Central_Apnea'].fillna(0)
    df['Obstructive_Apnea'] = df['Obstructive_Apnea'].fillna(0)
    
    #Filter Sleep Stages
    asleep = (df['Sleep_Stage'] != 'P') & (df['Sleep_Stage'] != 'W') & (df['Sleep_Stage'] != 'Missing')
    df = df[asleep].copy()
    
    #Map Sleep Stages to Numbers 
    stage_mapping = {'N1': 1.0, 'N2': 2.0, 'N3': 3.0, 'R': 4.0}
    df['Sleep_Stage'] = df['Sleep_Stage'].map(stage_mapping)
    
    
    # Forces column order
    df = df[cols]
    
    return df

# Calculates size for to make to format the npy array
def calculate_dataset_shape(people_set):
    print("Step 1: Calculating total dataset size...")
    valid_subjects = 0
    n_channels = 0
    
    for person in people_set:
        df = load_subject_df(person)
        current_epochs = len(df) // epoch_size
        if current_epochs < epochs_per_person: #Makes sure to skip people who didn't sleep enough
            print(f"  Subject {person}: Not enough data. Skipping.")
            continue
            
        if n_channels == 0:
            n_channels = df.shape[1]
            
        valid_subjects += 1
        
    total_epochs = valid_subjects * epochs_per_person
    print(f"Final Count: {valid_subjects} subjects -> {total_epochs} total epochs.")
    print(f"Dimensions: ({total_epochs}, {epoch_size}, {n_channels})")
    
    return total_epochs, n_channels

# Fill the np file
def build_formatted_dataset(people_set, output_x_path):
    
    total_N, n_channels = calculate_dataset_shape(people_set)
    
    if total_N == 0:
        print("No valid data found.")
        return 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create Memory-Mapped File
    fp = np.lib.format.open_memmap(
        output_x_path, 
        mode='w+', 
        dtype='float32', 
        shape=(total_N, epoch_size, n_channels)
    )
    
    current_idx = 0
    
    print(f"\nStep 2: Writing data to {output_x_path}...")
    for i in people_set:
        df = load_subject_df(i)
        
        if df is None: continue 
        n_epochs = len(df) // epoch_size
        if n_epochs < epochs_per_person: continue 
        
        # Prepare Data
        n_samples = n_epochs * epoch_size
        clean_data = df.iloc[:n_samples].values.astype('float32')
        Data = clean_data.reshape(n_epochs, epoch_size, n_channels)
        
        #HOMOGENIZATION 
        # We loop through both apnea columns
        targets = ['Central_Apnea', 'Obstructive_Apnea']
        
        for target_col in targets:
            # Find Column Index
            col_idx = df.columns.get_loc(target_col)
            
            # Extract Data
            raw_channel = Data[:, :, col_idx]
            
            # Calculate Fraction (Mean)
            fraction = np.mean(raw_channel, axis=1) # Shape: (n_epochs,)
            
            # Apply Threshold (> 1/3)
            # If more than 33% of the epoch is apnea, label whole epoch as 1
            new_labels = (fraction > 0.33).astype('float32')
            
            # Overwrite Data (Broadcast back to 3000 samples)
            Data[:, :, col_idx] = np.repeat(new_labels[:, np.newaxis], epoch_size, axis=1)

        
        # Slice to exact size
        Data = Data[:epochs_per_person]
        
        # Write to disk
        end_idx = current_idx + epochs_per_person
        fp[current_idx:end_idx] = Data
        
        current_idx = end_idx
        print(f"  Subject {i}: Saved {epochs_per_person} epochs. (Labels Cleaned)")
        
    del fp 
    print(f"SUCCESS: Saved to {output_x_path}")
    return current_idx

# --- RUN IT ---
build_formatted_dataset(trainers, output_x_path_train)
build_formatted_dataset(validators, output_x_path_val)
build_formatted_dataset(testers, output_x_path_test)
#%%
BIG_FILE_PATH = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data/train_data_XL.npy'
dataset = np.load(BIG_FILE_PATH)

#%% A bit to inpsect the actual csv file

df = pd.read_csv(filepath(folder_path, 2))

# Display the first few rows to confirm the import
print(df.head())
    
# Check the data structure and types
print(df.info())



