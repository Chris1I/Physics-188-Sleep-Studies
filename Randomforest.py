# -*- coding: utf-8 -*-
"""
Does the Random forest training and evaluation properly with SID split datasets
Created on Sun Dec 14 12:47:54 2025

@author: Quint + Gemini 3 Pro
"""

import numpy as np
import os
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# --- CONFIG ---
fs = 100
data_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data'

train_file = os.path.join(data_path, 'train_data.npy')
test_file  = os.path.join(data_path, 'test_data.npy')

# --- HELPER: HJORTH PARAMETERS ---
def hjorth_params(x):
    activity = np.var(x)
    diff1 = np.diff(x)
    mobility = np.sqrt(np.var(diff1) / activity)
    diff2 = np.diff(diff1)
    complexity = (np.sqrt(np.var(diff2) / np.var(diff1))) / mobility
    return activity, mobility, complexity

# --- FEATURE EXTRACTION ---
def extract_robust_features(epoch, fs):
    features = []
    
    # 1. Frequency Domain
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
    freqs, psd = welch(epoch, fs, nperseg=fs*2)
    
    total_power = 0
    band_powers = []
    for low, high in bands.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        p = np.trapz(psd[idx], freqs[idx])
        band_powers.append(p)
        total_power += p
    
    features.extend(band_powers)
    features.extend([p / (total_power + 1e-6) for p in band_powers]) # Relative power

    # 2. Time Domain
    features.append(np.std(epoch))
    features.append(skew(epoch))
    features.append(kurtosis(epoch))
    
    # 3. Complexity
    act, mob, comp = hjorth_params(epoch)
    features.extend([act, mob, comp])
    
    return features

def prepare_data(filename):
    print(f"Loading {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
        
    dataset = np.load(filename)
        
    # Extract Channel 1 (C4-M1)
    X_raw = dataset[:, :, 1]
    y = dataset[:, 0, -1]
    
    print(f"  Extracting features for {len(X_raw)} epochs...")
    X_features = []
    for i in range(len(X_raw)):
        feats = extract_robust_features(X_raw[i], fs)
        X_features.append(feats)
        
    return np.array(X_features), y

# --- MAIN ---

#Load and Process Train
X_train, y_train = prepare_data(train_file)

#Load and Process Test
X_test, y_test = prepare_data(test_file)

# --- CLEANING NANS ---
print("Cleaning NaN values from math errors...")
# Replace NaN (Not a Number) and Infinity with 0
X_train = np.nan_to_num(X_train)
X_test  = np.nan_to_num(X_test)

# Scale Features
print("Scaling data...")
scaler = StandardScaler()
# Fit only on TRAIN, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Model
print("\nTraining Gradient Boosting Model...")
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
clf.fit(X_train_scaled, y_train)

# Evaluate
print("\n--- RESULTS ---")
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)


# 2. Plot it nicely
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Apnea (1)"])
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.savefig("Confusion_Forest.pdf")
plt.show()
