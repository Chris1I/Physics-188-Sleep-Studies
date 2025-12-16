# -*- coding: utf-8 -*-
"""
Does the Random forest training and evaluation improperly with pooled datasets
Created on Mon Dec 15 14:21:17 2025

@author: Quint + Gemini 3 Pro
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 
fs = 100
file_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data/balanced_training_data.npy'

# HJORTH PARAMETERS (Measure of Signal Chaos) 
def hjorth_params(x):
    # Activity = Variance of signal
    activity = np.var(x)
    
    # Mobility = sqrt(Var(first_derivative) / Var(signal))
    # It estimates the mean frequency
    first_deriv = np.diff(x)
    if activity == 0: #To deal with 0/0 devidision
        mobility = 0
    else:
        mobility = np.sqrt(np.var(first_deriv) / activity)
    
    # Complexity = Mobility(first_deriv) / Mobility(signal)
    # It estimates how much the signal looks like a sine wave vs noise
    deriv_2 = np.diff(first_deriv)
    
    if mobility == 0: #To deal with 0/0 devidision
        complexity = 0
    else:
        mobility_deriv = np.sqrt(np.var(deriv_2) / np.var(first_deriv))
        complexity = mobility_deriv / mobility
    
    return activity, mobility, complexity

# Feature extraction
def extract_features(epoch, fs):
    features = []
    
    # FREQUENCY DOMAIN (The classic EEG analysis bands used in industry)
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
    
    # Relative Power 
    features.extend([p / (total_power + 1e-6) for p in band_powers])

    # TIME DOMAIN (The "Shape" of the wave)
    # Standard Deviation 
    features.append(np.std(epoch))
    
    # Skewness 
    features.append(skew(epoch))
    
    # Kurtosis 
    features.append(kurtosis(epoch))
    
    # COMPLEXITY (Hjorth Parameters)
    act, mob, comp = hjorth_params(epoch)
    features.extend([act, mob, comp])
    
    return features


print("Loading Data...")
dataset = np.load(file_path)


# Channel Index 1 (C4-M1) as EEG.
X_raw = dataset[:, :, 1] 
y = dataset[:, 0, -1] # Last column is label apnea

print("2. Extracting Features...")
X_features = []
for i in range(len(X_raw)):
    feats = extract_features(X_raw[i], fs)
    X_features.append(feats)

X_features = np.array(X_features)
print(f"   Feature Matrix: {X_features.shape}")

# Handle NaNs/Infinities from math errors
X_features = np.nan_to_num(X_features)

# SCALING 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("3. Training  Model...")
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Results
print("\n--- RESULTS ---")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Plotting
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Apnea (1)"])
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.savefig("Confusion_LeakyForest.pdf")
plt.show()
