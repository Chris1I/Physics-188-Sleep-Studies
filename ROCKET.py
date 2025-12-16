# -*- coding: utf-8 -*-
"""
ROCKET Pipeline for Sleep Apnea Detection

Created on Sun Dec  7 16:44:58 2025

@author: Quint + Gemini 3 Pro
"""

import numpy as np
import os
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("Loading Data")

# Update path if necessary
data_path = 'C:/Users/Quint/OneDrive/Documenten/Berkeley/EEG/Physics-188-Sleep-Studies/data'

train_file = os.path.join(data_path, 'train_data.npy')
test_file  = os.path.join(data_path, 'test_data.npy')

if not os.path.exists(train_file):
    raise FileNotFoundError(f"Could not find {train_file}. Did you run the split script?")

# Loading the files
X_train_set = np.load(train_file)
X_test_set  = np.load(test_file)

print(f"   Train shape: {X_train_set.shape}")
print(f"   Test shape:  {X_test_set.shape}")



# Data prep
# Index = 1 (C4-M1) and Labels are last column
X_train_raw = X_train_set[:, :, 1] 
y_train     = X_train_set[:, 0, -1] 

X_test_raw  = X_test_set[:, :, 1]
y_test      = X_test_set[:, 0, -1]

#Reshape for ROCKET 

print("Reshaping for ROCKET...")
X_train_rocket = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
X_test_rocket  = X_test_raw.reshape(X_test_raw.shape[0], 1, X_test_raw.shape[1])

print(f"   ROCKET Input Shape: {X_train_rocket.shape}")

print("Initializing ROCKET (Transformation + Classifier)...")

# ROCKET: 10,000 kernels 
# Transforms raw waves into feature vectors.
rocket = Rocket(num_kernels=10_000, random_state=42)

# Classifier: Ridge Regression
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

# Pipeline
model = make_pipeline(rocket, classifier)

print("Training...")
model.fit(X_train_rocket, y_train)

# Evaluate/predictions
print("Evaluating...")
y_pred = model.predict(X_test_rocket)

acc = accuracy_score(y_test, y_pred)
print(f"FINAL ACCURACY: {acc:.2%}")
cm = confusion_matrix(y_test, y_pred)


# Plotting
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Apnea (1)"])
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.savefig("Confusion_ROCKET.pdf")
plt.show()

