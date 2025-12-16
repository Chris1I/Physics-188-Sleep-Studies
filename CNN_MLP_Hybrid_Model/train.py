import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from cnn_base import CNNTransformerHybrid
from hybridModel import SleepHybridModel

def load_data():
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_val = np.load('processed_data/X_val.npy')
    y_val = np.load('processed_data/y_val.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    print(f"\nTrain: {X_train.shape}, {len(np.unique(y_train))} classes")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        print(f"{name} - NaN: {np.isnan(X).any()}, Inf: {np.isinf(X).any()}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    weights = weights * 2.0
    
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    
    print("\n" + "="*60)
    print("CLASS WEIGHTS (BOOSTED)")
    print("="*60)
    stage_names = {0: 'N1', 1: 'N2', 2: 'N3', 3: 'REM'}
    for cls, weight in class_weight_dict.items():
        print(f"  Stage {cls} ({stage_names.get(cls, '?')}): {weight:.3f}")
    
    return class_weight_dict

def create_improved_model():
    print("\n" + "="*60)
    print("CREATING IMPROVED MODEL")
    print("="*60)
    
    inputs = layers.Input(shape=(3000, 11))
    x = inputs
    
    x = layers.Conv1D(64, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(4)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SleepStageClassifier')
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TerminateOnNaN()
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    logits = model.predict(X_test, verbose=1)
    
    if np.isnan(logits).any():
        print("\nWARNING: Model produced NaN values!")
        return None, None
    
    probabilities = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(logits, axis=1)
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("         Predicted")
    stage_names = {0: 'N1', 1: 'N2', 2: 'N3', 3: 'REM'}
    stages = sorted(np.unique(y_test))
    
    header = "Actual   " + "  ".join([f"{stage_names.get(s, str(s)):>4s}" for s in stages])
    print(header)
    for i, actual_stage in enumerate(stages):
        row = f"{stage_names.get(actual_stage, str(actual_stage)):>6s}   "
        row += "  ".join([f"{cm[i,j]:4d}" for j in range(len(stages))])
        print(row)
    
    print("\nPer-class Metrics:")
    for i, stage in enumerate(stages):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"  {stage_names[stage]:>3s}: Recall={recall*100:5.1f}% Precision={precision*100:5.1f}%")
    
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=[stage_names.get(s, str(s)) for s in stages]))
    
    return y_pred, probabilities

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    print("\nSaved: results/training_history.png")
    plt.close()

def main():

    #TRAINING DATA
    # Load Raw Data (N, 3000, 9)
    X_train_raw = np.load('X_train_raw.npy', mmap_mode='r')
    
    #physics feature data
    X_train_feat = np.load('X_train_features.npy', mmap_mode='r') 
    y_train = np.load('Y_train_labels.npy', mmap_mode='r')
    
    #VALIDATION DATA
    X_val_raw = np.load('X_val_raw.npy', mmap_mode='r')
    
    #physics feature data
    X_val_feat = np.load('X_val_features.npy', mmap_mode='r') 
    y_val = np.load('Y_val_labels.npy', mmap_mode='r')
    
    

    cnn_config = {
        'num_layers': 4, 
        'd_model': 64, 
        'num_heads': 4, 
        'dff': 512, 
        'num_classes': 1, 
        'dropout_rate': 0.1
    }
    
    #input type of model
    model = SleepHybridModel(cnn_model_config=cnn_config, mlp_input_dim=12)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    
    # dual input training
    print("Starting Hybrid Training...")
    
    class_weights = compute_class_weights(y_train)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_hybrid_model.keras', 
            monitor='val_recall',               
            mode='max',                         
            save_best_only=True,                
            verbose=1
        ),
        
        #stop early option
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x=[X_train_raw, X_train_feat],
        y=y_train,
        validation_data=([X_val_raw, X_val_feat], y_val),
        epochs=50, 
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks  
    )
    
    # 3. Save the Final Model (just in case)
    model.save('final_hybrid_model.keras')
    print("Models saved: 'best_hybrid_model.keras' and 'final_hybrid_model.keras'")

    print("\n--- Evaluating on Test Set ---")

    del X_train_raw, X_train_feat, y_train
    import gc
    gc.collect()
    
    #TEST DATA
    X_test_raw = np.load('X_test_raw.npy', mmap_mode='r')
    
    #physics feature data
    X_test_feat = np.load('X_test_features.npy', mmap_mode='r') 
    y_test = np.load('Y_test_labels.npy', mmap_mode='r')
    

    # 1. Make Predictions
    # IMPORTANT: Pass the two inputs as a list, just like in .fit()
    y_pred_probs = model.predict([X_test_raw, X_test_feat], verbose=0)
    
    # 2. Apply Threshold (e.g., 0.5)
    # If your recall is low, try lowering this to 0.3 or 0.4
    y_pred = (y_pred_probs > 0.4).astype(int)
    
    # 3. Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Plot and Save
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',          # 'd' means integer (no decimals)
        cmap='Blues',     # Color scheme
        xticklabels=['Normal (0)', 'Apnea (1)'], 
        yticklabels=['Normal (0)', 'Apnea (1)']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Hybrid Model')
    
    # Save the plot to a file
    plt.savefig('confusion_matrix_result.png')
    print("Matrix saved as 'confusion_matrix_result.png'")
    
    # 5. Print Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Apnea']))

if __name__ == '__main__':
    main()