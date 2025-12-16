import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import os
from model import ApneaCNNTransformer

## imported specific functions from this file for running in colab, 
## results file path is different

def load_data():
    """
    Data loading function for train/val, used in running the model. No arguments, use predefined paths
    """
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_val = np.load('processed_data/X_val.npy')
    y_val = np.load('processed_data/y_val.npy')
    return X_train, y_train, X_val, y_val

def compute_class_weights(y_train):
    """
    Computing class weights based off of training data using sklearn functionality
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}

def create_model():
    """
    initializing the model creating an instantce of the ApneaCNNTRansformer class and building it. 
    Then configures with Adam learning and sparse categorical cross entropy loss. 
    Returns fully compiled keras.Model object.
    """
    model = ApneaCNNTransformer(n_channels=4, sequence_length=3000, n_classes=2, 
                                d_model=64, num_heads=4, num_transformer_layers=1)
    model.build((None, 3000, 4))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    model training function.
    returns model training history using model.fit keras method.
    """
    # making sure /models directory exists for saving models
    os.makedirs('models', exist_ok=True)
    
    # using early stopping and saving best performing model by ModelCheckpoint 
    # Uses ReduceLROnPlateau to reduce learnign rate by half if validation loss stalls for 8 epochs to allow for fine tuning
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.TerminateOnNaN()
    ]
    
    # training over 100 epochs and using class_weights for lack of apnea events over the sleep data
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, 
                       batch_size=32, class_weight=class_weights, callbacks=callbacks, verbose=1)
    return history

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()
    
    print("Computing class weights...")
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    print("Creating model...")
    model = create_model()
    model.summary()
    
    print("Training...")
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)
    
    print(f"Training complete. Best model saved to models/best_model.keras")

if __name__ == '__main__':
    main()