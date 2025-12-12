"""
train.py

"""

import tensorflow as tf
import keras
import sklearn
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from models import CNNTransformerHybrid

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the CNN-Transformer hybrid model.
    """
    model = CNNTransformerHybrid(
            n_channels=6,
            sequence_length=3000,
            n_classes=2,
            d_model=128,
            num_heads=8,
            num_transformer_layers=2
        )
    learning_rate = 1e-4      

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # compute_class_weight returns weights for {0: weight_normal, 1: weight_apnea}

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=[0, 1],
        y=y_train
    )
    class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}

    print("Class weights being used:", class_weight)


    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]


    # train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,       
        batch_size=32,      
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )


    return model, history



if __name__ == '__main__':
    pass
