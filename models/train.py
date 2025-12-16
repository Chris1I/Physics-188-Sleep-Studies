# models/train.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import callbacks

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from .MLP_base import build_mlp_model


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_size: int = 32,
    max_epochs: int = 20,
    base_lr: float = 1e-4,
    max_pos_weight: float = 10.0,  
):
    """
    Train the MLP apnea detector

    Improvements made:
    - y reshaped to (N,1) for Keras AUC changes
    - best threshold chosen on VAL
    - class weights normalized + positive weight capped 
    """

    print("TRAIN_MODEL CALLED WITH max_epochs =", max_epochs)

    # Dtypes + reshape labels to (N,1)
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32").reshape(-1, 1)

    if X_val is not None:
        X_val = X_val.astype("float32")
        y_val = y_val.astype("float32").reshape(-1, 1)

    sequence_length = X_train.shape[1]
    n_channels = X_train.shape[2]

    model = build_mlp_model(
        sequence_length=sequence_length,
        n_channels=n_channels,
        n_classes=1,
    )

    # class weights
    y_train_1d = y_train.ravel().astype(np.int64)
    classes = np.array([0, 1], dtype=np.int64)

    w = compute_class_weight("balanced", classes=classes, y=y_train_1d)
    w0, w1 = float(w[0]), float(w[1])

    # normalize so negative weight = 1, then cap positive weight
    w1_norm = w1 / w0
    w1_cap = min(w1_norm, float(max_pos_weight))
    class_weight_dict = {0: 1.0, 1: w1_cap}

    print(f"Balanced weights raw: w0={w0:.3f}, w1={w1:.3f}")
    print(f"Using weights: w0=1.000, w1={class_weight_dict[1]:.3f} (capped at {max_pos_weight})")

    # ROC + PR AUC 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=base_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )

    cb = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # figure out the threshold
    best_thr = 0.5
    if X_val is not None and y_val is not None:
        y_scores_val = model.predict(X_val, batch_size=32, verbose=0).ravel()
        y_val_1d = y_val.ravel().astype(int)

        best = {"thr": 0.5, "kappa": -1, "f1": 0.0, "prec": 0.0, "rec": 0.0}
        thresholds = np.linspace(0.01, 0.99, 99)

        for thr in thresholds:
            y_pred_val = (y_scores_val >= thr).astype(int)

            kappa = cohen_kappa_score(y_val_1d, y_pred_val)
            f1    = f1_score(y_val_1d, y_pred_val, zero_division=0)
            prec  = precision_score(y_val_1d, y_pred_val, zero_division=0)
            rec   = recall_score(y_val_1d, y_pred_val, zero_division=0)

            if kappa > best["kappa"]:
                best = {"thr": float(thr), "kappa": float(kappa), "f1": float(f1), "prec": float(prec), "rec": float(rec)}

        best_thr = best["thr"]
        print(
            f"Best VAL threshold: {best_thr:.2f} | "
            f"kappa={best['kappa']:.3f} | f1={best['f1']:.3f} | "
            f"P={best['prec']:.3f} | R={best['rec']:.3f}"
        )
    else:
        print("No validation data; using default threshold 0.50")

    return model, history, best_thr
