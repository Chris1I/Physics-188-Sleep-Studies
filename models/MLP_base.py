# models/MLP_base.py

import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_mlp_model(
    sequence_length: int = 3000,
    n_channels: int = 9,
    n_classes: int = 1,
    # per-timestep feature sizes (this is your “MLP over channels at each time step”)
    time_mlp_dims=(64, 128),
    # head after pooling
    head_dims=(128, 64),
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    "Exclusive MLP" with no flatten over time.
    Uses per-timestep Dense layers + global pooling to reduce false positives
    and improve calibration.
    """

    inputs = keras.Input(shape=(sequence_length, n_channels), name="signals")

    # normalize per channel across the window (more stable than BN for this use)
    x = layers.LayerNormalization(name="ln_in")(inputs)

    # per-timestep MLP (Dense acts on last dimension at each time step)
    for i, units in enumerate(time_mlp_dims):
        x = layers.Dense(units, activation="relu", name=f"time_dense_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"time_dropout_{i+1}")(x)

    # global pooling over time 
    x_avg = layers.GlobalAveragePooling1D(name="gap")(x)
    x_max = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([x_avg, x_max])

    # small head
    for i, units in enumerate(head_dims):
        x = layers.Dense(units, activation="relu", name=f"head_dense_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"head_dropout_{i+1}")(x)

    # outputs
    if n_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="apnea_prob")(x)
    else:
        outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MLPGlobalPoolApneaModel")
    return model


def MLPApneaDetector(
    n_channels: int = 9,
    sequence_length: int = 3000,
    n_classes: int = 1,
    **kwargs,
) -> keras.Model:
    return build_mlp_model(
        sequence_length=sequence_length,
        n_channels=n_channels,
        n_classes=n_classes,
        **kwargs,
    )
