import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length: int, d_model: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.proj = None

        pos = np.arange(sequence_length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((sequence_length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :], dtype=tf.float32)  # (1, L, D)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != self.d_model:
            self.proj = layers.Dense(self.d_model)

    def call(self, x):
        if self.proj is not None:
            x = self.proj(x)
        return x + self.pe[:, : tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        key_dim = max(1, d_model // num_heads)

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)

        self.ffn = keras.Sequential([
            layers.Dense(ff_mult * d_model, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))

        ff = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ff, training=training))
        return x
