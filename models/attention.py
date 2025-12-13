"""
models/attention.py

Transformer attention mechanisms for sleep apnea detection

Using transformer details outlined in lecture
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np


# ============================================================
# POSITIONAL ENCODING 
# ============================================================
class PositionalEncoding(layers.Layer):
    """
    Adds positional information to input sequences using sin/cos functions.

    Uses positional encoding formula slides from lecture
    
    Args:
        sequence_length: Number of time steps (3000 for 30 seconds @ 100Hz)
        d_model: Embedding dimension (e.g., 128)
    """
    
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)
    
    def _create_positional_encoding(self, length, d_model):
        """
        Recreates the positional encoding function from lecture
        arguments:
            - length of sequence,
            - dimension of model
        returns:
            - tensorflow constant representing the positional encoding of the sequence as a length x d_model array
        """
        position = np.arange(self.length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * - (np.log(10000.0) / self.d_model))

        pe = np.zeros((self.length, self.d_model))

        pe[0::2] = np.sin(position * div_term)
        pe[1::2] = np.cos(position * div_term)

        return tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        """Add positional encoding to input"""
        return x + self.pos_encoding


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================
class MultiHeadAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.
    
    Attention = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads 
        
        # initiliazing Q,K,V
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        
        # Final output projection 
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).

        Shape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
    
        """
        x = tf.reshape(x, shape = [batch_size, -1, self.num_heads, self.head_dim])
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    def call(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask (for masked attention, Slide 57)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = tf.shape(x)[0]
        
        # Computing Q,K,V from x
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # splitting heads
        Qh = self.split_heads(Q, batch_size=batch_size)
        Kh = self.split_heads(K, batch_size=batch_size)
        Vh = self.split_heads(V, batch_size=batch_size)

        # for computing sqrt
        d_k = tf.cast(self.head_dim, tf.float32)

        # attn
        softmax = tf.nn.softmax(tf.matmul(Qh, Kh, transpose = True) / tf.math.sqrt(d_k), axis = -1)

        attn = tn.matmul(softmax, Vh)

        return attn


# ============================================================
# TRANSFORMER BLOCK
# ============================================================
class TransformerBlock(layers.Layer):
    """
    Single transformer encoder block.
    
    Components:
    1. Multi-Head Self-Attention
    2. Add & Normalize (residual connection + layer norm)
    3. Feed Forward Network
    4. Add & Normalize (residual connection + layer norm)
    
    Args:
        d_model: Model dimension (128)
        num_heads: Number of attention heads (8)
        dff: Feed-forward dimension (usually 4 * d_model = 512)
    """
    
    def __init__(self, d_model, num_heads, dff=None):
        super().__init__()
        
        if dff is None:
            dff = d_model * 4
        
        # attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # feed forward
        #         
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation = 'relu'), 
            layers.Dense(d_model)
        ])
        
        # layer normalization
        
        self.layernorm1 = layers.LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon= 1e-6)
        
        # dropout

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
    
    def call(self, x, training=None, mask=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input (batch, seq_len, d_model)
            training: Boolean for dropout
            mask: Optional attention mask
        
        Returns:
            Output (batch, seq_len, d_model)
        """
        
        # multihead attention layer with dropout and normalization
        attn_output = self.attention(x)
        attn_output = self.dropout1(attn_output, training = training)
        x = self.layernorm1(x + attn_output)

        # feedforward layer with dropout and normalization
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training = training)
        x = self.layernorm2(x+ffn_output)
        
        return x