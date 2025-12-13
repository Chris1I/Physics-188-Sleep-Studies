"""
models/attention.py

Transformer attention mechanisms for sleep apnea detection.
Based on "Attention Is All You Need" (Vaswani et al., 2017)

KEY CONCEPTS FROM SLIDES:
=========================
1. Positional Encoding: Tells model WHERE in time each signal occurs
2. Multi-Head Attention: Learns WHAT patterns relate to EACH OTHER
3. Self-Attention: Each time step attends to all other time steps

SLIDE EQUATIONS:
================
- Positional Encoding: E(p, 2k) = sin(p / 10000^(2k/d))
                       E(p, 2k+1) = cos(p / 10000^(2k/d))
- Attention: Y = softmax(QK^T / √d_k) V
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np


# ============================================================
# POSITIONAL ENCODING (From Slide 23)
# ============================================================
class PositionalEncoding(layers.Layer):
    """
    Adds positional information to input sequences using sine/cosine functions.
    
    From slides: "positional encoding (location of token in a sequence)"
    
    WHY WE NEED THIS:
    - Transformers have no inherent sense of order (unlike RNNs/CNNs)
    - For sleep apnea: Model needs to know "this oxygen drop happened 
      10 seconds after breathing stopped"
    
    FORMULA FROM SLIDE 23:
    even dimensions: E(p, 2k) = sin(p / 10000^(2k/d))
    odd dimensions:  E(p, 2k+1) = cos(p / 10000^(2k/d))
    
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
        TODO: Implement the sine/cosine positional encoding from Slide 23
        
        Steps:
        1. Create position array: [0, 1, 2, ..., length-1]
        2. Create div_term using the formula: exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
        3. Apply sin to even indices
        4. Apply cos to odd indices
        5. Return as TensorFlow constant
        
        Hint: Look at Slide 23 for the exact formula
        """
        position = np.arange(self.length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * - (np.log(10000.0) / self.d_model))

        pe = np.zeros((self.length, self.d_model))

        pe[0::2] = np.sin(position * div_term)
        pe[1::2] = np.cos(position * div_term)

        return tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        """Add positional encoding to input"""
        # From slides: Simply add positional encoding to input embeddings
        return x + self.pos_encoding


# ============================================================
# MULTI-HEAD ATTENTION (From Slides 48-51)
# ============================================================
class MultiHeadAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.
    
    From slides: "we want to recognize many underlying patterns 
                  → multiple attention layers in parallel"
    
    WHAT THIS DOES FOR SLEEP APNEA:
    - Head 1 might learn: "airflow stops → oxygen drops"
    - Head 2 might learn: "oxygen drops → heart rate increases"
    - Head 3 might learn: "heart rate spikes → arousal happens"
    - etc.
    
    ARCHITECTURE FROM SLIDES:
    Input (N, E) → 
        Q = Input @ W_Q  (Query: what am I looking for?)
        K = Input @ W_K  (Key: what do I contain?)
        V = Input @ W_V  (Value: what should I output?)
    
    Attention = softmax(QK^T / √d_k) V
    
    Args:
        d_model: Model dimension (128)
        num_heads: Number of attention heads (8 from slides)
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # From slides: M = 64 when E = 512, H = 8
        
        # ============================================================
        # WEIGHT MATRICES (From Slide 48)
        # ============================================================
        # TODO: Create three Dense layers for Q, K, V projections
        # These are the "learnable" W(Q), W(K), W(V) from the slides
        
        # self.W_q = ???
        # self.W_k = ???
        # self.W_v = ???
        
        # YOUR CODE HERE
        
        # Final output projection (from slides: Ω matrix, Slide 51)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        
        From slides: We split E dimensions into H heads of dimension M
        Shape: (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        
        TODO: Implement this reshape operation
        Hint: Use tf.reshape and tf.transpose
        """
        # YOUR CODE HERE
        pass
    
    def call(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        
        STEPS FROM SLIDES:
        1. Compute Q, K, V (Slide 48)
        2. Split into multiple heads (Slide 48)
        3. Compute attention scores: QK^T (Slide 49)
        4. Scale by √d_k (Slide 45: "dot product scales with E")
        5. Apply softmax (Slide 49)
        6. Apply attention to V (Slide 50)
        7. Concatenate heads (Slide 51)
        8. Final projection (Slide 51)
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask (for masked attention, Slide 57)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = tf.shape(x)[0]
        
        # TODO: 
        # 1. Compute Q, K, V by passing x through W_q, W_k, W_v
        # 2. Split heads
        # 3. Compute attention: softmax(QK^T / √d_k) V
        # 4. Concatenate heads
        # 5. Final projection
        
        # YOUR CODE HERE
        pass


# ============================================================
# TRANSFORMER BLOCK (From Slide 52)
# ============================================================
class TransformerBlock(layers.Layer):
    """
    Single transformer encoder block.
    
    From Slide 52: "Encoder" architecture
    Components:
    1. Multi-Head Self-Attention
    2. Add & Normalize (residual connection + layer norm)
    3. Feed Forward Network
    4. Add & Normalize (residual connection + layer norm)
    
    PHYSICAL INTERPRETATION FOR SLEEP APNEA:
    - Attention layer: Learns temporal relationships between signals
      (e.g., "oxygen drop correlates with airflow cessation 10 seconds ago")
    - Feed forward: Processes each time step independently
      (e.g., "this particular heart rate value suggests arousal")
    
    Args:
        d_model: Model dimension (128)
        num_heads: Number of attention heads (8)
        dff: Feed-forward dimension (usually 4 * d_model = 512)
    """
    
    def __init__(self, d_model, num_heads, dff=None):
        super().__init__()
        
        if dff is None:
            dff = d_model * 4  # Standard practice from slides
        
        # ============================================================
        # ATTENTION LAYER (Slide 52: "Self-Attention")
        # ============================================================
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # ============================================================
        # FEED-FORWARD NETWORK (Slide 52: "Feed Forward")
        # ============================================================
        # TODO: Create a 2-layer feed-forward network
        # Architecture: Dense(dff, relu) → Dense(d_model)
        # Hint: Use keras.Sequential
        
        # self.ffn = ???
        
        # YOUR CODE HERE
        
        # ============================================================
        # LAYER NORMALIZATION (Slide 52: "Add & Normalize")
        # ============================================================
        # TODO: Create two LayerNormalization layers
        # One after attention, one after feed-forward
        
        # self.layernorm1 = ???
        # self.layernorm2 = ???
        
        # YOUR CODE HERE
        
        # ============================================================
        # DROPOUT (For regularization)
        # ============================================================
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
    
    def call(self, x, training=None, mask=None):
        """
        Forward pass through transformer block.
        
        FROM SLIDE 52:
        1. Self-Attention
        2. Add (residual connection) & Normalize
        3. Feed Forward  
        4. Add (residual connection) & Normalize
        
        Args:
            x: Input (batch, seq_len, d_model)
            training: Boolean for dropout
            mask: Optional attention mask
        
        Returns:
            Output (batch, seq_len, d_model)
        """
        
        # TODO: Implement the transformer block architecture from Slide 52
        # 
        # Structure:
        # attn_output = self.attention(x, mask)
        # attn_output = self.dropout1(attn_output, training)
        # x = self.layernorm1(x + attn_output)  # ← Residual connection
        # 
        # ffn_output = self.ffn(x)
        # ffn_output = self.dropout2(ffn_output, training)
        # x = self.layernorm2(x + ffn_output)  # ← Residual connection
        
        # YOUR CODE HERE
        pass


# ============================================================
# HELPER FUNCTION: Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Core attention calculation from slides.
    
    FORMULA FROM SLIDE 45:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    INTUITION:
    - QK^T: How much does each query attend to each key?
    - √d_k: Scale factor (prevents softmax saturation)
    - softmax: Convert scores to probabilities
    - V: Weighted combination of values
    
    Args:
        q: Query (batch, num_heads, seq_len_q, depth)
        k: Key (batch, num_heads, seq_len_k, depth)
        v: Value (batch, num_heads, seq_len_v, depth)
        mask: Optional mask
    
    Returns:
        output: (batch, num_heads, seq_len_q, depth)
        attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
    """

    
    
    pass