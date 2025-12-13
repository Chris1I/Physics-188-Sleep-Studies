diya
diyaslight
Do Not Disturb

Icarus — 8/29/2025 2:51 PM
Like you can advocate for them but theres no guarantee
diya — 8/29/2025 7:22 PM
Okay
He will probs be there at mine tmrw 😔
😔
Icarus — 8/29/2025 8:10 PM
Lol cool
Should we bring drinks or anything?
diya — 8/29/2025 8:16 PM
No need we have stuff!! At most maybe mix if you want anything in particular
Icarus — 8/29/2025 8:59 PM
If you have gin I’ll bring some tonic water cause I’m a gnt fiend
diya — 8/29/2025 9:00 PM
Lowk not sure I have gin but I will probably acquire during alc run because I like gin so yes do that 
diya — 8/31/2025 3:16 PM
Wait what time should I come tonight?
Icarus — 8/31/2025 3:21 PM
7pm
617-899-5755 
diya — 9/9/2025 2:31 PM
I can come for the first like 20 minutes but I have a meeting at 5:30
Icarus — 9/9/2025 2:37 PM
okay
diya — 9/9/2025 2:44 PM
I'll get one of the new officers to report on transfer reading group stuff cause we're meeting today
diya — 9/26/2025 1:29 PM
Image
did you have weird issues with running this cell too
it's not letting me type inside of it lol
i actually cant cope w this bs
jk there was a post about it in ed sorry
diya — 9/26/2025 1:59 PM
the only written answer was the last one right
Icarus — 9/26/2025 2:30 PM
yeah that cell is weird
diya — 10/13/2025 12:25 PM
alex
lets set up a meeting with markus hohle
next wee
wek
weel
week
Icarus — 10/13/2025 1:18 PM
Yah
Icarus — 11/20/2025 10:02 AM
I feel like we gotta ask how AI is affecting math class grading/exams
diya — 11/20/2025 10:04 AM
we def can i just haven't added it yet
sorry midterm in an hour and then I'll finish it
Icarus — 11/20/2025 10:04 AM
gl
diya — 11/20/2025 8:45 PM
https://www.instagram.com/reel/DQM7Cl6Ervv/?igsh=MzRlODBiNWFlZA==

jxnfrank
Tell me you’re joking 😭
 #fyp #foryou #relatable #reels #instagram #viral #viralreels #fypシ
Likes
508720

Instagram
diya — 11/21/2025 2:13 AM
also i lowk can't make officer meeting tmrw 
Luckily everyone was at town haall
diya — 12/11/2025 12:33 PM
physics_188_sleep_apnea/
├── models/
│   ├── init.py
│   ├── cnn_base.py              # Core CNN architecture
│   ├── attention.py             # Attention mechanisms
│   └── model_factory.py         # Easy model instantiation
├── config/
│   ├── init.py
│   ├── model_config.yaml        # Hyperparameters
│   └── config_loader.py         # Config parser
├── utils/
│   ├── init.py
│   └── model_utils.py           # Helper functions (weight init, etc.)
├── tests/
│   ├── init.py
│   └── test_model.py            # Unit tests with dummy data
├── notebooks/
│   └── model_exploration.ipynb  # Interactive testing
├── requirements.txt
└── README.md
https://prod.liveshare.vsengsaas.visualstudio.com/join?8A87F8176978C94AD09C923E96D834ABE363
Visual Studio Code for the Web
Build with Visual Studio Code, anywhere, anytime, entirely in your browser.
diya — 12/11/2025 2:39 PM
https://www.baeldung.com/cs/ml-relu-dropout-layers
Baeldung on Computer Science
How ReLU and Dropout Layers Work in CNNs | Baeldung on Computer Sci...
Study two fundamental components of Convolutional Neural Networks - the Rectified Linear Unit and the Dropout Layer.
How ReLU and Dropout Layers Work in CNNs | Baeldung on Computer Sci...
diya — 12/11/2025 3:00 PM
https://prod.liveshare.vsengsaas.visualstudio.com/join?8A87F8176978C94AD09C923E96D834ABE363
Visual Studio Code for the Web
Build with Visual Studio Code, anywhere, anytime, entirely in your browser.
diya — 12/11/2025 3:22 PM
we also need to build the way to run the model
lemme think
diya — 12/11/2025 4:48 PM
"""
models/attention.py

Transformer attention mechanisms for sleep apnea detection.
Based on "Attention Is All You Need" (Vaswani et al., 2017)
Expand
message.txt
12 KB
copying this here, im erasing comments before i git push in case they check version history
https://www.overleaf.com/3368764124xbqyqwvbjnmq#75d4a7
Overleaf, Online LaTeX Editor
An online LaTeX editor that’s easy to use. No installation, real-time collaboration, version control, hundreds of LaTeX templates, and more.
Image
diya — 12/11/2025 5:00 PM
ok i pushed. can u check attention.py and generally sanity check. and if ur able also add more output for when we run the model, maybe a correlation matrix or plot or somethin
Icarus — 12/11/2025 5:01 PM
real
okay I will give it a look later tn
diya — 12/11/2025 5:02 PM
ok cool
its chill i think as long as running them doesnt take too long
Icarus — Yesterday at 3:52 PM
"""
models/attention.py

Transformer attention mechanisms for sleep apnea detection

Using transformer details outlined in lecture
Expand
attention.py
7 KB
﻿
Icarus
icarus9684
 
gadzooks!
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
        ffn_output = self.ffn(x)git push -
        ffn_output = self.dropout2(ffn_output, training = training)
        x = self.layernorm2(x+ffn_output)
        
        return x
attention.py
7 KBhit 