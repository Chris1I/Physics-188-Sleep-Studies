"""
models/cnn_base.py

"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from .attention import PositionalEncoding, TransformerBlock

# cnn+transformer hybrid model
class CNNTransformerHybrid(keras.Model):
    """
    Hybrid architecture combining CNN and Transformer.
    
   
    input - TBD (batch, 3000, 9) - raw physiological signals
    output - TBD (batch, 2) - logits for [normal, apnea]
    
    """
    
    def __init__(
        self,
        n_channels: int = 6,
        sequence_length: int = 3000,
        n_classes: int = 2,
        d_model: int = 128,      
        num_heads: int = 8,      
        num_transformer_layers: int = 2,
        name: str = 'CNNTransformerHybrid'
    ):
        super().__init__(name=name)
        
        self.d_model = d_model
        
        
        self.cnn_blocks = []
        filters = [32,64,128,128] # 4 convolutional blocks
        # each block will have BatchNorm, ReLU, MaxPool, Dropout


        for f in filters:
            block = [
                layers.Conv1D(filters=f, kernel_size=3, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2)
            ]
            self.cnn_blocks.append(block) # should end with (187,128)
        
        # positional encoding from attention.py
        
        reduced_length = sequence_length // 16  # 3000 // 16 = 187
        self.pos_encoding = PositionalEncoding(reduced_length, d_model)
        
        # Stack of transformer blocks
        
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads)
            for i in range(num_transformer_layers)
        ]
        
        
        # after transformer, we need to classify
        
        
        self.global_pool = layers.GlobalAveragePooling1D() # collapse time dimension
        self.dropout = layers.Dropout(0.3) # dropout before final layer
        self.classifier = layers.Dense(n_classes) # final classification layer
    
    def call(self, inputs, training=None):
        """
        Forward pass through hybrid model.
        
        1. CNN: Extract local features
        2. Positional Encoding: Add time information
        3. Transformer: Learn long-range relationships
        4. Classifier: Predict apnea vs normal
        
        Args:
            inputs: (batch, 3000, 9) - raw physiological signals
            training: Boolean for dropout/batchnorm
            outputs: (batch, 2) - logits for [normal, apnea]
        """
        

        x = inputs
        for block in self.cnn_blocks: #pass through each cnn block
            for layer in block:
                x = layer(x, training=training) # pass training for dropout/batchnorm       

        x = self.pos_encoding(x) #add positional encoding
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        x = self.global_pool(x) # global average pooling
        x = self.dropout(x, training=training) #dropout
        x = self.classifier(x) #classification layer
        return x

