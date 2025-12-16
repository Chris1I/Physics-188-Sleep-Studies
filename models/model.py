import tensorflow as tf
from tensorflow import keras
from keras import layers
from attention import PositionalEncoding, TransformerBlock

"""
Defines the hybrid CNN-Transformer NN for apnea detection using only respiratory features
Processes time series data (sleep epochs) using a 
    - 1d CNN for local feature extraction and,
    - tranformer layers for long range dependencies
"""

class ApneaCNNTransformer(keras.Model):
    """
    Keras model for combining convolutional and attentional mechanisms
    """
    
    def __init__(
        self, 
        n_channels=4, # for each engineered respiratory feature
        sequence_length=3000, # length of time series 30 seconds * 100hz per epoch
        n_classes=2, # output classes: apnea vs normal epoch
        d_model=64, #embedding dim for transformer
        num_heads=4, # number of attention heads
        num_transformer_layers=1, # only one transformer layer
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.cnn_blocks = []
        
        # initialize 3 CNN blocks. each performs convolution, batch normalization, activation and pooling to extract features
        for filters in [32, 64, 64]:
            block = [
                # large kernel for capturing time dependent patters
                layers.Conv1D(filters=filters, kernel_size=7, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling1D(pool_size=4),
                layers.Dropout(0.2)
            ]
            self.cnn_blocks.append(block)
        
        self.projection = layers.Dense(d_model)
        # downsampling sequence_length by MaxPooling1D with pool size = 4 -> 4^3 downsample
        self.pos_encoding = PositionalEncoding(sequence_length // 64, d_model)
        # projects cnn filters to transformer dimension if necessary (not used because num of filters = d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_transformer_layers)]
        # compresses time dimension so that output is 2d tensor
        self.global_pool = layers.GlobalAveragePooling1D()
        # dropout after transformer layer
        self.dropout = layers.Dropout(0.4)
        # compressing output from transformer to prepare to classify data at the end 
        self.dense1 = layers.Dense(32, activation='relu')
        # dropout on last layer
        self.dropout2 = layers.Dropout(0.3)
        # classifier to determine if normal/apnea
        self.classifier = layers.Dense(n_classes)
    
    def call(self, inputs, training=None):
        """
        inputs 
        -> CNN pass through stacked blocks 
        -> positional encoding 
        -> transformer attention
        -> global pooling over time dimension
        -> classification
        """
        x = inputs
        for block in self.cnn_blocks:
            for layer in block:
                x = layer(x, training=training)
        x = self.projection(x)
        x = self.pos_encoding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)