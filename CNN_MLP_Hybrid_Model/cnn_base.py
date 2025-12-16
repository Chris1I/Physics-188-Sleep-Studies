import tensorflow as tf
from tensorflow import keras
from keras import layers
from attention import PositionalEncoding, TransformerBlock

class CNNTransformerHybrid(keras.Model):
    
    def __init__(
        self,
        n_channels: int = 11,
        sequence_length: int = 3000,
        n_classes: int = 4,
        d_model: int = 128,      
        num_heads: int = 8,      
        num_transformer_layers: int = 2,
        name: str = 'CNNTransformerHybrid',
        **kwargs
    ):

        clean_kwargs = kwargs.copy()
        keys_to_remove = ['num_layers', 'd_model', 'num_heads', 'dff', 
                          'input_shape', 'num_classes', 'dropout_rate']
        
        for key in keys_to_remove:
            clean_kwargs.pop(key, None)

        super().__init__(name=name, **clean_kwargs)
        
        self.d_model = d_model
        
        self.cnn_blocks = []
        filters = [32, 64, 64, 64]
        
        for f in filters:
            block = [
                layers.Conv1D(filters=f, kernel_size=3, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2)
            ]
            self.cnn_blocks.append(block)
        
        reduced_length = sequence_length // 16
        self.pos_encoding = PositionalEncoding(reduced_length, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads)
            for i in range(num_transformer_layers)
        ]
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(n_classes)
    
    def call(self, inputs, training=None, return_features=False):
        x = inputs
        
        for block in self.cnn_blocks:
            for layer in block:
                x = layer(x, training=training)
        
        x = self.pos_encoding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        #x = self.classifier(x)

        if return_features:
            return x
        
        return self.dense(x)
