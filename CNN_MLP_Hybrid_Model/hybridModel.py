import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from cnn_base import CNNTransformerHybrid  

class SleepHybridModel(Model):
    def __init__(self, cnn_model_config, mlp_input_dim):
        super(SleepHybridModel, self).__init__()

        self.cnn_model_config = cnn_model_config
        self.mlp_input_dim = mlp_input_dim
        
        #CNN
        self.cnn_branch = CNNTransformerHybrid(**cnn_model_config)
        
        #MLP
        self.mlp_dense1 = Dense(64, activation='relu')
        self.mlp_bn1 = BatchNormalization()
        self.mlp_dense2 = Dense(32, activation='relu')
        
        #concatenate these two
        self.concat = Concatenate()
        
        self.final_dense1 = Dense(64, activation='relu')
        self.final_dropout = Dropout(0.3)
        self.final_output = Dense(1, activation='sigmoid') 

    def call(self, inputs, training=False):
        # inputs is now a list: [raw_signal, physics_features]
        raw_input = inputs[0]
        physics_input = inputs[1]
        
        #CNN
        cnn_features = self.cnn_branch(raw_input, training=training, return_features=True)
        
        #MLP
        mlp_out = self.mlp_dense1(physics_input)
        mlp_out = self.mlp_bn1(mlp_out, training=training)
        mlp_out = self.mlp_dense2(mlp_out)
        
        combined = self.concat([cnn_features, mlp_out])
        
        z = self.final_dense1(combined)
        z = self.final_dropout(z, training=training)
        return self.final_output(z)

    def get_config(self):
        config = super(SleepHybridModel, self).get_config()
        config.update({
            "cnn_model_config": self.cnn_model_config,
            "mlp_input_dim": self.mlp_input_dim,
        })
        return config