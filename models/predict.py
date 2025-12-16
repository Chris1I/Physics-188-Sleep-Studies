import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import ApneaCNNTransformer
from attention import PositionalEncoding, TransformerBlock, MultiHeadAttention

# very low optimized threshhold because we minimizing false negatives but possibly generating false positives
# the goal is screening. Since our patients have a history of apnea we are more interested in predicting the severity. 
# false positives are fine because we intend this to be a tool for helping a human doctor assess the severity and treat sleep apnea
# also detailed in report
OPTIMIZED_THRESHOLD = 0.0052

# predicting on test data using best model from training runs
def predict():
    print("Loading model...")
    custom_objects = {
        'ApneaCNNTransformer': ApneaCNNTransformer,
        'PositionalEncoding': PositionalEncoding,
        'TransformerBlock': TransformerBlock,
        'MultiHeadAttention': MultiHeadAttention
    }
    model = keras.models.load_model('models/best_model.keras', custom_objects=custom_objects)
    
    print("Loading test data...")
    X_test = np.load('processed_data/X_test.npy')
    
    print("Predicting...")
    logits = model.predict(X_test, batch_size=32, verbose=1)
    probabilities = tf.nn.softmax(logits).numpy()
    
    predictions_default = np.argmax(logits, axis=1)
    predictions_optimized = (probabilities[:, 1] >= OPTIMIZED_THRESHOLD).astype(int)
    
    np.save('predictions/predictions_default.npy', predictions_default)
    np.save('predictions/predictions_optimized.npy', predictions_optimized)
    np.save('predictions/probabilities.npy', probabilities)
    
    print(f"Predictions saved to predictions/")
    print(f"Default threshold (0.5): {np.sum(predictions_default==1)}/{len(predictions_default)} positive")
    print(f"Optimized threshold ({OPTIMIZED_THRESHOLD}): {np.sum(predictions_optimized==1)}/{len(predictions_optimized)} positive")

if __name__ == '__main__':
    predict()