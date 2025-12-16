import numpy as np

from models.train import train_model
from models.cnn_base import CNNTransformerHybrid


def extract_participant(person_array):
    """
    person_array has shape: (350, 11)
    9 feature columns each contain array of length 3000
    
    returns:
        X: (350, 3000, 9)
        y: (350,)
    """
    X = []
    y = []
    for row in person_array: # each person will be listed as person[x].npy in the data set. first 7 are training, 8&9 are test, 10&11 are validation
        channels = np.stack(row[1:10], axis=-1)  # (3000, 9)
        X.append(channels)
        y.append(row[10])

    return np.array(X), np.array(y)

def load_training_data():
    X_list, y_list = [], []

    for person_id in range(1, 8):  # people 1 to 7
        print(f"Loading person {person_id}...")
        arr = np.load(f"data/person{person_id}.npy", allow_pickle=True)
        Xi, yi = extract_participant(arr)
        X_list.append(Xi)
        y_list.append(yi)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    print("Training data shapes:")
    print("X_train:", X_train.shape)  # expect (2450, 3000, 9) for 7 participants
    print("y_train:", y_train.shape)

    return X_train, y_train

def load_test_data():
    X_list, y_list = [], []

    for person_id in [8, 9]:
        print(f"Loading test {person_id}...")
        arr = np.load(f"data/person{person_id}.npy", allow_pickle=True)
        Xi, yi = extract_participant(arr)
        X_list.append(Xi)
        y_list.append(yi)

    X_test = np.concatenate(X_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)

    print("Test data shapes:")
    print("X_test:", X_test.shape)   # Expect (700, 3000, 9)
    print("y_test:", y_test.shape)

    return X_test, y_test


def main():

    # this was for our initial cnn/transform
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    print("\nStarting training...\n")
    model, history = train_model(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        model_type="hybrid"
    )


    print("\nEvaluating on test subjects")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {acc:.4f}")

    model.save("saved_models/hybrid_model")
    np.save("saved_models/hybrid_history.npy", history.history)

    print("\nTraining complete. Model saved to saved_models/.\n")

# need to add what outputs we need to validate

if __name__ == "__main__":
    main()
