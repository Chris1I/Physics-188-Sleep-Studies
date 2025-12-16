# main.py

import os
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    accuracy_score,
)

from models.train import train_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_SAVE_PATH   = os.path.join(SAVED_DIR, "mlp_model.keras")
HISTORY_SAVE_PATH = os.path.join(SAVED_DIR, "mlp_history.npy")

TRAIN_IDS = [2, 3, 9, 16, 20, 24, 34, 35, 36, 57, 65, 69]
VAL_IDS   = [7, 10, 39]
TEST_IDS  = [11, 17, 54, 62]


def load_person(pid):
    x_path = os.path.join(DATA_DIR, f"person{pid}.npy")
    y_path = os.path.join(DATA_DIR, f"person{pid}_labels.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def summarize_split(ids, name):
    Xs, ys = [], []
    for pid in ids:
        X, y = load_person(pid)
        print(f"{name}: person{pid} -> X {X.shape}, y {y.shape}, positives: {int(y.sum())}/{y.size}")
        Xs.append(X)
        ys.append(y)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    pos = int(y_all.sum())
    total = y_all.size
    print(f"\n{name} total: X {X_all.shape}, y {y_all.shape}, positives: {pos}/{total} ({pos/total:.4%})\n")
    return X_all, y_all


def smooth_probs(p, win=7):
    # simple moving average
    if win <= 1:
        return p
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(p, k, mode="same")


def remove_short_positive_runs(y_bin, min_len=3):
    # remove predicted apnea runs shorter than min_len
    y = y_bin.copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            if (j - i) < min_len:
                y[i:j] = 0
            i = j
        else:
            i += 1
    return y


def pretty_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:\n")
    print("                 Predicted")
    print("                 Normal  Apnea")
    print(f"Actual Normal   {tn:7d}{fp:7d}")
    print(f"       Apnea    {fn:7d}{tp:7d}\n")


def main():
    os.makedirs(SAVED_DIR, exist_ok=True)

    print("\nLoading TRAIN")
    X_train, y_train = summarize_split(TRAIN_IDS, "TRAIN")

    print("Loading VAL")
    X_val, y_val = summarize_split(VAL_IDS, "VAL")

    print("Loading TEST")
    X_test, y_test = summarize_split(TEST_IDS, "TEST")

    print("\nStarting training...\n")
    model, history, best_thr = train_model(X_train, y_train, X_val, y_val)

    model.save(MODEL_SAVE_PATH)
    np.save(HISTORY_SAVE_PATH, history.history)
    print(f"\nSaved trained MLP model to {MODEL_SAVE_PATH}")
    print(f"Saved training history to {HISTORY_SAVE_PATH}")
    print(f"Best VAL threshold to use on TEST: {best_thr:.2f}\n")

    # keras eval 
    print("\nEvaluating on TEST subjects...\n")

    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")
    y_test_eval = y_test.reshape(-1, 1)

    loss, acc_05, auc_roc_keras, auprc_keras = model.evaluate(X_test, y_test_eval, verbose=0)
    print(f"Final TEST loss:     {loss:.4f}")
    print(f"Final TEST accuracy @0.5 (keras): {acc_05:.4f}")
    print(f"Final TEST ROC-AUC (keras): {auc_roc_keras:.4f}")
    print(f"Final TEST PR-AUC  (keras): {auprc_keras:.4f}\n")

    # probability prediction
    y_scores_all = []
    y_pred_all = []
    y_true_all = []

    smooth_win = 7       
    min_run = 3          

    for pid in TEST_IDS:
        Xp, yp = load_person(pid)
        Xp = Xp.astype("float32")
        yp = yp.astype(int)

        ps = model.predict(Xp, batch_size=32, verbose=0).ravel()
        ps_smooth = smooth_probs(ps, win=smooth_win)

        yhat = (ps_smooth >= best_thr).astype(int)
        yhat = remove_short_positive_runs(yhat, min_len=min_run)

        y_scores_all.append(ps_smooth)
        y_pred_all.append(yhat)
        y_true_all.append(yp)

    y_scores = np.concatenate(y_scores_all)
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    # metrics at best_thr
    acc_thr = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"Postprocess: smooth_win={smooth_win}, min_run={min_run}")
    print(f"Accuracy @thr={best_thr:.2f}: {acc_thr:.4f}")
    print(f"Cohen's Kappa @thr={best_thr:.2f}: {kappa:.4f}\n")


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pretty_confusion_matrix(cm)

    print("Detailed classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Apnea"],
            digits=3,
            zero_division=0,
        )
    )

    # probability-based metrics (threshold-free)
    roc = roc_auc_score(y_true, y_scores)
    ap  = average_precision_score(y_true, y_scores)
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {ap:.4f}")


if __name__ == "__main__":
    main()
