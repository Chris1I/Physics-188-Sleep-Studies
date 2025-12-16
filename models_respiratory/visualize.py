import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os

sns.set_style("whitegrid")

def load_data():
    y_test = np.load('processed_data/y_test.npy')
    y_pred = np.load('predictions/predictions_optimized.npy')
    y_prob = np.load('predictions/probabilities.npy')
    test_subjects = np.load('processed_data/test_subjects.npy')
    return y_test, y_pred, y_prob, test_subjects

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_roc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
    auc_pr = auc(rec_curve, prec_curve)
    
    return {
        'cm': cm, 'recall': recall, 'precision': precision, 'f1': f1,
        'auc_roc': auc_roc, 'fpr': fpr, 'tpr': tpr,
        'auc_pr': auc_pr, 'prec_curve': prec_curve, 'rec_curve': rec_curve
    }

def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['No Apnea', 'Apnea'])
    ax.set_yticklabels(['No Apnea', 'Apnea'], rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(metrics, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2.5, label=f"AUC = {metrics['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(metrics, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(metrics['rec_curve'], metrics['prec_curve'], 'g-', linewidth=2.5, label=f"AUC = {metrics['auc_pr']:.3f}")
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_summary(metrics, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Recall', 'Precision', 'F1 Score', 'AUC-ROC', 'AUC-PR']
    values = [metrics['recall'], metrics['precision'], metrics['f1'], metrics['auc_roc'], metrics['auc_pr']]
    colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.5 else '#e74c3c' for v in values]
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_subject(y_test, y_pred, y_prob, subjects, save_path):
    unique_subjects = np.unique(subjects)
    subject_data = []
    
    for subj_id in unique_subjects:
        mask = subjects == subj_id
        subj_y_true = y_test[mask]
        subj_y_pred = y_pred[mask]
        n_true = np.sum(subj_y_true == 1)
        
        if n_true > 0:
            tp = np.sum((subj_y_true == 1) & (subj_y_pred == 1))
            recall = tp / n_true
            subject_data.append({'id': subj_id, 'n_apnea': n_true, 'recall': recall * 100})
    
    if not subject_data:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    subjects_list = [d['id'] for d in subject_data]
    recalls = [d['recall'] for d in subject_data]
    n_apnea = [d['n_apnea'] for d in subject_data]
    
    axes[0].bar(range(len(subjects_list)), recalls, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Subject', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Recall per Subject', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(subjects_list)))
    axes[0].set_xticklabels([str(s) for s in subjects_list])
    axes[0].axhline(50, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(range(len(subjects_list)), n_apnea, color='coral', alpha=0.8)
    axes[1].set_xlabel('Subject', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Apnea Events', fontsize=11, fontweight='bold')
    axes[1].set_title('Apnea Events per Subject', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(subjects_list)))
    axes[1].set_xticklabels([str(s) for s in subjects_list])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading results...")
    y_test, y_pred, y_prob, test_subjects = load_data()
    
    print("Computing metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    os.makedirs('results', exist_ok=True)
    
    print("Generating plots...")
    plot_confusion_matrix(metrics['cm'], 'results/confusion_matrix.png')
    plot_roc_curve(metrics, 'results/roc_curve.png')
    plot_precision_recall_curve(metrics, 'results/precision_recall_curve.png')
    plot_metrics_summary(metrics, 'results/metrics_summary.png')
    plot_per_subject(y_test, y_pred, y_prob, test_subjects, 'results/per_subject.png')
    
    print("\nResults:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  Recall: {metrics['recall']*100:.1f}%")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"\nAll plots saved to results/")

if __name__ == '__main__':
    main()