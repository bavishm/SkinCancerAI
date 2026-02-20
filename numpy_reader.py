import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# load
DIR = './eval_data/5_swinv2/run_1/test_results.npy'
data = np.load(DIR, allow_pickle=True).item()

predictions   = data['predictions']
true_labels   = data['true_labels']
probabilities = data['probabilities']
class_names   = data['class_names']
metrics       = data['metrics']
model_name    = data['model_name']

# Short class names for readability
short_names = [
    'Nevi', 'Melanoma', 'BKL', 'BCC', 'AKIEC', 'Vasc.', 'Derm.'
]

# layout
fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'Skin Cancer AI — {model_name}\n'
             f'Accuracy: {metrics["accuracy"]:.2%}  |  '
             f'F1 Macro: {metrics["f1_macro"]:.4f}  |  '
             f'F1 Weighted: {metrics["f1_weighted"]:.4f}',
             fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# conf mat
ax1 = fig.add_subplot(gs[0, :2])
cm = confusion_matrix(true_labels, predictions)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.5, ax=ax1, cbar_kws={'label': 'Proportion'})
ax1.set_title('Confusion Matrix (counts, colour = row-normalised)', fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')
ax1.tick_params(axis='x', rotation=30)

# accuracy per class
ax2 = fig.add_subplot(gs[0, 2])
per_class_acc = cm_norm.diagonal()
colors = ['#e74c3c' if v < 0.7 else '#f39c12' if v < 0.85 else '#2ecc71'
          for v in per_class_acc]
bars = ax2.barh(short_names, per_class_acc, color=colors)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel('Accuracy')
ax2.set_title('Per-class Accuracy', fontweight='bold')
ax2.axvline(metrics['accuracy'], color='steelblue', linestyle='--',
            linewidth=1.5, label=f'Overall ({metrics["accuracy"]:.2%})')
ax2.legend(fontsize=8)
for bar, val in zip(bars, per_class_acc):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{val:.1%}', va='center', fontsize=8)

# clss dist
ax3 = fig.add_subplot(gs[1, 0])
true_counts = np.bincount(true_labels, minlength=len(class_names))
pred_counts = np.bincount(predictions, minlength=len(class_names))
x = np.arange(len(short_names))
w = 0.35
ax3.bar(x - w/2, true_counts, w, label='True', color='steelblue', alpha=0.8)
ax3.bar(x + w/2, pred_counts, w, label='Predicted', color='coral', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, rotation=30, ha='right')
ax3.set_ylabel('Sample count')
ax3.set_title('Class Distribution', fontweight='bold')
ax3.legend()

# prob per clss
ax4 = fig.add_subplot(gs[1, 1])
mean_probs = probabilities.mean(axis=0)
ax4.bar(short_names, mean_probs, color='mediumpurple', alpha=0.85)
ax4.set_ylabel('Mean predicted probability')
ax4.set_title('Mean Softmax Probability per Class', fontweight='bold')
ax4.tick_params(axis='x', rotation=30)
for i, v in enumerate(mean_probs):
    ax4.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)

# confidence hist
ax5 = fig.add_subplot(gs[1, 2])
confidence = probabilities.max(axis=1)
correct_mask = predictions == true_labels
ax5.hist(confidence[correct_mask],  bins=30, alpha=0.7, color='#2ecc71', label='Correct')
ax5.hist(confidence[~correct_mask], bins=30, alpha=0.7, color='#e74c3c', label='Wrong')
ax5.set_xlabel('Model confidence (max softmax)')
ax5.set_ylabel('Sample count')
ax5.set_title('Confidence Distribution', fontweight='bold')
ax5.legend()

save_path = os.path.join(DIR.replace('/test_results.npy',''), 'results_dashboard.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved → {save_path}")
plt.show()