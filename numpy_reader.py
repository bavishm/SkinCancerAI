import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==========================================
# 1. LOAD DATA (Raw and Threshold)
# ==========================================
RAW_DIR = './eval_data/5_convnext_full_flim_focal_tta/raw/test_results.npy'
THRESH_DIR = './eval_data/5_convnext_full_flim_focal_tta/threshold_0.2/test_results.npy' 

raw_data = np.load(RAW_DIR, allow_pickle=True).item()
thresh_data = np.load(THRESH_DIR, allow_pickle=True).item()

# Raw data extraction
raw_preds = raw_data['predictions']
true_labels = raw_data['true_labels']
metrics = raw_data['metrics']
class_names = raw_data['class_names']

if "dual" in RAW_DIR:
    model_name = "Dual Ensemble (Swin + ConvNeXt)"
else:
    model_name = raw_data['model_name']

# Threshold data extraction
thresh_preds = thresh_data['predictions']

# Short class names for readability
short_names = ['Nevi', 'Melanoma', 'BKL', 'BCC', 'AKIEC', 'Vasc.', 'Derm.']

# ==========================================
# 2. CALCULATE ACCURACIES
# ==========================================
cm_raw = confusion_matrix(true_labels, raw_preds)
cm_raw_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
acc_raw = cm_raw_norm.diagonal()

cm_thresh = confusion_matrix(true_labels, thresh_preds)
cm_thresh_norm = cm_thresh.astype(float) / cm_thresh.sum(axis=1, keepdims=True)
acc_thresh = cm_thresh_norm.diagonal()

diffs = acc_thresh - acc_raw

# ==========================================
# 3. SET UP FIGURE & LAYOUT
# ==========================================
# UPDATED: Changed width_ratios to give the bar chart more room
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1, 1.6]})

# UPDATED: Pushed the plots closer to the edges of the image to remove dead space
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

fig.suptitle(f'Skin Cancer AI — {model_name}\n'
             f'Accuracy: {metrics["accuracy"]:.2%}  |  '
             f'F1 Macro: {metrics["f1_macro"]:.4f}  |  '
             f'F1 Weighted: {metrics["f1_weighted"]:.4f}',
             fontsize=14, fontweight='bold', y=0.98)

# ==========================================
# 4. PLOT 1: RAW CONFUSION MATRIX
# ==========================================
# UPDATED: Added 'shrink': 0.72 to make the gradient bar shorter
sns.heatmap(cm_raw_norm, annot=cm_raw, fmt='d', cmap='Greys',
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.5, ax=ax1, square=True,
            cbar_kws={'label': 'Proportion', 'shrink': 0.72})

ax1.set_title('Confusion Matrix (counts, colour = row-normalised)', fontweight='bold', pad=15)
ax1.set_ylabel('True Label', fontweight='bold')
ax1.set_xlabel('Predicted Label', fontweight='bold')
ax1.tick_params(axis='x', rotation=30)

# ==========================================
# 5. PLOT 2: RAW VS THRESHOLD BAR CHART
# ==========================================
y_pos = np.arange(len(short_names))
bar_height = 0.55

color_base = '#808080'  # Standard grey
color_gain = '#2ecc71'  # Green for improvement
color_loss = '#e74c3c'  # Red for reduction

# Plot base raw accuracy (Grey bars)
ax2.barh(y_pos, acc_raw, height=bar_height, color=color_base, label='Raw (standard argmax)')

for i in range(len(short_names)):
    raw_val = acc_raw[i]
    thresh_val = acc_thresh[i]
    diff = diffs[i]
    
    text_str = f"{thresh_val:.1%} ({diff*100:+.1f} pp)"
    
    if diff > 0:
        ax2.barh(y_pos[i], diff, left=raw_val, height=bar_height, color=color_gain)
        ax2.text(thresh_val + 0.015, y_pos[i], text_str, va='center', ha='left', fontsize=10)
    elif diff < 0:
        ax2.barh(y_pos[i], -diff, left=thresh_val, height=bar_height, color=color_loss)
        ax2.text(thresh_val - 0.015, y_pos[i], text_str, va='center', ha='right', fontsize=10)
    else:
        ax2.text(raw_val + 0.015, y_pos[i], text_str, va='center', ha='left', fontsize=10)

import matplotlib.patches as mpatches
patch_gain = mpatches.Patch(color=color_gain, label='Improved after threshold')
patch_loss = mpatches.Patch(color=color_loss, label='Reduced after threshold')
handles, labels = ax2.get_legend_handles_labels()
handles.extend([patch_gain, patch_loss])

mean_raw_acc = np.mean(acc_raw)
line_mean = ax2.axvline(mean_raw_acc, color='#555555', linestyle='--', linewidth=1.5, label='Raw mean')
handles.insert(0, line_mean) 

ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_names, fontsize=11)
ax2.set_xlim(0, 1.0)
ax2.set_xlabel('Accuracy', fontsize=11)
ax2.set_title('Per-class Accuracy: Raw vs Threshold 0.2 (Dual Ensemble)', fontweight='bold', pad=15)

# Moved legend to the bottom right corner so it never covers the bars
ax2.legend(handles=handles, loc='lower right', fontsize=9, framealpha=1.0)
ax2.grid(axis='x', linestyle='-', alpha=0.3)

ax2.invert_yaxis() 

# ==========================================
# 6. SAVE AND SHOW
# ==========================================
save_path = os.path.join(RAW_DIR.replace('/test_results.npy',''), 'results_combined_dashboard.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved → {save_path}")
plt.show()