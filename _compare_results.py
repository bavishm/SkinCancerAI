import numpy as np, os

dirs = [
    ('5_convnext_focal_notta/raw', 'ConvNeXt run1 (no TTA)'),
    ('5_convnext_focal_tta/raw', 'ConvNeXt run1 (TTA)'),
    ('5_convnext_focal_film_notta/raw', 'ConvNeXt FiLM (no TTA)'),
    ('5_swinv2_old/run_1', 'SwinV2 old run1'),
    ('10_model_dual_ensemble_focal_tta/raw', 'Dual Ensemble (TTA)'),
    ('10_model_dual_ensemble_focal_notta_old_swin/raw', 'Dual Ensemble old swin (no TTA)'),
    ('5_convnext_focal_notta/threshold_0.2', 'ConvNeXt run1 + threshold'),
    ('5_convnext_focal_tta/threshold_0.2', 'ConvNeXt run1 TTA + threshold'),
    ('5_convnext_focal_film_notta/threshold_0.2', 'ConvNeXt FiLM + threshold'),
    ('10_model_dual_ensemble_focal_tta/threshold_0.2', 'Dual Ensemble TTA + threshold'),
    ('10_model_dual_ensemble_focal_notta_old_swin/threshold_0.2', 'Dual Ens old swin + threshold'),
]

header = f"{'Config':<45} {'Acc':>7} {'F1_Mac':>7} {'F1_Wtd':>7}"
print(header)
print('-' * 70)
for subdir, label in dirs:
    p = os.path.join('eval_data', subdir, 'test_results.npy')
    if os.path.exists(p):
        r = np.load(p, allow_pickle=True).item()
        m = r['metrics']
        print(f"{label:<45} {m['accuracy']:>7.4f} {m['f1_macro']:>7.4f} {m['f1_weighted']:>7.4f}")
    else:
        print(f"{label:<45} NOT FOUND")
