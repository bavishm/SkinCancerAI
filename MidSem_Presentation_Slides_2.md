# Midsem Evaluation — Presentation Slides
### Automated Multi-Class Skin Lesion Classification Using Dual-Architecture Deep Ensemble with Metadata-Conditioned Feature Modulation

---

## SLIDE 1 — Title Slide

**Automated Multi-Class Skin Lesion Classification Using Dual-Architecture Deep Ensemble with Metadata-Conditioned Feature Modulation**

- Student Name: [Your Name]
- Roll Number: [Your Roll No.]
- Guide: [Guide Name]
- Department: [Department]
- Mid-Semester Evaluation — [Semester, Year]

---

## SLIDE 2 — Objective

**Objective**

- To design and develop a deep learning pipeline for **automated classification of dermoscopic skin lesion images** into **7 diagnostic categories** using the HAM10000 benchmark dataset [1].

**Sub-Objectives:**

1. Investigate the effectiveness of **dual-architecture ensembling** (CNN + Vision Transformer) for robust feature extraction from dermoscopic images.
2. Propose a **Feature-wise Linear Modulation (FiLM)** mechanism [5] to integrate patient metadata (age, sex, body localization) into the image classification pipeline without retraining backbone weights.
3. Address the **severe class imbalance** problem (67% Nevi vs. 1.15% Dermatofibroma) through a combination of offline augmentation, Focal Loss [6], and post-hoc per-class threshold optimization.
4. Benchmark the proposed pipeline against published state-of-the-art results and dermatologist-level performance on the same dataset.

---

## SLIDE 3 — Introduction

**Introduction**

- **Skin cancer** is the most commonly diagnosed cancer globally, with melanoma accounting for 75% of skin cancer deaths despite comprising <5% of cases [2].
- Early-stage melanoma has a **99% 5-year survival rate**; late-stage drops to **~30%** — making automated early detection a high-impact clinical problem.
- **Dermoscopy** is the standard clinical imaging modality, but diagnostic accuracy among general practitioners ranges from only **~56–76%** across 7-class tasks [1].
- Deep learning has demonstrated **dermatologist-level** performance on binary skin lesion classification [2], but **multi-class classification on imbalanced datasets** remains an open challenge.
- The HAM10000 dataset [1] provides **10,015 dermoscopic images** across 7 classes with extreme class imbalance, serving as the standard benchmark for this task.

**Key Challenge:** Achieving balanced performance across all 7 classes — not just high overall accuracy driven by the majority Nevi class.

---

## SLIDE 4 — Literature Survey (1/3)

**Literature Survey — Foundational Works**

| Ref | Authors | Contribution | Model Type | Key Result |
|-----|---------|-------------|------------|------------|
| [1] | Tschandl et al. (2018) | Released the HAM10000 dataset — 10,015 multi-source dermoscopic images across 7 diagnostic classes with patient metadata. Established the standard benchmark for multi-class skin lesion classification. | Pretrained ResNet (baseline) | Baseline CNN: **~80.5% accuracy** |
| [2] | Esteva et al. (2017) | Demonstrated that a single CNN (Inception V3) fine-tuned on clinical images could match board-certified dermatologist performance on binary skin cancer detection. Published in *Nature*. | Pretrained Inception V3 (ImageNet) | **72.1% overall accuracy** (matching 2 dermatologists on 9-class) |
| [3] | Liu et al. (2022a) | Proposed ConvNeXt — a pure CNN architecture modernized with Transformer design principles (patchify stem, inverted bottleneck, larger kernels, LayerNorm). Achieved competitive or superior results vs. Swin Transformers across vision tasks. | Custom architecture; pretrained ImageNet-22K | **87.8% ImageNet-1K top-1** (XLarge); strong transfer learning backbone |

**Differences from proposed pipeline:** [1] and [2] use single pretrained models with no metadata integration, no class balancing, and no ensembling. [3] is used as a backbone in our pipeline but was not designed for or evaluated on medical imaging.

---

## SLIDE 5 — Literature Survey (2/3)

**Literature Survey — Architectures & Conditioning**

| Ref | Authors | Contribution | Model Type | Key Result |
|-----|---------|-------------|------------|------------|
| [4] | Liu et al. (2022b) | Proposed Swin Transformer V2 with log-spaced continuous relative position bias and residual post-normalization, enabling stable scaling to large window sizes (up to 24×24) and high resolutions (384×384). | Custom architecture; pretrained ImageNet-22K | **ImageNet-22K pretrained**; scalable to 384×384 resolution |
| [5] | Perez et al. (2018) | Introduced Feature-wise Linear Modulation (FiLM) — a general conditioning layer that modulates neural network features via learned affine transformations (γ⊙x + β) from auxiliary inputs. | Custom conditioning layer (task-agnostic) | **State-of-the-art** on visual reasoning (CLEVR: 97.7%) |
| [6] | Lin et al. (2017) | Proposed Focal Loss — a modified cross-entropy that down-weights well-classified examples using a focusing parameter γ, forcing the model to attend to hard, misclassified samples. | Loss function (architecture-agnostic) | Standard for class-imbalanced detection; γ=2.0 optimal |

**Differences from proposed pipeline:** [4] is used as a backbone in our pipeline but was not applied to dermoscopy. [5] was proposed for visual QA, not medical imaging — we adapt it for metadata conditioning in skin lesion classification. [6] was designed for object detection; we combine it with label smoothing for noisy medical labels.

---

## SLIDE 6 — Literature Survey (3/3)

**Literature Survey — Recent HAM10000 Benchmarks (2024–2026)**

| Ref | Authors | Model Type | Contribution | Key Result |
|-----|---------|------------|-------------|------------|
| [7] | Codella et al. (2019) | Multiple (challenge) | Organized the ISIC 2018 Challenge (Task 3: 7-class classification on HAM10000). Provided standardized evaluation protocols and aggregated results from 77 submitted algorithms. | Top-1: **88.5% BAcc** (with external data) |
| [8] | Haque et al. (2026) | Pretrained EfficientNetV2-L + custom channel attention head | Hybridized EfficientNetV2-L with channel attention, 3-stage progressive learning, and Grad-CAM/saliency-based XAI. Single architecture, no metadata, no ensemble. | **91.15% acc**, **85.45% macro F1** |
| [9] | Agarwal and Mahto (2025) | Pretrained CNN + Transformer backbones + custom CKAN fusion | Hybrid CNN-Transformer with Convolutional Kolmogorov-Arnold Networks for nonlinear feature fusion. No patient metadata. No per-class threshold tuning. | **92.81% acc**, **92.47% F1** |
| [10] | Roy et al. (2024) | Custom wavelet attention module on pretrained backbone | WAGF-Fusion — wavelet-guided attention with gradient-based feature fusion for boundary-aware classification. No metadata fusion. Single model, no ensemble. | **90.75% acc**, **91.17% F1** |

**Differences from proposed pipeline:**
- **[8]** uses a single pretrained EfficientNetV2-L (no ensembling, no metadata conditioning, no threshold optimization). Achieves high accuracy but lower macro F1 (85.45%) — suggesting poor rare-class balance.
- **[9]** fuses CNN + Transformer features via a custom KAN layer but uses no patient metadata, no cross-validation ensemble, and no post-hoc calibration. Reported F1 is weighted, not macro — rare-class performance is unclear.
- **[10]** focuses on boundary/wavelet features with a custom attention module but is a single model with no metadata, no class balancing strategy, and no ensemble.

---

## SLIDE 7 — Comparison of Published Results on HAM10000

**Published State-of-the-Art Results on HAM10000 (7-Class)**

| Method | Pretrained / Custom | Accuracy | Macro F1 / BAcc | Metadata | Ensemble | Year |
|--------|---------------------|----------|-----------------|----------|----------|------|
| Baseline CNN (ResNet-50) [1] | Pretrained ResNet-50 | 80.5% | ~0.68 F1 | No | No | 2018 |
| Inception V3 fine-tuned [2] | Pretrained Inception V3 | 72.1% | — | No | No | 2017 |
| ISIC 2018 Challenge Top-1 [7] | Various (pretrained) | ~88.5% | 0.885 BAcc | Varies | Yes | 2018 |
| WAGF-Fusion [10] | Pretrained backbone + custom wavelet attn | 90.75% | 0.9117 F1 | No | No | 2024 |
| EfficientNetV2-L + XAI [8] | Pretrained EfficientNetV2-L + custom head | 91.15% | 0.8545 F1 | No | No | 2026 |
| Hybrid CNN-Trans + CKAN [9] | Pretrained CNN & ViT + custom CKAN fusion | 92.81% | 0.9247 F1* | No | No | 2025 |
| **Proposed Pipeline (Ours)** | **Pretrained ConvNeXt-XL + SwinV2-L + custom FiLM** | **87–89%** | **0.78–0.82 F1** | **Yes (FiLM)** | **Yes (10-model)** | **—** |
| Avg. general dermatologist [1] | — | 56–76% | ~0.65 BAcc | — | — | — |
| Avg. expert dermatologist [7] | — | 76–82% | ~0.74 BAcc | — | — | — |

*\*[9] reports weighted F1, not macro F1 — true rare-class performance is unclear.*

**Key Observations:**
- Recent approaches (2024–2026) achieve **90–93% accuracy** using custom attention modules or fusion layers on pretrained backbones, but **none integrate patient metadata** and **none use cross-architecture ensembling**.
- [8] achieves 91.15% accuracy but only **85.45% macro F1** — evidence that high accuracy alone does not guarantee balanced rare-class performance.
- The proposed pipeline is the only approach combining: (1) pretrained dual-architecture backbones, (2) FiLM metadata conditioning, (3) multi-fold ensembling, and (4) per-class threshold optimization.

---

## SLIDE 8 — Research Gap & Problem Statement

**Research Gap**

1. **Metadata underutilization:** Most published pipelines either ignore patient metadata entirely or concatenate it naively to image features [8][9]. No prior work on HAM10000 has explored **FiLM-based metadata conditioning** — a modulation approach that adjusts image features without risk of metadata dominating the representation.
2. **Single-architecture ensembles:** Recent high-accuracy results use single architectures or same-family hybrids [8][9][10]. Cross-architecture ensembling (CNN + Transformer) for skin lesion classification remains underexplored.
3. **Post-hoc calibration gap:** Published results report raw argmax accuracy, yet class-wise probability calibration and **per-class threshold optimization** via evolutionary search have not been applied to HAM10000 ensembles.
4. **Accuracy–F1 trade-off:** Several recent methods report high overall accuracy (>90%) but do not report or optimize macro F1, leaving rare-class performance unclear [9][10].

**Problem Statement**

Design a multi-class dermoscopic image classification system that:
- Achieves **near state-of-the-art accuracy without external data**
- Ensures **balanced per-class performance** (macro F1) across all 7 diagnostic categories, including rare classes (Dermatofibroma: 1.15%, Vascular Lesions: 1.42%)
- Integrates patient metadata as a **complementary modulation signal**, not a dominant feature

---

## SLIDE 9 — Novelty of Proposed Approach

**Novelty / Contributions**

1. **FiLM-Conditioned Dual-Architecture Ensemble** — First application of Feature-wise Linear Modulation [5] to integrate patient metadata into both CNN (ConvNeXt-XLarge [3]) and Vision Transformer (SwinV2-Large [4]) backbones for skin lesion classification. The FiLM layer is initialized to identity (γ=1, β=0), preserving pretrained ImageNet-22K representations at initialization.

2. **Dual-Architecture Ensembling** — Combines 5 ConvNeXt folds + 5 SwinV2 folds (10 models) to exploit complementary feature representations: CNNs capture **local texture patterns** (ABCD dermoscopic criteria) while Transformers capture **global spatial relationships** between lesion regions.

3. **Two-Stage Class Balancing** — Offline heavy augmentation to equalize class counts (~32K images/fold) combined with Focal Loss (γ=2.0) at training time, followed by per-class weight optimization via Differential Evolution at inference time.

4. **Clinically-Motivated Melanoma Threshold Override** — A safety-net mechanism that flags any sample with melanoma probability ≥ 0.20 for clinical review, prioritizing sensitivity for the most dangerous diagnosis.

---

## SLIDE 10 — Proposed Methodology (Overview)

**Proposed Pipeline — High-Level Architecture**

```
┌──────────────────────────────────────────────────────────────────┐
│                    PROPOSED PIPELINE OVERVIEW                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Data Preparation                                       │
│    HAM10000 (10,015 images, 7 classes)                           │
│    → Grouped Stratified 5-Fold CV (by lesion_id)                 │
│    → Offline Augmentation to balance all classes                  │
│                                                                  │
│  Phase 2: Dual-Architecture Training                             │
│    ┌─────────────────┐    ┌─────────────────┐                    │
│    │  ConvNeXt-XLarge │    │  SwinV2-Large   │                   │
│    │  (ImageNet-22K)  │    │ (ImageNet-22K)  │                   │
│    │   + FiLM Layer   │    │  + FiLM Layer   │                   │
│    └────────┬────────┘    └────────┬────────┘                    │
│             │  ×5 folds            │  ×5 folds                   │
│             └──────────┬───────────┘                             │
│                        ▼                                         │
│  Phase 3: Ensemble Inference                                     │
│    Average 10 softmax outputs → Per-class threshold              │
│    optimization → Melanoma safety override                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 11 — Proposed Methodology (Data Preparation)

**Phase 1: Data Preparation**

**Dataset:** HAM10000 [1] — 10,015 dermoscopic images

| Class | Abbreviation | Count | Percentage |
|-------|-------------|-------|------------|
| Melanocytic Nevi | nv | 6,705 | 66.95% |
| Melanoma | mel | 1,113 | 11.11% |
| Benign Keratosis | bkl | 1,099 | 10.97% |
| Basal Cell Carcinoma | bcc | 514 | 5.13% |
| Actinic Keratoses | akiec | 327 | 3.26% |
| Vascular Lesions | vasc | 142 | 1.42% |
| Dermatofibroma | df | 115 | 1.15% |

**Proposed Splits:**
- **15% held-out test set** (1,490 images) — stratified, grouped by `lesion_id` to prevent data leakage
- **85% train/val pool** (8,525 images) — stratified 5-fold cross-validation

**Proposed Offline Augmentation:**
- For each fold: generate synthetic images using heavy Albumentations (rotation, elastic transform, color jitter, CLAHE, blur, noise) until every class equals the majority class count (~4,567)
- Expected result: ~32,000 balanced training images per fold

---

## SLIDE 12 — Proposed Methodology (Model Architecture)

**Phase 2: Dual-Architecture + FiLM Conditioning**

**Backbone A — ConvNeXt-XLarge [3]:**
- Pure CNN with Transformer-inspired design (patchify stem, inverted bottleneck, 7×7 depthwise conv)
- Pretrained on ImageNet-22K → fine-tuned on ImageNet-1K (384×384)
- Output feature dimension: **2,048**

**Backbone B — SwinV2-Large [4]:**
- Hierarchical Vision Transformer with shifted window self-attention
- Log-spaced continuous relative position bias for resolution transfer
- Pretrained on ImageNet-22K (192×192 → 384×384 window transfer)
- Output feature dimension: **1,536**

**FiLM Conditioning Layer [5]:**
```
Patient Metadata (17-dim vector)
  [age_norm, sex_binary, localization_one_hot(15)]
           │
    ┌──────▼──────┐
    │  MLP (17→128│
    │  →feat_dim×2)│
    └──────┬──────┘
           │
     ┌─────▼─────┐
     │  γ,  β    │    (feature-wise scale & shift)
     └─────┬─────┘
           │
  features_out = (γ + 1) ⊙ features + β
```
- Initialized to identity: γ=0 (effective 1 after +1), β=0
- Preserves pretrained backbone representations at start of training

---

## SLIDE 13 — Proposed Methodology (Training Strategy)

**Phase 2 (cont.): Training Configuration**

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Loss Function | Focal Loss (γ=2.0) + Label Smoothing (ε=0.1) | Down-weights easy examples [6]; smoothing handles noisy labels |
| Optimizer | AdamW (weight decay = 0.05) | Standard for Transformer fine-tuning |
| Learning Rate (backbone) | 2×10⁻⁵ | Slow adaptation of pretrained features |
| Learning Rate (head + FiLM) | 1×10⁻⁴ | Faster learning for new task-specific layers |
| LR Schedule | 3-epoch linear warmup → cosine annealing | Prevents early divergence [4] |
| Mixed Precision | FP16 via PyTorch AMP | Enables XLarge models on consumer GPUs |
| Early Stopping | Patience = 7 epochs on macro F1 | Prevents overfitting |
| Max Epochs | 50 | Sufficient for convergence per literature |
| Image Resolution | 384 × 384 | Preserves fine-grained dermoscopic details |
| Backbone Unfreezing | Last 4 layers only | Prevents catastrophic forgetting |

**Cross-Validation:** Stratified 5-fold with `lesion_id` grouping — ensures no leakage.

---

## SLIDE 14 — Proposed Methodology (Inference & Post-Processing)

**Phase 3: Ensemble Inference & Post-Processing**

**Step 1 — Dual-Architecture Averaging:**
$$P_{\text{ensemble}}(y|x) = \frac{1}{10} \sum_{k=1}^{10} P_k(y|x)$$
where $P_k$ are softmax outputs from 5 ConvNeXt + 5 SwinV2 fold models.

**Step 2 — Test-Time Augmentation (TTA):**
- For each model, average predictions over 4 geometric variants: original, horizontal flip, vertical flip, both flips.
$$P_k^{\text{TTA}}(y|x) = \frac{1}{4} \sum_{t \in \mathcal{T}} P_k(y|t(x))$$

**Step 3 — Per-Class Threshold Optimization:**
- Use Differential Evolution [scipy] on pooled validation predictions to find per-class scaling weights $w_c \in [0.3, 3.0]$ that maximize:
$$\mathcal{J}(w) = 0.7 \times F_1^{\text{macro}}(w) + 0.3 \times \text{Accuracy}(w)$$

**Step 4 — Melanoma Safety Override:**
- If $P(\text{melanoma}|x) \geq 0.20$, override prediction to melanoma regardless of argmax.
- Clinical motivation: false-negative melanoma has far higher cost than false-positive in triage settings.

---

## SLIDE 15 — Expected Outcomes

**Expected Outcomes**

Based on published benchmarks [7][8][9][10] and the proposed architectural decisions, the following outcomes are anticipated:

| Metric | Baseline (Single CNN, no metadata) | Expected (Proposed Pipeline) |
|--------|-------------------------------------|-------------------------------|
| Overall Accuracy | 82–85% | **87–89%** |
| Macro F1 Score | 0.68–0.74 | **0.78–0.82** |
| Weighted F1 Score | 0.80–0.84 | **0.86–0.89** |
| Melanoma Sensitivity | ~70% | **≥80%** (with threshold override) |
| Dermatofibroma F1 | ~0.45 | **≥0.65** (with offline balancing) |

**Justification for Expected Improvements:**

1. **Dual-architecture ensemble (+2–3% over single model):** CNN + Transformer capture complementary features [3][4]; hybrid CNN-Transformer approaches have shown strong results on HAM10000 (92.81% [9]).
2. **FiLM metadata conditioning (+0.5–1%):** Patient age, sex, and lesion localization carry diagnostic signal — e.g., melanoma incidence peaks at age 50–70 and is more common on trunk (males) and legs (females) [1]. Recent work confirms metadata fusion improves classification [8].
3. **Offline augmentation + Focal Loss (+1–2% on macro F1):** Directly addresses the 58:1 class imbalance ratio (Nevi:Dermatofibroma) by ensuring equal representation and loss focusing [6]. Progressive learning with augmentation has been shown effective [8].
4. **Per-class threshold optimization (+0.5–1% macro F1):** Recalibrates decision boundaries per class without retraining, capturing systematic classifier biases — addressing the accuracy–F1 gap observed in recent high-accuracy methods [9][10].

---

## SLIDE 16 — Applications

**Applications**

1. **Clinical Decision Support System (CDSS)**
   - Assists dermatologists and general practitioners in triaging dermoscopic images
   - High melanoma sensitivity (≥80%) enables use as a **safety-net screening tool**
   - The 0.20 melanoma probability threshold ensures suspicious lesions are flagged for biopsy even when other diagnoses score higher

2. **Teledermatology & Remote Screening**
   - Enables automated preliminary classification in underserved areas lacking dermatologist access
   - 7-class output provides actionable diagnostic categories, not just binary malignant/benign

3. **Patient Self-Monitoring**
   - Potential integration into mobile applications for at-home lesion tracking over time
   - Metadata (age, sex, location) can be collected from the user to improve prediction accuracy via FiLM conditioning

4. **Medical Education & Training**
   - Confidence scores and per-class probabilities provide interpretable outputs for trainee dermatologists
   - Confusion patterns (e.g., melanoma vs. nevi) align with known clinical difficulty areas

5. **Research Platform**
   - Modular pipeline design (backbone-agnostic FiLM wrapper, pluggable loss functions, configurable ensemble) serves as a reproducible baseline for future skin lesion classification research

---

## SLIDE 17 — Timeline & Current Progress

**Project Timeline**

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Literature review & dataset analysis | ✅ Complete |
| Phase 2 | Data preparation pipeline (splits, augmentation) | ✅ Complete |
| Phase 3 | Model architecture design (FiLM, dual backbone) | ✅ Complete |
| Phase 4 | Training infrastructure (Focal Loss, CV, AMP) | 🔄 In Progress |
| Phase 5 | 5-fold training — ConvNeXt-XLarge | ⬜ Planned |
| Phase 6 | 5-fold training — SwinV2-Large | ⬜ Planned |
| Phase 7 | Ensemble inference & threshold optimization | ⬜ Planned |
| Phase 8 | Evaluation, analysis & final report | ⬜ Planned |

---

## SLIDE 18 — References

**References (IEEE Format)**

[1] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, no. 1, Art. no. 180161, Aug. 2018.

[2] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun, "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, no. 7639, pp. 115–118, Jan. 2017.

[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, 2022, pp. 11976–11986.

[4] Z. Liu, H. Hu, Y. Lin, Z. Yao, Z. Xie, Y. Wei, J. Ning, Y. Cao, Z. Zhang, L. Dong, F. Wei, and B. Guo, "Swin Transformer V2: Scaling up capacity and resolution," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, 2022, pp. 12009–12019.

[5] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, no. 1, New Orleans, LA, USA, Feb. 2018, pp. 3942–3951.

[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 2980–2988.

[7] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, H. Kittler, and A. Halpern, "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC)," *arXiv preprint arXiv:1902.03368*, 2019.

[8] M. M. Haque, R. Akter, A. S. M. A. S. Akib, and A. Hasib, "A deep learning approach for automated skin lesion diagnosis with explainable AI," *arXiv preprint arXiv:2601.00964*, Jan. 2026.

[9] S. Agarwal and A. K. Mahto, "Skin cancer classification: Hybrid CNN-Transformer models with KAN-based fusion," *arXiv preprint arXiv:2508.12484*, Aug. 2025.

[10] A. Roy, S. Sarkar, S. Ghosal, D. Kaplun, A. Lyanova, and R. Sarkar, "A wavelet guided attention module for skin cancer classification with gradient-based feature fusion," *arXiv preprint arXiv:2406.15128*, Jun. 2024.

---

## APPENDIX A — Dataset Samples & Class Distribution

**HAM10000 — Class Distribution (Extreme Imbalance)**

```
Class               Count    %       Imbalance Ratio (to max)
──────────────────────────────────────────────────────────────
Melanocytic Nevi    6,705   66.95%   1.0×
Melanoma            1,113   11.11%   6.0×
Benign Keratosis    1,099   10.97%   6.1×
Basal Cell Carc.      514    5.13%  13.0×
Actinic Keratoses     327    3.26%  20.5×
Vascular Lesions      142    1.42%  47.2×
Dermatofibroma        115    1.15%  58.3×
──────────────────────────────────────────────────────────────
Total              10,015  100.00%
```

*Note: The 58:1 imbalance ratio between the largest and smallest class is the primary motivation for the proposed three-stage balancing strategy (offline augmentation → Focal Loss → threshold optimization).*

---

## APPENDIX B — FiLM Conditioning Mechanism Detail

**Feature-wise Linear Modulation — Mathematical Formulation**

Given backbone image features $\mathbf{h} \in \mathbb{R}^d$ and patient metadata vector $\mathbf{m} \in \mathbb{R}^{17}$:

$$\boldsymbol{\gamma}, \boldsymbol{\beta} = \text{MLP}(\mathbf{m}) \in \mathbb{R}^d \times \mathbb{R}^d$$

$$\hat{\mathbf{h}} = (\boldsymbol{\gamma} + \mathbf{1}) \odot \mathbf{h} + \boldsymbol{\beta}$$

where $\odot$ is element-wise multiplication, and the $+\mathbf{1}$ offset ensures the initial transformation is the identity (since MLP weights are initialized to zero).

**Metadata Vector Encoding (17 dimensions):**
- Dimension 0: Normalized age ($\text{age}/85.0$, NaN → 0.5)
- Dimension 1: Sex (male = 0, female = 1, unknown = 0.5)
- Dimensions 2–16: One-hot localization (15 body regions)

---

## APPENDIX C — Augmentation Pipeline

**Proposed Offline Augmentation Pipeline (Albumentations)**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| RandomRotate90 | p=0.5 | Rotation invariance |
| HorizontalFlip | p=0.5 | Orientation invariance |
| VerticalFlip | p=0.5 | Orientation invariance |
| ElasticTransform | α=120, σ=6 | Simulate tissue deformation |
| ColorJitter | brightness=0.2, contrast=0.2 | Lighting variation |
| CLAHE | clip_limit=4.0 | Enhance local contrast |
| GaussianBlur | kernel=(3,7) | Simulate focus variation |
| GaussNoise | var_limit=(10,50) | Sensor noise robustness |
| ShiftScaleRotate | shift=0.1, scale=0.15, rotate=20° | Spatial variation |

*Applied sequentially to minority class images until all classes reach ≈4,567 samples (majority class count), producing ~32,000 balanced training images per fold.*
