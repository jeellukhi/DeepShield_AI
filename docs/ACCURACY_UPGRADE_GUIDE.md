# DeepShield Accuracy Upgrade Guide

This guide explains what was upgraded and how to retrain for better real-world accuracy.

## What was upgraded

1. Leakage-safe image training split:
- Dataset is split into train/test first.
- Augmentation is applied only on training data.
- This prevents duplicate-like samples from leaking into test metrics.

2. Stronger image feature extraction (`v3_multicue_stack`):
- Texture maps (grayscale + edges + Laplacian).
- Chroma/saturation cues (YCrCb + HSV).
- Frequency/artifact summary (DCT ratios, blockiness, blur/noise).

3. Better training/evaluation outputs:
- Added `balanced_accuracy` and `roc_auc`.
- Added confusion matrix in training output.
- Threshold parsing now supports both decimal and percent format.

4. Better robustness in image/video inference:
- Image fusion now adds small uncertainty boost for suspicious disagreement.
- Video probability now uses consistent threshold semantics.

5. Dataset quality audit endpoint:
- New admin endpoint: `GET /api/v1/dataset/quality`.
- Checks class balance, corrupt files, low resolution, blur, duplicate candidates.

## Admin workflow (recommended)

1. Login as admin.
2. Open `Admin` page.
3. Click `Check Dataset Quality`.
4. Review recommendations and clean dataset if needed.
5. Set training sizes:
- `Image Max/Class`: start with `4000` to `8000`.
- `Profile Max Samples`: start with `8000` to `15000` (if enough labeled feedback exists).
6. Click `Train Image Model`.
7. Click `Train Profile Model`.
8. Click `Load Model Evaluation`.
9. If fake samples still pass too often:
- Lower `Image Fake Threshold` by 2-5 points and re-check.
- Lower `Video Fake Threshold` by 2-5 points and re-check.

## Dataset quality rules (industry-style baseline)

- Keep class balance near 1:1 (real vs fake).
- Remove corrupt and unreadable files.
- Remove duplicates/near-duplicates.
- Avoid very low-resolution files (min side below 128 px).
- Keep train data diverse (different faces, lighting, compression, platforms).
- Use strict labels (wrong labels hurt more than small dataset size).

## Suggested external datasets (for future upgrade)

- FaceForensics++
- DFDC preview/full subsets
- Celeb-DF
- DeeperForensics-1.0

Use clear train/validation/test separation and never mix same source clip across splits.
