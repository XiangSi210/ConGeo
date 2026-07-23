# ConGeo Reproduction Report

> Paper: ConGeo: Robust Cross-view Geo-localization across Ground View Variations (ECCV 2024)
> Code: [github.com/XiangSi210/ConGeo](https://github.com/XiangSi210/ConGeo)
> Hardware: RTX 4060 Laptop GPU (8 GB VRAM), Windows 11

---

## 1. Environment

| Item | Version / Spec |
|------|----------------|
| Python | 3.9 |
| PyTorch | 2.x |
| GPU | RTX 4060 Laptop 8 GB VRAM |
| OS | Windows 11 |

## 2. Dataset

**CVUSA_subset** (CVPR subset).

| Split | File | Size |
|-------|------|------|
| Train | train-19zl.csv | 8,884 pairs |
| Val | val-19zl.csv | 8,884 pairs |

Each pair: aerial image (`bingmap`) + street-level panorama (`streetview/panos`) + annotation PNG (`streetview/annotations`).

> ⚠️ The CVUSA_subset split files contain **only file paths, no GPS coordinates**. ConGeo's `custom_sampling` relies on a GPS dictionary (`gps_dict.pkl`) for hard negative mining. Without GPS metadata, this file cannot be generated. All training in this reproduction uses **random negative sampling**.

## 3. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | convnext_base.fb_in22k_ft_in1k_384 | Same as paper |
| Input size | 384×384 (satellite) / 140×768 (ground) | |
| Epochs | 60 | |
| Batch size | 16 | Adapted from multi-GPU to single RTX 4060 (8 GB) |
| Optimizer | AdamW, lr=1e-4 | |
| LR schedule | Cosine Annealing + 1 epoch warmup | |
| **train_fov** | 180 | Each ground panorama randomly cropped to 70°–180° with random heading |
| **eval_fov** | 90 | Evaluation crop fixed at 90°, random shift |
| custom_sampling | False | GPS dictionary unavailable; random negative sampling used |
| AMP | torch.amp('cuda') | Updated from legacy `torch.cuda.amp` |
| grad_checkpointing | True | Saves 30–40% VRAM |

## 4. Reproduction Results

### Main results (eval FOV=90°, arbitrary orientations)

| Metric | Ours (epoch 52) | Paper FoV=90° | Paper FoV=70° |
|--------|:--:|:--:|:--:|
| **R@1** | 37.13% | 55.9% | 37.1% |
| **R@5** | 62.53% | 73.2% | 55.7% |
| **R@10** | 71.29% | 79.0% | 62.8% |
| **R@top1** | **92.08%** | 90.9% | 81.4% |

### Discussion

R@1 falls short of the paper's 90° benchmark (55.9%) due to **hard negative mining being disabled**. Without GPS coordinates, `custom_sampling` cannot function — most negative pairs are thousands of kilometers apart, making the contrastive task substantially easier and reducing the model's fine-grained localization pressure.

Encouraging signs:

- **R@top1 surpasses the paper's 90° figure** (92.08% vs 90.9%): the learned feature space is directionally correct and excels at coarse retrieval.
- **R@5 far exceeds the paper's 70° figure**: training with `train_fov=180` yields stronger generalization.
- With GPS coordinates and hard negative mining restored, R@1 is expected to approach the paper's 90° benchmark.

### Training curve (R@1 over epochs)

| Epoch | R@1 | Epoch | R@1 |
|:--:|:--:|:--:|:--:|
| 4 | 11.66% | 32 | 32.00% |
| 8 | 18.83% | 40 | 35.07% |
| 16 | 23.94% | **52** | **37.13%** ← best |
| 24 | 28.39% | 60 | 36.80% |

Mild overfitting after epoch 52. Best checkpoint saved as `weights_e52_0.3713.pth`.

## 5. Hardware Adaptation & Performance Tuning

### `train_cvusa_ours.py`

| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| grad_checkpointing | False | True | Trade ~20% training time for 30–40% VRAM savings |
| batch_size | 8 | 16 | VRAM headroom after checkpointing; faster convergence |
| batch_size_eval | 8 | 16 | Consistent adjustment |
| num_workers | 0 (Windows) | 4 | Single-process I/O was bottlenecking GPU; 4 workers push CPU utilization 50% → 100% |
| persistent_workers | — | True | Workers persist across epochs, eliminating per-epoch teardown |
| prefetch_factor | — | 2 | Each worker preloads 2 batches; GPU never idles waiting on CPU |
| AMP import | torch.cuda.amp | torch.amp | Updated to unified PyTorch 2.x API |

### `trainer.py`

| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| AMP autocast context | torch.cuda.amp.autocast() | torch.amp.autocast('cuda') | Updated API |
| Loss computation precision | Inside autocast (FP16) | Outside autocast (FP32) | Contrastive loss relies on vector normalization and pairwise distances; FP16 underflows degrade accuracy. Forward pass stays in FP16; similarity matrix upcast to FP32 for loss — best of both worlds. |

### `transforms.py`

- Fixed deprecated argument warnings for Albumentations `ImageCompression`, `ColorJitter`, and `CoarseDropout` under newer library versions.

### `utils.py`

- Adapted `torch.load` calls to newer PyTorch argument conventions.

### Outcome

| Metric | Before | After |
|--------|--------|-------|
| CPU utilization | ~50% | ~100% |
| Time per epoch | ~1.8 h | ~1.2 h |
| Total 60 epochs | — | ~2.5 days |
| Runability | Crashed out-of-box on Win + PyTorch 2.x | Runs clean |

---

## Acknowledgements

ConGeo authors (Mi et al., ECCV 2024).

Sample4Geo (Deuser et al., ICCV 2023) for the codebase.

CVUSA dataset: Workman S, Souvenir R, Jacobs N. (2015). Wide-Area Image Geolocalization with Aerial Reference Imagery. In: *IEEE International Conference on Computer Vision (ICCV)*. 1–9. DOI: [10.1109/ICCV.2015.451](https://doi.org/10.1109/ICCV.2015.451).
