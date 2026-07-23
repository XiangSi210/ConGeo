# ConGeo 复现记录

> 论文：ConGeo: Robust Cross-view Geo-localization across Ground View Variations (ECCV 2024)
> 代码库：[github.com/XiangSi210/ConGeo](https://github.com/XiangSi210/ConGeo)
> 硬件：RTX 4060 Laptop GPU (8GB VRAM)，Windows 11

---

## 1. 实验环境

| 项目 | 版本/配置 |
|------|----------|
| Python | 3.9 |
| PyTorch | 2.x |
| GPU | RTX 4060 Laptop 8GB VRAM |
| 系统 | Windows 11 |

## 2. 数据集

**CVUSA_subset**（CVPR 子集）。

| 集合 | 文件 | 数量 |
|------|------|------|
| 训练 | train-19zl.csv | 8,884 对 |
| 验证 | val-19zl.csv | 8,884 对 |

每对数据：航拍图 (`bingmap`) + 街景全景拼接 (`streetview/panos`) + 标注 PNG (`streetview/annotations`)。

> ⚠️ CVUSA_subset 的 split CSV 仅包含文件路径，**不含 GPS 坐标**。ConGeo 原论文的 `custom_sampling` 依赖 GPS 字典 (`gps_dict.pkl`) 做 hard negative mining，本复现无法生成该文件，全程使用**随机负采样**。

## 3. 训练配置

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | convnext_base.fb_in22k_ft_in1k_384 | 与论文一致 |
| 输入尺寸 | 384×384 (卫星) / 140×768 (街景) | |
| epochs | 60 | |
| batch_size | 16 | 原论文多卡，本机 8GB 显存适配 |
| 优化器 | AdamW, lr=1e-4 | |
| 学习率策略 | Cosine Annealing + 1 epoch warmup | |
| **train_fov** | 180 | 每张街景随机裁 70°–180° 并随机旋转朝向 |
| **eval_fov** | 90 | 验证时裁切至 90° 窄视野，随机偏移 |
| custom_sampling | False | 无 GPS 字典，退化随机负采样 |
| AMP | torch.amp('cuda') | 新版 PyTorch API |
| grad_checkpointing | True | 节省 30-40% 显存 |

## 4. 复现结果

### 主要结果（eval FOV=90°，随机朝向）

| 指标 | 本复现 (epoch 52) | 论文 FoV=90° | 论文 FoV=70° |
|------|:--:|:--:|:--:|
| **R@1** | 37.13% | 55.9% | 37.1% |
| **R@5** | 62.53% | 73.2% | 55.7% |
| **R@10** | 71.29% | 79.0% | 62.8% |
| **R@top1** | **92.08%** | 90.9% | 81.4% |

### 解读

R@1 未达论文 90° 的 55.9%，核心原因：**无 GPS 字典 → custom_sampling 关闭 → hard negative mining 缺失**。随机负采样下，绝大多数负样本与查询点相距数千公里，模型区分难度大幅降低。

积极信号：

- **R@top1 反超论文 90°**（92.08% vs 90.9%）：特征方向学对了，粗筛能力优秀。
- **R@5 远超论文 70°**：宽范围训练 (train_fov=180) 带来更强的泛化。
- 若能补充 GPS 坐标 + hard negative mining，预计 R@1 可逼近论文 90° 档。

### 训练曲线

| Epoch | R@1 | Epoch | R@1 |
|:--:|:--:|:--:|:--:|
| 4 | 11.66% | 32 | 32.00% |
| 8 | 18.83% | 40 | 35.07% |
| 16 | 23.94% | **52** | **37.13%** ← 最佳 |
| 24 | 28.39% | 60 | 36.80% |

Epoch 52 后轻度过拟合。最佳权重保存为 `weights_e52_0.3713.pth`。

## 5. 硬件适配与性能调优

### `train_cvusa_ours.py`

| 改动 | 改前 | 改后 | 原因 |
|------|------|------|------|
| grad_checkpointing | False | True | 时间换空间，省 30-40% 显存 |
| batch_size | 8 | 16 | 显存有余量，加速收敛 |
| batch_size_eval | 8 | 16 | 同步提 |
| num_workers | 0 (Windows) | 4 | 打破单进程 I/O 瓶颈，CPU 利用率 50% → 100% |
| persistent_workers | — | True | worker 跨 epoch 复用，免销毁重建 |
| prefetch_factor | — | 2 | 每 worker 预取 2 batch，GPU 不等 CPU |
| AMP import | torch.cuda.amp | torch.amp | 新版 PyTorch 统一 API |

### `trainer.py`

| 改动 | 改前 | 改后 | 原因 |
|------|------|------|------|
| AMP autocast 上下文 | torch.cuda.amp.autocast() | torch.amp.autocast('cuda') | 适配新版 API |
| Loss 计算位置 | autocast 内 (FP16) | autocast 外 (FP32) | 对比 loss 依赖向量归一化 + pairwise 距离，FP16 精度不够；模型前传用 FP16，相似度矩阵转 FP32 再算 loss——速度与精度兼顾 |

### `transforms.py`

- 修复新版 Albumentations 中 `ImageCompression`、`ColorJitter`、`CoarseDropout` 的废弃参数警告

### `utils.py`

- 适配新版 PyTorch 中 `torch.load` 的参数变更

### 效果

| 指标 | 改前 | 改后 |
|------|------|------|
| CPU 利用率 | ~50% | ~100% |
| 单 epoch 时间 | ~1.8h | ~1.2h |
| 60 epoch 总耗时 | — | ~2.5 天 |

---

## 致谢

ConGeo 原作者（Mi et al., ECCV 2024）。

Sample4Geo（Deuser et al., ICCV 2023）提供代码基础。

引用 CVUSA_subset 数据集，来源：Workman S, Souvenir R, Jacobs N. 2015. Wide-Area Image Geolocalization with Aerial Reference Imagery. In: *IEEE International Conference on Computer Vision (ICCV)*. 1–9. DOI: 10.1109/ICCV.2015.451.
