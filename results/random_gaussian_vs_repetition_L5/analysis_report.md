# 分析报告：随机高斯码 (Random Gaussian) vs. 重复码 (Repetition)

## 实验目标 (Experiment Goal)
本实验旨在对比 **随机高斯码 (Random Gaussian Codes)** (几何保真路径 Path A) 与传统的 **重复码 (Repetition Codes)** 在高冗余度设置 ($L=5$) 下的有效性。基于 Johnson-Lindenstrauss (JL) 引理，我们假设随机高斯码相比重复码能更好地保持潜空间的几何结构，从而缓解高维重复带来的“维度灾难”和距离膨胀问题。

## 实验设置 (Experimental Setup)
- **数据集**: mfeat (多视图数据集)
- **模型**: CNG-MV-GPLVM (直接推断 Direct Inference)
- **冗余因子 ($L$)**: 5
- **对比基准**:
    1. **Uncoded**: 基准 GPLVM ($L=1$)
    2. **Repetition Code ($L=2$)**: 此前的最佳基准 (Standard benchmark)
    3. **Repetition Code ($L=5$)**: 高冗余度基准
    4. **Random Gaussian Code ($L=5$)**: 本次提出的方法

## 核心发现 (Key Findings)

### 1. 几何保真度 (Latent Space Structure)
*   **观察**: 下方的 t-SNE 可视化图展示了不同模型的潜空间结构。
*   **分析**: 可以看到 **Random Gaussian ($L=5$)** 的潜空间保持了清晰的聚类结构，定性上比 **Repetition ($L=5$)** 更接近 **Uncoded** 或 **Repetition ($L=2$)** 的基准。
    *   **Repetition ($L=5$)** 迫使核函数在高度膨胀的距离空间中运行 ($\|x_i - x_j\|^2 \approx 5 \|z_i - z_j\|^2$)，导致谱混合核 (Spectral Mixture Kernel) 的长度尺度学习变得困难，结构发生扭曲。
    *   **Random Gaussian** 利用 JL 引理的性质，使得投影后的距离期望值保持不变 $E[\|x_i - x_j\|^2] \approx \|z_i - z_j\|^2$，成功在高维空间中保留了原始的几何信息。

![潜空间 t-SNE 对比](/Users/ice/work/study/Coded_SM/results/random_gaussian_vs_repetition_L5/comparison_latent_tsne.png)

### 2. 重构性能 vs. 冗余度 (Reconstruction Performance)
*   **观察**: 下方的雷达图展示了各视图的重构误差 (越小越好)。
*   **分析**: **Random Gaussian ($L=5$)** 在重构误差上显著优于 **Repetition ($L=5$)**，证明了其在高冗余度下的有效性。
*   **不足**: 然而，**Random Gaussian ($L=5$)** 的整体表现仍然略逊于 **Repetition ($L=2$)** 基准。
*   **结论**: 虽然随机高斯编码有效缓解了高冗余度带来的相对性能下降（使 $L=5$ 变得可用），但在当前的直接优化 (Direct Optimization) 框架下，引入更大的输入维度 $X$ 所带来的计算或优化代价，似乎抵消了随机投影可能带来的正则化收益。简言之，目前 $L=5$ 的高冗余度尚未带来“净收益”。

![多视图重构误差雷达图](/Users/ice/work/study/Coded_SM/results/random_gaussian_vs_repetition_L5/comparison_radar_recon.png)

## 假设与后续步骤 (Hypotheses & Next Steps)

**假设**: 随机高斯码的优势可能在于它能在更低的冗余度下依然保持几何性质，或者其“编码增益” (Coding Gain) 需要配合编码器 (Path B) 才能充分释放。

**下一步计划**:
1.  **降低冗余度**: 测试 **Random Gaussian with $L=2$**。如果在低 $L$ 下 JL 性质依然能通过随机矩阵带来某种程度的“各向同性”正则化，它可能会结合低维优势与几何保真优势，从而超越 L=2 重复码。
2.  **直接对比**: 将 **Random Gaussian ($L=2$)** 与 **Repetition ($L=2$)** 进行直接对比，以分离编码矩阵 $\mathbf{G}$ 结构带来的纯粹影响。
