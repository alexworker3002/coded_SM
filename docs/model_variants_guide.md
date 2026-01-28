# Coded-SM 模型架构与配置指南

本文档全面梳理 CNG-MV-GPLVM 框架支持的**推断模式**、**纠错编码 (ECC)** 及**组合策略**，帮助设计对比实验。

---

## 1. 核心推断模式 (Inference Modes)

我们支持五种推断模式，分别对应不同的 $Y \to Z$ 路径和优化策略：

| 模式名称                 | 命令行参数                         | 流程描述                                                                   | 适用场景                            | 关键机制                                                 |
| :----------------------- | :--------------------------------- | :------------------------------------------------------------------------- | :---------------------------------- | :------------------------------------------------------- |
| **Direct Optimization**  | `--inference direct`               | $\mu, \sigma$ 是直接优化的参数，不通过编码器。                             | 小数据集 (N < 2000)，追求最高精度。 | **最纯净**: 无平摊误差。                                 |
| **Amortized (VAE)**      | `--inference amortized`            | $Y \xrightarrow{Enc} Z$                                                    | 大数据集，需快速前向推断。          | **最快**: 仅优化 Encoder 权重。                          |
| **Semi-Amortized**       | `--inference semi_amortized`       | 同时维护 Direct $Z_{opt}$ 和 Encoder $Z_{enc}$，通过 Alignment Loss 对齐。 | 既要高精度又要泛化能力。            | **双流**: 保持 Anchor 稳定，训练 Encoder 逼近。          |
| **Coded-Amortized**      | `--inference coded_amortized`      | $Y \xrightarrow{Enc} X_{noisy} \xrightarrow{Dec} Z_{clean}$                | 这里的 $X$ 是 $L \cdot Z$ 维。      | **推断抗噪**: 利用 ECC 解码去除 Encoder 预测噪声。       |
| **Coded-Semi-Amortized** | `--inference coded_semi_amortized` | $Y \xrightarrow{Enc} X \xrightarrow{Dec} Z_{enc} \leftrightarrow Z_{opt}$  | 高精度 + 抗噪推断的终极组合。       | **Coded Alignment**: 迫使 Encoder 学习可解码的高维结构。 |

---

## 2. 模式拓扑图解

### 2.1 Standard Semi-Amortized
```mermaid
graph LR
    subgraph Direct
    Z_opt[Z_opt (Param)]
    end
    subgraph Amortized
    Y --> Enc[Encoder] --> Z_enc
    end
    Z_opt <-->|Alignment Loss| Z_enc
    Z_opt -->|Reparam| Z_sample -->|ECC| X -->|Kernel| Y_recon
```

### 2.2 Coded-Amortized
```mermaid
graph LR
    Y --> Enc[Encoder] --> X_noisy[X (High Dim)]
    X_noisy -->|ECC Decoder| Z_clean
    Z_clean -->|Reparam| Z_sample -->|ECC| X_clean -->|Kernel| Y_recon
```
> **注意**: 这里的关键是推断路径 $Y \to X$。即使生成时我们也会再把 $Z$ 映射回 $X$ (use_ecc=True)，Coded-Amortized 的核心贡献在于利用冗余性来“清洗”推断结果。

---

## 3. 纠错编码类型 (ECC Types)

ECC 用于扩展潜在空间的维度结构，从 $Z \in \mathbb{R}^k$ 到 $X \in \mathbb{R}^{L \cdot k}$。

| 类型                | 参数                         | 描述                | 数学原理                                                       |
| :------------------ | :--------------------------- | :------------------ | :------------------------------------------------------------- |
| **None**            | `--ecc_type none`            | 不使用 ECC，$L=1$。 | 标准 GPLVM/VAE 基线。                                          |
| **Repetition**      | `--ecc_type repetition`      | 重复 $L$ 次         | $X = [Z, Z, ..., Z]$。<br>通过简单的平均机制降低方差。         |
| **Random Gaussian** | `--ecc_type random_gaussian` | 随机投影矩阵 $G$    | $X = ZG$。<br>利用高维几何的集中现象 (RIP 性质) 保持距离结构。 |

---

## 4. 实验组合与对比策略

您可以通过组合上述选项来构建不同的实验组。

### 4.1 经典对比组：ECC 的有效性
**目标**: 验证引入冗余是否能提升聚类性能。
*   **Baseline**: `inference=semi_amortized`, `ecc=none`
*   **Exp A**: `inference=semi_amortized`, `ecc=repetition`, `L=2`
*   **Exp B**: `inference=semi_amortized`, `ecc=random_gaussian`, `L=2`

### 4.2 进阶对比组：推断流向 (Inference Flow)
**目标**: 验证 "先编码为 X 再解码" (`coded`) 是否比 "直接编码为 Z" (`semi`) 更鲁棒。
*   **Model A (Standard)**: `inference=semi_amortized`, `ecc=repetition`, `L=2`
*   **Model B (Coded)**: `inference=coded_semi_amortized`, `ecc=repetition`, `L=2`
> **Hypothesis**: 如果数据包含大量观测噪声，Model B 的 Decoder 步骤应该能起到很好的去噪作用，使得 $Z_{enc}$ 更稳定。

---

## 5. 推荐配置清单

1.  **追求 SOTA (Best Accuracy)**:
    ```bash
    --inference semi_amortized --ecc_type repetition --redundancy 2 --z_dim 64
    ```

2.  **鲁棒性研究 (Robustness Research)**:
    ```bash
    --inference coded_semi_amortized --ecc_type random_gaussian --redundancy 5
    ```
    *注意：使用 coded_semi_amortized 时，建议开启梯度裁剪，因为涉及高维投影。*

3.  **快速基线 (Fast Baseline)**:
    ```bash
    --inference amortized --ecc_type none --z_dim 32
    ```

---
*文档作者: Antigravity Agent*
