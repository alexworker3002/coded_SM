# AI Agent 指南：CNG-MV-GPLVM 实验分析与对比逻辑

本文件旨在指导服务器端的 AI Agent 理解本项目的实验结构、模型变体及分析方法，以便自主执行实验监控和结果报告生成任务。

---

## 1. 模型体系 (Model Taxonomy)

我们的模型基于 `inference_mode` 和 `ecc_type` 的组合划分为不同变体。Agent 在分析时应根据这些标签对实验进行分类。

### 1.1 核心推断模式 (Inference Modes)
*   **Tier 1: 基线 (Baseline)**
    *   `amortized`: 纯 VAE 模式（Yang等人的基线），速度快但在复杂核上不稳定。
*   **Tier 2: 稳健 (Robust / Standard)**
    *   `semi_amortized`: 当前的主力模型。直接优化参数 + 编码器对齐。
    *   `direct`: 仅用于小规模数据的理论上限参考（无编码器）。
*   **Tier 3: 实验性 (Experimental / Advanced)**
    *   `coded_amortized`: "Infer-X-Decode-Z"。利用 ECC 解码来通过推断路径抗噪。
    *   `coded_semi_amortized`: 结合了 Semi 的锚点稳定性和 Coded 的推断抗噪。

### 1.2 纠错类型 (ECC Types)
*   `none` (L=1): 对照组。
*   `repetition` (L>1): 结构化强正则，通常表现最好。
*   `random_gaussian` (L>1): 理论上的 RIP 保持，但在低冗余度下可能不如重复码稳定。

---

## 2. 对比逻辑 (Comparison Logic)

Agent 在生成报告时，应遵循以下标准的对比组（Comparison Groups）：

### Group A: ECC 有效性验证 (Effectiveness of ECC)
**目的**: 证明引入冗余信息（L>1）能提升聚类性能。
**筛选条件**: 固定 `dataset`, `z_dim`, `inference=semi_amortized`。
**对比项**:
1.  **Baseline**: `ecc_type=none`
2.  **Repetition**: `ecc_type=repetition, L=2` (期待胜出)
3.  **Random**: `ecc_type=random_gaussian, L=2`

### Group B: 推断流向研究 (Inference Flow Study)
**目的**: 验证“先编码为 X 再解码” (`coded_*`) 是否比“直接编码为 Z” (`semi_*`) 更稳健。
**筛选条件**: 固定 `dataset`, `z_dim`, `ecc_type=repetition`, `L=2`。
**对比项**:
1.  **Standard Semi**: `inference=semi_amortized` (Feature: Z -> X -> K)
2.  **Coded Semi**: `inference=coded_semi_amortized` (Feature: Y -> X -> Z -> X -> K)
**关注点**: `latent_shift_mean`（越低越好）和 `NMI`（越高越好）。

### Group C: 稳定性边界测试 (Stability Boundary)
**目的**: 探究高维输入的崩溃点。
**筛选条件**: 固定 `inference=coded_semi_amortized`。
**变量**: `redundancy` (L) = [1, 2, 5, 10]。
**分析指标**: 检查 NMI 是否随着 L 增加而急剧下降（崩溃）。

---

## 3. 分析方法 (Analysis Methodology)

Agent 应当编写脚本（如 `auto_analyze.py`）执行以下流程：

### 3.1 数据聚合
从 `results/<dataset>_final/comprehensive_metrics.csv` 读取所有数据。该 CSV 包含关键列：
*   `nmi`, `acc`: 聚类性能（核心 KPI）。
*   `latent_shift_mean`: 编码器与最优参数的对齐度（越低表示推断越准）。
*   `avg_recon_error`: 重构误差（检查是否模式坍塌）。
*   `z_mean_norm`: 潜变量范数（若 >100 则表示爆炸）。

### 3.2 自动报告模板
对于每个 Group，生成 Markdown 表格：

**示例表格 (Group A):**
| Model      |   L   |   NMI    |   ACC    | Latent Shift | Status   |
| :--------- | :---: | :------: | :------: | :----------: | :------- |
| Baseline   |   1   |   0.88   |   0.70   |     0.15     | ✅ Normal |
| Repetition |   2   | **0.91** | **0.75** |     0.12     | 🚀 SOTA   |
| Random     |   2   |   0.89   |   0.72   |     0.14     |          |

### 3.3 异常检测
Agent 必须自动标记以下异常：
*   **NaN Loss**: 训练日志中出现 NaN。
*   **Explosion**: `z_mean_norm > 10` 或 `avg_recon_error > 5`。
*   **Collapse**: `NMI < 0.1` (表示模型将所有点分到了同一类)。

---

## 4. 指令示例

当用户发送指令如：“分析 100Leaves 的最新实验结果”时，Agent 应：
1.  定位 `results/leaves_final`。
2.  加载最新的 CSV。
3.  按上述 `Group A/B/C` 构建对比视图。
4.  输出一份包含“最佳配置推荐”和“失败案例分析”的简报。

---
*Reference: `docs/model_variants_guide.md`, `docs/coded_inference_research.md`*
