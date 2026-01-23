# Q&A Record: ECC Implementation & Environment Setup
**Date**: 2026-01-23
**Module**: `src/modules/ecc.py`, Environment Configuration

---

## 1. ECC Matrix Construction (Linear Manifold Regularization)

### Q: 如何正确实现线性纠错码（Linear Code）的生成矩阵 G？
**Context**: 
我们在 `src/modules/ecc.py` 中实现 `LinearECCProjection`。最初作为 Fallback 的重复码（Repetition Code）逻辑是简单的块状复制。

**Discussion**:
用户指出代码实现与 Coded-Vae 论文截图中的形式不符。
- **Initial Implementation (Block Repetition)**:
  $$ G_{code} = [I, I, \dots, I] $$
  输出形式为 $[z_1, z_2, \dots, z_1, z_2, \dots]$。
- **Paper Specification (Element-wise Repetition)**:
  截图显示矩阵呈阶梯状结构：
  $$ G_{paper} = I \otimes \mathbf{1}^T = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix} $$
  输出形式为 $[z_1, z_1, z_2, z_2, \dots]$。

**Conclusion & Action**:
虽然两者在数学秩（Rank）上等价，但为了严格对齐论文可视化及后续处理，我们修改了代码，使用 Kronecker Product (`torch.kron`) 替代简单的 `torch.cat`。

**Mathematical Formulation**:
$$ \mathbf{X} = \mathbf{Z} \mathbf{G}, \quad \text{where } \mathbf{G} = \mathbf{I}_{d_z} \otimes \mathbf{1}_{1 \times L} $$
此操作将潜变量空间 $\mathcal{Z} \subset \mathbb{R}^{d_z}$ 映射到编码空间 $\mathcal{X} \subset \mathbb{R}^{L \cdot d_z}$。

---

## 2. Environment Reconstruction

### Q: 是否需要重建虚拟环境以适配 CNG-MV-GPLVM 项目需求？
**Context**: 
当前环境 `vae_test` 存在包版本混乱（scipy pip vs conda 版本不一致）及架构优化不足的问题。

**Diagnosis**:
1. **Version Conflict**: `pip list` 显示 `scipy` 版本冲突，可能导致线性代数运算（如 Cholesky 分解）的不稳定。
2. **Project Isolation**: 新一代科研项目应避免与旧的 experiment 混用环境。
3. **MPS Optimization**: 需要确保 PyTorch 是为 Apple Silicon (M2/M3) 编译的 Nightly 版本。

**Action**:
创建了专用的 `environment.yaml` 并重建环境 `cng_mvlvm`。

**Core Dependencies**:
- **Python**: 3.12 (Conda-forge)
- **PyTorch**: Nightly Build (MPS Optimized)
- **GPyTorch**: Latest Stable (1.15.1)
- **Math/Viz**: NumPy, SciPy, Matplotlib, Seaborn

**Status**:
环境已成功创建并就绪。后续所有 `src/kernels` 和 `src/trainers` 的开发将在此纯净环境中进行。
