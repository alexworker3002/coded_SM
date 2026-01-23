# Q&A Record: NG-SM Kernel RFF Approximation
**Date**: 2026-01-23
**Module**: `src/kernels/ng_sm_kernel.py`

---

## 1. Random Fourier Features (RFF) for NG-SM

### Q: NG-SM 核函数的特征映射 $\phi(x)$ 具体形式是什么？
**Context**:
Next-Gen Spectral Mixture (NG-SM) 核通过建模谱密度 $S(\omega)$ 来学习复杂的非平稳模式。为了在变分推断中高效计算，我们需要显式的特征映射 $\mathbf{z} = \phi(\mathbf{x})$，满足 $k(\mathbf{x}, \mathbf{x}') \approx \phi(\mathbf{x})^\top \phi(\mathbf{x}')$。

**Derivation**:
根据 Bochner 定理，$k(\tau) = \int S(\omega) e^{j\omega^\top \tau} d\omega$。
NG-SM 假设 $S(\omega) = \sum_{q=1}^Q w_q \mathcal{N}(\omega | \mu_q, \Sigma_q)$。

我们采用 **Concatenated Cos/Sin** 形式（无需随机相位 $b$）以降低估计方差：

$$
\phi(\mathbf{x}) = \text{Cat} \left( \left[ \frac{1}{\sqrt{S}} \sqrt{w_q} \cos(\omega_{q,s}^\top \mathbf{x}), \quad \frac{1}{\sqrt{S}} \sqrt{w_q} \sin(\omega_{q,s}^\top \mathbf{x}) \right] \right)_{q=1..Q, s=1..S}
$$

其中：
- $Q$: 混合成分数量 (Mixture Components)
- $S$: 每个成分的蒙特卡洛采样数 (Output Dim per Mixture)
- $\omega_{q,s} \sim \mathcal{N}(\mu_q, \Sigma_q)$: 采样的频率向量
- 总输出维度 $D_{out} = 2 \cdot Q \cdot S$

**Advantage (vs Standard RFF)**:
标准的 RFF 使用 $\cos(\omega^\top \mathbf{x} + b)$。虽然维度减半 ($D_{out} = Q \cdot S$)，但由于 $b \sim U[0, 2\pi]$ 的随机性，其收敛速度通常慢于 Cos/Sin 拼接法。对于变分推断，去除 $b$ 使得对潜变量的期望计算更加直接。

**Implementation Plan**:
1. Remove `random_biases` buffer.
2. Sample $\omega_{q,s} = \Sigma_q^{1/2} \epsilon + \mu_q$, where $\epsilon \sim \mathcal{N}(0, I)$.
3. Compute projection $P = \mathbf{x} \Omega^\top$.
4. Return $[\text{scale} \cdot \cos(P), \text{scale} \cdot \sin(P)]$.
