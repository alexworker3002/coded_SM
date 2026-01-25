# Semi-Amortized 架构：数学逻辑与推导

## 1. 动机：摊销空隙 (The Amortization Gap)
在变分推理 (Variational Inference) 中，我们的目标是最大化证据下界 (ELBO)：
$$ \log p(Y) \ge \mathbb{E}_{q(Z)}[\log p(Y|Z)] - KL(q(Z)||p(Z)) $$

定义近似后验 $q(Z)$ 通常有两种方式：
1.  **直接优化 (Direct Optimization, 如 SMLVM)**：将 $Z$ 视为自由的变分参数 $\lambda = \{ \mu_i, \sigma_i \}_{i=1}^N$。
    *   **优点**：能找到最优的流形结构 ($\mathbf{Z}^*$)，NMI 的上限。
    *   **缺点**：无法对新数据进行实时推断。
2.  **摊销推断 (Amortized Inference, 如 VAE)**：使用共享的神经网络进行映射 $Z = f_\phi(X)$。
    *   **优点**：推断速度快，支持鲁棒性测试。
    *   **缺点**：编码器 $f_\phi$ 可能无法完美映射到最优 $Z^*$，从而产生 **摊销空隙 (Amortization Gap)**：
        $$ \Delta_{gap} = \text{ELBO}(\mathbf{Z}^*) - \text{ELBO}(f_\phi(X)) \ge 0 $$

我们的目标是在保留推断能力的同时，**消除或缩小这一空隙**。

## 2. Semi-Amortized 学习目标
我们提出了一种混合训练目标函数，将 **流形学习 (Manifold Learning)** 与 **推断学习 (Inference Learning)** 进行解耦。

### 2.1 联合损失函数
令 $\mathbf{Z}_{opt}$ 为直接优化的变分参数，$\mathbf{Z}_{enc} = f_\phi(X)$ 为编码器的输出。我们最大化以下联合目标：

$$ \mathcal{J}(\mathbf{Z}_{opt}, \phi, \theta) = \underbrace{\text{ELBO}(\mathbf{Z}_{opt}, \theta)}_{\text{流形优化}} - \beta \cdot \underbrace{\mathcal{R}_{align}(\mathbf{Z}_{opt}, \mathbf{Z}_{enc})}_{\text{推断对齐}} $$

其中：
*   **项 1 (ELBO)**：驱动 $\mathbf{Z}_{opt}$ 利用 NG-SM 核函数的全部表达能力来寻找数据的真实底层流形。
*   **项 2 (Alignment)**：强迫编码器 $f_\phi$ 模仿 $\mathbf{Z}_{opt}$ 的布局。
    $$ \mathcal{R}_{align} = \sum_{i=1}^N \| \mu_{opt}^{(i)} - \mu_\phi(X^{(i)}) \|_2^2 $$

### 2.2 梯度流动逻辑
*   **针对 $\mathbf{Z}_{opt}$**：它同时接收来自重建项和对齐项的梯度。
*   **针对编码器 $\phi$**：它纯粹为了最小化对齐误差而优化。它不需要在训练中直接权衡“重建 vs KL 散度”，而是通过**插值 (Interpolate)** 学习由 SMLVM 找到的最优流形映射。这保证了编码器即便在推断时，也能达到接近 SMLVM 级别的聚类质量。

## 3. 1D-CNN 平滑效应的数学直觉
在 Coded VAE 中，我们利用线性纠错码 (Linear ECC) 将 $Z \to \mathbf{X} \in \mathbb{R}^{L \cdot d_z}$。当某些视图 $Y_k$ 被遮蔽 (Masked) 时，输入信号域会产生不连续的断裂或阶跃伪影。

### 为什么选择 1D-CNN？
如果将多视图特征序列 $\tilde{Y} = [Y_1, ..., Y_v]$ 视为伪一维序列，CNN 的卷积操作具有以下特性：
*   **局部等变性 (Equivariance)**：即使输入序列的某些部分失效，卷积核在有效区域提取到的底层特征是不变的。
*   **平滑与阻尼**：卷积本质上是一种滑动加权平均。这种操作能过滤输入中的高频噪声（即遮蔽产生的边界点），并平滑因维度缺失带来的激活突变，从而在 $Z$ 空间中表现为更稳定的位移（更小的 Latent Shift）。

结合 **Alignment Loss**，CNN 编码器学会了即便在部分视图缺失时，也能通过其平滑的映射函数，稳定地定位到由 $\mathbf{Z}_{opt}$ 所锚定的高纯度语义坐标。
