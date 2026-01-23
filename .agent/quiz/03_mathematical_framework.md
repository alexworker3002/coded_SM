# Q&A 记录: CNG-MV-GPLVM 的数学框架
**日期**: 2026-01-23
**主题**: 完整生成模型、NG-SM RFF 证明及 ELBO 详细推导

---

## 1. 生成模型结构 (Generative Model Structure)

CNG-MV-GPLVM (Coded Next-Gen Multi-View GPLVM) 针对观测数据视图 $Y^{(v)} \in \mathbb{R}^{N \times D_v}$ 设定了如下生成过程：

1.  **潜变量采样 (Latent Sampling)**: 
    对样本 $n=1,\dots,N$ 采样低维潜变量：$\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$。
2.  **编码投影 (Coded Projection - ECC)**: 
    通过生成矩阵 $\mathbf{G} \in \mathbb{R}^{d_z \times d_x}$ 将潜变量映射到冗余编码空间 $\mathcal{X}$：
    $$ \mathbf{x}_n = \mathbf{z}_n \mathbf{G} $$
    这里，$\mathbf{G}$ 编码了线性流形结构（如重复码、随机投影等）。
3.  **函数先验 (Function Prior - NG-SM GP)**: 
    这是一个关键概念。我们不直接定义数据分布，而是定义生成数据的**函数** $f$ 的分布。
    对于每个输出维度 $j \in \{1, \dots, D_y\}$，我们假设其由一个独立的随机函数 $f_j(\cdot)$ 生成：
    $$ f_j \sim \mathcal{GP}\left(m(\mathbf{x}) \equiv 0, \; k_{NGSM}(\mathbf{x}, \mathbf{x}'; \theta_{ker})\right) $$
    **符号解析**：
    *   $\mathcal{GP}$: 高斯过程分布，意味着对于任意输入集合 $X$，函数值向量 $\mathbf{f}_j = [f_j(\mathbf{x}_1), \dots, f_j(\mathbf{x}_N)]^\top$ 服从多元高斯分布。
    *   $m(\mathbf{x}) \equiv 0$: 均值函数设为 0，表示我们先验假设数据是去中心化的，或者没有全局偏移。
    *   $k_{NGSM}$: 核函数（协方差函数），决定了函数的平滑度、周期性等性质。这里使用我们自定义的 NG-SM 核。
    
    一旦采样了函数 $f_j$（在 RFF 视角下，等价于采样了权重 $\mathbf{w}_j$），它就固定下来用于生成所有数据点。

4.  **观测 (Observation)**: 
    采样带有高斯噪声的数据：
    $$ y_{nj} = f_j(\mathbf{x}_n) + \epsilon_{nj}, \quad \epsilon_{nj} \sim \mathcal{N}(0, \sigma^2_n) $$

---

## 2. NG-SM 核与 RFF 近似 (NG-SM Kernel & RFF Approximation)

### 2.1 核函数定义
NG-SM 核由其谱密度 $S(\mathbf{\omega})$ 定义 (Yang et al., 2025)：
$$ k(\mathbf{\tau}) = \int_{\mathbb{R}^D} S(\mathbf{\omega}) e^{2\pi j \mathbf{\omega}^\top \mathbf{\tau}} d\mathbf{\omega} $$
$$ S(\mathbf{\omega}) = \sum_{q=1}^Q w_q \mathcal{N}(\mathbf{\omega}; \mathbf{\mu}_q, \mathbf{\Sigma}_q) $$

### 2.2 RFF 近似证明
我们采用 **拼接 Cos/Sin (Concatenated Cos/Sin)** 的形式（选项 B）。
设 $\mathbf{\omega}_{q,s} \sim \mathcal{N}(\mathbf{\mu}_q, \mathbf{\Sigma}_q)$ 为采样频率。
定义特征映射 $\phi(\mathbf{x}) \in \mathbb{R}^{2QS}$：
$$ \phi(\mathbf{x}) = \bigoplus_{q=1}^Q \frac{\sqrt{w_q}}{\sqrt{S}} \left( \begin{bmatrix} \cos(\mathbf{\omega}_{q,1}^\top \mathbf{x}) \\ \vdots \\ \cos(\mathbf{\omega}_{q,S}^\top \mathbf{x}) \end{bmatrix} \oplus \begin{bmatrix} \sin(\mathbf{\omega}_{q,1}^\top \mathbf{x}) \\ \vdots \\ \sin(\mathbf{\omega}_{q,S}^\top \mathbf{x}) \end{bmatrix} \right) $$

**无偏性证明 (Proof of Unbiasedness)**:
我们需要证明 $\mathbb{E}[\phi(\mathbf{x})^\top \phi(\mathbf{x}')] = k(\mathbf{x} - \mathbf{x}')$。
考虑单个成分 $q$ 和采样 $s$ 的内积：
$$ \begin{aligned}
\phi_{q,s}(\mathbf{x})^\top \phi_{q,s}(\mathbf{x}') &= \frac{w_q}{S} [ \cos(\omega^\top \mathbf{x})\cos(\omega^\top \mathbf{x}') + \sin(\omega^\top \mathbf{x})\sin(\omega^\top \mathbf{x}') ] \\
&= \frac{w_q}{S} \cos( \omega^\top (\mathbf{x} - \mathbf{x}') ) \quad \text{(三角恒等式)}
\end{aligned} $$
对 $\omega \sim \mathcal{N}(\mu_q, \Sigma_q)$ 取期望并对 $q, s$ 求和：
$$ \mathbb{E}[\Phi^\top \Phi] = \sum_{q=1}^Q w_q \int \mathcal{N}(\omega; \mu_q, \Sigma_q) \cos(\omega^\top \tau) d\omega = \text{Re}[k_{NGSM}(\tau)] $$
由于核是实对称的，这有效地恢复了 $k(\tau)$。证毕。

---

## 3. 变分推断 (Variational Inference - ELBO Derivation)

我们的目标是最大化观测数据的对数边际似然 $\log p(Y)$。由于积分不可解，我们寻找一个下界（ELBO）。

在这个框架中，我们引入 RFF 近似，将非参数 GP 转化为参数化线性模型：
$$ f_j(\mathbf{x}) \approx \phi(\mathbf{x})^\top \mathbf{w}_j, \quad \mathbf{w}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $$
因此，隐变量包括潜变量矩阵 $Z$ 和 权重矩阵 $W$。

### 3.1 变分分布设定
我们假设变分后验 $q$ 可以根据平均场理论分解：
$$ q(W, Z) = q(W) q(Z) = \left( \prod_{j=1}^{D_y} q(\mathbf{w}_j) \right) \left( \prod_{n=1}^N q(\mathbf{z}_n) \right) $$
具体形式：
*   $q(\mathbf{z}_n) = \mathcal{N}(\mathbf{m}_{z,n}, \text{diag}(\mathbf{s}_{z,n}^2))$
*   $q(\mathbf{w}_j) = \mathcal{N}(\mathbf{m}_{w,j}, \mathbf{\Sigma}_{w,j})$ (或是 Delta 分布如果做 MAP)

### 3.2 ELBO 逐步推导
从对数边际似然开始：
$$ \log p(Y) = \log \iint p(Y, W, Z) dW dZ $$
引入变分分布 $q(W, Z)$ 并乘以/除以它：
$$ = \log \mathbb{E}_{q(W, Z)} \left[ \frac{p(Y | W, Z) p(W) p(Z)}{q(W, Z)} \right] $$
利用 **詹森不等式 (Jensen's Inequality)** ($\log \mathbb{E}[x] \ge \mathbb{E}[\log x]$)：
$$ \ge \mathbb{E}_{q(W, Z)} \left[ \log p(Y | W, Z) + \log \frac{p(W)}{q(W)} + \log \frac{p(Z)}{q(Z)} \right] $$
整理各项得到 ELBO 公式：
$$ \mathcal{L} = \underbrace{\mathbb{E}_{q(W, Z)} [\log p(Y | W, Z)]}_{\text{数据重构项}} - \underbrace{\text{KL}(q(W) \| p(W))}_{\text{模型复杂度惩罚}} - \underbrace{\text{KL}(q(Z) \| p(Z))}_{\text{潜变量正则化}} $$

### 3.3 各项的具体计算

1.  **数据重构项 (Reconstruction)**:
    由于 $y_{nj} \sim \mathcal{N}(\mathbf{w}_j^\top \phi(\mathbf{z}_n\mathbf{G}), \sigma^2)$：
    $$ \log p(Y | W, Z) = \sum_{n,j} \left( -\frac{1}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} (y_{nj} - \mathbf{w}_j^\top \phi(\mathbf{z}_n\mathbf{G}))^2 \right) $$
    我们使用蒙特卡洛积分计算其期望：从 $q(\mathbf{z}_n)$ 采样 $\mathbf{z}_n^{(l)}$，计算 $\phi$，再代入公式。

2.  **潜变量 KL (Latent KL)**:
    $$ \text{KL}(q(\mathbf{z}_n) \| p(\mathbf{z}_n)) = \frac{1}{2} \sum_{d=1}^{d_z} \left( m_{n,d}^2 + s_{n,d}^2 - 1 - \log(s_{n,d}^2) \right) $$
    这是两个高斯分布间的闭式解。

3.  **权重 KL (Weight KL)**:
    如果我们将 $W$ 视为变分参数（Full Bayesian），这也是高斯 KL。
    如果我们只做极大似然估计 (MLE) 或 MAP，这一项变成 $W$ 的 L2 正则化（Weight Decay）。

---

## 4. 为什么要用编码 G？ (理论洞察)

为什么不直接学习 $Z$？
通过固定 $\mathbf{G}$（例如重复或随机投影），我们施加了 **度量约束 (Metric Constraint)**。
核函数看到的距离是：
$$ \| \mathbf{x}_i - \mathbf{x}_j \|^2_{\Sigma^{-2}} = (\mathbf{z}_i - \mathbf{z}_j) \mathbf{G} \Lambda^{-2} \mathbf{G}^\top (\mathbf{z}_i - \mathbf{z}_j)^\top $$
这有效地结构化了梯度流。如果 $\mathbf{G}$ 对应于纠错码，它确保了 $Y$ 中那些会导致 $X$ 空间偏离代码流形的特定维度微扰（噪声）在推断 $Z$ 时被抑制。

这也符合 **Martínez-García et al. (2025)** 的观点，即结构化冗余作为一种正则化器，能够对抗似然评估中的“对抗性”或“噪声”破坏。
