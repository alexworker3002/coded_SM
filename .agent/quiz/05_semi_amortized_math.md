# Semi-Amortized Architecture: Mathematical Logic & Derivation

## 1. Motivation: The Amortization Gap
In Variational Inference, we aim to maximize the Evidence Lower Bound (ELBO):
$$ \log p(Y) \ge \mathbb{E}_{q(Z)}[\log p(Y|Z)] - KL(q(Z)||p(Z)) $$

We have two ways to define the approximate posterior $q(Z)$:
1.  **Direct Optimization (SMLVM)**: treating $Z$ as free variational parameters $\lambda = \{ \mu_i, \sigma_i \}_{i=1}^N$.
    *   Pros: Finds the optimal manifold structure ($\mathbf{Z}^*$).
    *   Cons: No inference capability for new data.
2.  **Amortized Inference (VAE)**: using a shared neural network $Z = f_\phi(X)$.
    *   Pros: Fast inference, enables robustness tests.
    *   Cons: The encoder $f_\phi$ may fail to map $X$ to the optimal $Z^*$, creating an **Amortization Gap**:
        $$ \Delta_{gap} = \text{ELBO}(\mathbf{Z}^*) - \text{ELBO}(f_\phi(X)) \ge 0 $$

Our goal is to **close this gap** while retaining the inference capability.

## 2. Semi-Amortized Objective
We propose a hybrid training objective that decouples **manifold learning** from **inference learning**.

### 2.1 The Joint Loss
Let $\mathbf{Z}_{opt}$ be the direct variational parameters, and $\mathbf{Z}_{enc} = f_\phi(X)$ be the encoder output. We maximize:

$$ \mathcal{J}(\mathbf{Z}_{opt}, \phi, \theta) = \underbrace{\text{ELBO}(\mathbf{Z}_{opt}, \theta)}_{\text{Manifold Optim.}} - \beta \cdot \underbrace{\mathcal{R}_{align}(\mathbf{Z}_{opt}, \mathbf{Z}_{enc})}_{\text{Inference Alignment}} $$

Where:
*   **Term 1 (ELBO)**: Drives $\mathbf{Z}_{opt}$ to find the true underlying manifold of the data, utilizing the full power of the NG-SM kernels.
*   **Term 2 (Alignment)**: Forces the Encoder $f_\phi$ to mimic $\mathbf{Z}_{opt}$.
    $$ \mathcal{R}_{align} = \sum_{i=1}^N \| \mu_{opt}^{(i)} - \mu_\phi(X^{(i)}) \|_2^2 $$

### 2.2 Gradient Flow
*   **For $\mathbf{Z}_{opt}$**: It receives gradients from both ELBO (reconstruction) and Alignment. In practice, we often `detach()` $\mathbf{Z}_{opt}$ in the alignment term to stabilize training, using it as a "Teacher" target for the "Student" Encoder.
*   **For Encoder $\phi$**: It is purely optimized to minimize the Alignment error. It does not need to balance the KL-divergence vs Reconstruction trade-off itself; it just learns to **interpolate** the optimal manifold found by SMLVM.

## 3. Robustness thru 1D-CNN Smoothing
When using Linear ECC, we map $Z \to \mathbf{X} \in \mathbb{R}^{L \cdot d_z}$. The encoder observes multiple views $Y = \{Y_1, ..., Y_v\}$.
SMLVM cannot handle missing views (Shift is undefined).
Semi-Amortized Encoder can.

### Why 1D-CNN?
If we treat the concatenated heterogeneous features $\tilde{Y} = [Y_1, ..., Y_v]$ as a pseudo-sequence.
When a view block $Y_k$ is masked (set to 0), it creates a "step function" artifact in the input signal domain.
*   **MLP (Fully Connected)**: $h = W \cdot \tilde{Y}$. Every neuron is affected globally by the zeroed block. Scaling factors shift linearly, causing global bias.
*   **1D-CNN (Local)**: $(f * g)[n] = \sum_k f[n-k]g[k]$.
    *   **Smoothing Property**: Convolution acts as a sliding average. The "edge" between valid data and zeroed data is smoothed out.
    *   **Equivariance**: The features extracted from the valid regions $Y_{valid}$ remain spatially invariant (or equivariant), preserving their local semantic contribution to $Z$.

Combined with **Alignment Loss**, the CNN Encoder learns to map even partial inputs ($Y_{partial}$) to the same intrinsic coordinate $\mathbf{Z}_{opt}$ that was learned from full data, significantly reducing Latent Shift.
