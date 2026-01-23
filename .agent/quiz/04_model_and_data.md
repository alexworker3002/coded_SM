# Q&A 记录: 模型实现与数据规格
**日期**: 2026-01-23
**模块**: `src/models/cng_model.py`, `src/utils/data_utils.py`

---

## 1. 模型架构实现 (Model Architecture)

### Q: 如何构建支持消融实验的 CNG-MV-GPLVM 模型？
**Context**:
为了验证 Coded Regularization 的有效性，模型需要能够方便地在该功能开启/关闭之间切换。同时，我们需要从零实现变分推断逻辑，而不是依赖 GPyTorch 的黑盒 `VariationalStrategy`。

**Implementation Details (`src/models/cng_model.py`)**:
我们实现了一个继承自 `nn.Module` 的自定义类 `CNG_MV_GPLVM`。

1.  **可配置性 (Ablation Support)**:
    - 引入 `use_ecc` (bool) 参数。
    - `True`: 启用 `LinearECCProjection` ($Z \to X_{coded}$)。
    - `False`: 使用 `nn.Identity` ($Z \to Z$)，退化为基准 NG-GPLVM。

2.  **变分推断 (Inference Strategy)**:
    - 采用 **Amortized Inference** 的简化形式（或者说是 VS-GPLVM 风格的全局参数）：
    - $q(\mathbf{z}_n) \sim \mathcal{N}(\mathbf{\mu}_n, \mathbf{\sigma}_n^2 \mathbf{I})$。
    - 直接将 $\mathbf{\mu}$ 和 $\log \mathbf{\sigma}$ 注册为 `nn.Parameter` 进行优化。

3.  **读出层 (Readout)**:
    - 输入：NG-SM 核产生的 RFF 特征 $\phi(\mathbf{x}) \in \mathbb{R}^{2QS}$。
    - 输出：通过线性层映射到观测空间 $\mathbb{R}^{D_y}$。
    - 这实际上隐式地执行了贝叶斯线性回归（或者说是权重空间视角的 GP）。

---

## 2. 数据集规格 (Dataset Specifications)

### Q: 我们使用什么多视图数据进行实验？
**Dataset**: **UCI Multiple Features (mfeat)**
手写数字 (0-9) 的多视图特征数据集。

**Statistics**:
- **Samples**: $N = 2000$ (10 classes $\times$ 200 samples/class).
- **Views**: 6 个异构视图。

**Dimensions ($D_v$)**:
| View Name | Description | Dimensions | Type |
| :--- | :--- | :--- | :--- |
| **fac** | Profile correlations | 216 | Dense |
| **fou** | Fourier shape descriptors | 76 | Dense |
| **kar** | Karhunen-Love coefficients | 64 | Dense |
| **pix** | Pixel averages | 240 | Image-like |
| **zer** | Zernike moments | 47 | Dense |
| **mor** | Morphological features | 6 | Dense |

**Preprocessing**:
- 所有视图均使用 `StandardScaler` 进行标准化 (零均值，单位方差)，以适配 GP 核函数的敏感度。
