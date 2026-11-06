# 模型与概率接口约定

五种模式统一由 `src/utils/experiment.py` 构建，训练与评估不得独立猜测模型参数。

| 模式 | 编码器目标维度 | 训练中的生成潜变量 | 编码器训练目标 |
|---|---|---|---|
| direct | 无 | 逐样本 Z | 无 |
| amortized | Z | 编码器 Z | 数据损失与 KL |
| semi_amortized | Z | 逐样本 Z | 均值对齐到停止梯度的优化 Z |
| coded_amortized | X=L×Z | 解码后的编码器 Z | 数据损失与 KL |
| coded_semi_amortized | X=L×Z | 逐样本 Z | 解码后的均值对齐到停止梯度的优化 Z |

## 编码与解码

`X=ZG` 中 G 的形状为 `(d_z, L*d_z)`，生成与解码必须使用同一个矩阵。
重复投影按坐标重复，`G = I ⊗ ones(1,L)`；随机高斯矩阵元素标准差为 `1/sqrt(L*d_z)`。
两者的范数缩放不同，实验中的性能差异不能自动归因于纠错能力。

所有跨模块分布接口均为 `(mu, log_sigma)`，其中 `var=exp(2*log_sigma)`。
设伪逆为 `G_pinv`：

```text
mu_z = mu_x @ G_pinv
var_z = exp(2*log_sigma_x) @ (G_pinv ** 2)
log_sigma_z = 0.5 * log(var_z)
```

这只保留传播后协方差的对角线。重复码中独立单位方差的 L 个预测，其平均方差为 `1/L`。
不能把 `logvar` 直接送进需要 `log_sigma` 的重参数化或 KL 计算。

## 损失与随机性

GP 目标通过 `K=Phi Phi^T + noise_var I` 计算边际 NLL。
依据样本数和特征数选择直接 Cholesky 或 Woodbury 公式，两条路径使用一致的有效噪声方差。
本实现保留 `KL/(N*50)` 的项目约定，两个 semi 模式使用相同的均值对齐机制。

GP 训练会重采样频率；评估频率作为 buffer 保存在 checkpoint 中。
重构训练固定频率，以维持输出层的特征坐标。
评估使用潜变量均值。固定种子控制 Python、NumPy、PyTorch、DataLoader 及确定性算法。

## 评估边界

完整视图聚类、缺失视图聚类、编码器对齐、重构误差分别报告。
direct 没有新样本编码器，因此缺失视图推断指标为 NaN。
GP 模式不训练线性重构输出层，因此重构 MSE 为 NaN。
完整/缺失视图的 KMeans 分别拟合，类别标签仅参与评价。
这些指标来自训练样本，不能当成留出样本测试结果。

各推断模式是否优于其他模式、冗余投影是否提高抗噪性，必须通过相同预算下的多种子实验判断。
