# 实验分析约定

统一入口为 `auto_analyze.py`。先核对 checkpoint 格式、完整参数、配置散列和预处理后的数据指纹。
任一步失败，该实验不得产生有效指标；失败需写入 `failures.json` 并通过非零退出码体现。

比较必须固定数据、潜变量维度、推断模式、网络、核混合数/RFF 数、训练轮数、学习率和 batch 目标。
`variant_id` 表示去除实验名称和训练种子后的配置散列。
均值/样本标准差按该配置分组，同时报告有效值数量和独立种子数。
相同种子的重复用于检查复现，不充当独立统计样本。

解释指标时遵守 `metric_definitions.json`：

- `acc`、`nmi` 为传导式聚类评价，编码器模式使用编码器后验均值。
- semi 模式的 `optimized_acc`、`optimized_nmi` 单独表示优化潜变量表现。
- `alignment_shift_mean` 只描述编码器与优化潜变量的对齐。
- `robustness_shift_mean` 与 `masked_acc_mean` 等来自逐个移除视图的实验。
- GP 模式重构误差、direct 缺失视图指标以及其他不可用值为 NaN。

不能预设重复投影或任何编码必然获胜；不按实验名称贴“SOTA”标签。
短训练、少量种子和训练样本上的结果不足以支持最终性能或泛化结论。
已有小规模验证报告见 `reports/reproducibility_20261106/REPORT.md`。
