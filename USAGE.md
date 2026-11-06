# 实验运行与结果核查

在仓库根目录激活 `.venv` 后执行命令。环境和数据要求见 README。

## 单次训练

```bash
python -m src.trainer.engine --config configs/mfeat_z20/baseline.yaml
```

也可用 `create_experiment.py` 创建独立实验；其 `run.sh` 可从实验目录执行。
`--experiment_dir` 将所有产物放到该目录下，已有完成 checkpoint 时拒绝覆盖。

## 批次训练

```bash
python server_runner.py \
  --config_dir configs/mfeat_z20 --max_workers 1 \
  --auto_analyze --result_name mfeat_z20
```

每次调用创建时间戳批次，并把配置快照保存在 `configs/archive/<batch_id>`。
checkpoint 位于 `checkpoints/<batch_id>/<experiment_name>_<timestamp>/`。
批次之间不会自动复用旧 checkpoint。任意训练失败会返回非零退出码，并停止自动分析。
GPU 数量由运行器检测；`--max_workers` 控制并发任务数。每个任务占用一个 GPU
槽位直到完成，父进程的 `CUDA_VISIBLE_DEVICES` 也会被保留。将工作进程数设为
高于 GPU 数量时，属于显式允许多个任务共享同一 GPU。

## 独立评估

```bash
python auto_analyze.py \
  --experiments 'checkpoints/<batch_id>/*' \
  --output results/<batch_id> --device cpu --evaluation-seed 42
```

独立实验目录和其 `checkpoints/` 子目录都可以作为输入。
checkpoint 的配置、数据数值/顺序和全部参数必须匹配。
旧格式 checkpoint 会被拒绝，需要重新训练。

输出：

- `comprehensive_metrics.csv`：逐个训练种子的指标与 checkpoint 路径。
- `summary.csv`：相同配置下的均值、样本标准差、有效值数量、独立种子数。
- `missing_view_metrics.json`：逐个缺失视图的 ACC/NMI/潜变量变化。
- `metric_definitions.json`：指标定义与传导式评价范围。
- `failures.json`：失败实验及原因；有任何失败时进程返回非零退出码。
- `*_comparison.png`：完整视图与缺失视图准确率比较。

`alignment_shift_mean` 是编码器与优化参数的距离，仅适用于 semi 模式；
`robustness_shift_mean` 是移除视图后的距离，仅适用于编码器模式。
GP 模式没有训练重构输出层，所以 `avg_recon_error` 留空。

## 重复性对照

```bash
python run_reproducibility.py \
  --dataset mfeat --epochs 20 --seeds 42 43 44 --device cpu \
  --run-dir experiments/repro-new --output reports/repro-new
```

输出目录必须全新。报告会保存真实数据文件散列、预处理数值指纹、代码指纹、版本和配置预算。
额外的同种子重复不计入三种子的统计汇总。
检查实现是否能复现与判断算法是否优于对照是不同任务；短训练不能承担最终性能结论。
