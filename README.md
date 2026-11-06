# CNG-MV-GPLVM

多视图无监督潜变量模型的研究代码，组合共享潜变量、频谱随机傅里叶特征（RFF）、
冗余线性投影以及可选的编码器推断。本修复分支基于 `origin/coded` 的 `a5fd098`；
原 `main` 是较早实现。

## 环境与验证

运行时直接依赖固定在 `requirements.txt`，测试依赖在 `requirements-dev.txt`。
本次实际验证环境为 Python 3.13.13、PyTorch 2.12.1+cu130；真实数据对照在 CPU 上运行。
PyTorch 的 CPU/CUDA 构建由本机安装方式决定，版本号保持一致。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

已有对应版本 PyTorch 的环境可以使用 `python -m venv --system-site-packages .venv`。
`environment.yaml` 与 `server_env.yaml` 均使用同一份运行依赖。
当前模型直接使用 PyTorch，不再依赖 GPyTorch 或 torchvision。

## 模型与训练目标

各视图独立编码，再以包含标准正态先验的 Product of Experts 融合。
所有跨模块概率接口均返回 `(mu, log_sigma)`。

| 推断模式 | 训练与推断路径 |
|---|---|
| `direct` | 逐样本优化潜变量参数；没有新样本编码器 |
| `amortized` | 可用视图 → 编码器 → Z |
| `semi_amortized` | 同时优化 Z；编码器均值对齐到停止梯度的 Z 均值 |
| `coded_amortized` | 可用视图 → 高维 X 后验 → ECC 解码 → Z |
| `coded_semi_amortized` | 优化 Z，同时训练上述带解码的推断路径 |

ECC 使用 `X = ZG`，支持重复投影和随机高斯投影。解码器使用生成路径的同一个 G，
通过伪逆传播均值与对角方差；一般投影产生的非对角协方差仍被省略。
冗余投影不会增加原始信息，也不能单独作为抗噪收益的证明。

默认数据损失为 RFF 近似 GP 边际负对数似然，并保留原项目的 `KL/(N*50)` 缩放。
两个 semi 模式加入 `alignment_beta * sum((mu_enc - stopgrad(mu_opt))**2)`。
全数据 batch 计算完整数据上的近似 GP 似然；小 batch 是分批 GP 目标，
不等价于完整 GP 似然的无偏估计，运行元数据会记录两者的区别。
当前实现与原论文的理论等价性尚未独立验证。

`--no_gp_loss` 使用可训练线性输出层的高斯重构目标。
GP 模式关闭输出层；重构模式固定 RFF 基，以使线性权重对应一致的特征坐标。
评估时使用已保存的固定频率样本和潜变量后验均值。
CUDA GP 训练在支持时对编码器使用 BF16，核、ECC 和 Cholesky 使用 FP32；
不支持 BF16 的设备对 GP 使用 FP32，避免 FP16 梯度溢出。实际精度策略写入元数据。

## 单个实验

```bash
python create_experiment.py \
  --name mfeat_coded_check --dataset mfeat \
  --inference coded_semi_amortized --encoder mlp \
  --z_dim 10 --redundancy 2 --ecc_type repetition \
  --epochs 20 --batch_size 2000 --rff-samples 32 --num-mixtures 2 --lr 0.001

# 将目录名替换为创建命令实际输出的路径；可从该目录执行脚本。
cd experiments/<timestamp>_mfeat_coded_check
bash run.sh
```

生成的脚本保存解释器路径，自动切到仓库根目录。
每次实验包含配置、TensorBoard 日志，以及 `checkpoints/` 下的
`final_model.pth`、`config.yaml`、`metadata.json` 和 `history.csv`。
checkpoint 格式为 v2，包含配置、严格加载的模型状态及数据/代码指纹。
旧 checkpoint 涉及已修复的分布和损失错误，需要重新训练，不能直接比较。

## 评估

```bash
python auto_analyze.py \
  --experiments 'experiments/<timestamp>_mfeat_coded_check' \
  --output results/mfeat_coded_check --device cpu
```

编码器模式报告编码器均值的 NMI/ACC，direct 报告已优化潜变量的 NMI/ACC。
semi 模式另报优化潜变量指标和编码器对齐距离。
鲁棒性通过逐个移除视图评估，报告潜变量变化及缺失视图后的聚类指标。
对齐距离与缺失视图鲁棒性是不同指标。不可计算或未训练的指标写为 NaN，不能解释成零误差。
所有聚类评价都在训练数据上进行，是传导式评价，不代表留出样本上的泛化能力。

配置、数据指纹、checkpoint 或模型参数不匹配时，评估失败；批次中有失败会返回非零退出码。
结果包含逐次指标 CSV、按训练种子汇总的均值/样本标准差、缺失视图明细和指标定义。

## 小规模真实数据复现

```bash
python run_reproducibility.py \
  --dataset mfeat --epochs 20 --seeds 42 43 44 --device cpu \
  --run-dir experiments/my_repro --output reports/my_repro
```

五组模型各训练三个种子，并额外重复一次相同种子，检查全部模型张量、损失历史及指标一致。
请使用新的输出目录。已执行结果见
[验证报告](reports/reproducibility_20261106/REPORT.md)。
20 轮短训练验证了执行与复现流程，没有证明 ECC 的稳定性能优势。
仓库根目录的两张早期 `comparison_*.png` 为历史图表，不能与该报告的结果混用。

## 数据与边界

支持 Mfeat、Caltech101-7、100Leaves 特征数据，以及 ALOI 图像。
仓库已有前三者的原始文件；训练记录文件 SHA256 与预处理后数值指纹。
Caltech/100Leaves 下载失败会报错，绝不生成随机数据替代；训练与评估拒绝显式 mock 数据。
ALOI 需单独准备图片文件；缺失图片会报错。
实际完整 ALOI 数据实验尚未在本次验证中运行。

更多命令见 [USAGE.md](USAGE.md)，概率与指标约定见
[模型指南](docs/model_variants_guide.md)。
