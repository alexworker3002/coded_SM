# 统一的文件组织结构

## 运行实验

```bash
python server_runner.py --config_dir configs/mfeat_z20 --max_workers 7
```

## 文件结构（完全一致）

```
configs/
  └── mfeat_z20/              # 批次配置
      ├── baseline.yaml
      ├── L2_rep.yaml
      └── ...

checkpoints/
  └── mfeat_z20/              # 批次checkpoints（自动创建）
      ├── baseline_xxx/
      │   └── final_model.pth
      ├── L2_rep_xxx/
      └── ...

logs/
  └── mfeat_z20/              # 批次日志（自动创建）
      ├── run_baseline.log
      ├── run_L2_rep.log
      └── ...

results/
  └── mfeat_z20/              # 批次结果（手动分析时创建）
      ├── metrics.csv
      └── figures/
```

**说明：**
- **configs/** - 手动创建（配置源）
- **checkpoints/**, **logs/** - `server_runner.py`自动创建批次子目录  
- **results/** - 分析时创建

## 分析结果

```bash
python auto_analyze.py \
  --experiments "checkpoints/mfeat_z20/*" \
  --output "results/mfeat_z20"
```

或使用自动分析：
```bash
python server_runner.py \
  --config_dir configs/mfeat_z20 \
  --max_workers 7 \
  --auto_analyze \
  --result_name "mfeat_z20"
```

## 优势

✅ **完全一致** - configs, checkpoints, logs, results都按batch_name组织  
✅ **易于管理** - 每个数据集/实验批次独立目录  
✅ **无需手动创建** - checkpoints和logs自动创建  
✅ **可追溯** - 所有相关文件都在一个批次下
