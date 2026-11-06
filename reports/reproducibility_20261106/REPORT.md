# Real-data reproducibility check

- Dataset: mfeat, N=2000, all available views.
- Controls: five variants, seeds [42, 43, 44], 20 epochs each; one extra same-seed repeat.
- Identical training budget: Z=10, Q=2, 32 RFF samples per mixture, full-data batches, LR=0.001.
- Evaluation: transductive KMeans (20 starts, seed 42), encoder means, leave-one-view-out robustness.
- All repeated model tensors, loss histories and evaluated metrics match exactly.
- GP readouts are disabled; reconstruction MSE is unavailable, not zero.
- This short run validates the repaired pipeline and does not establish a performance advantage.

| Mode | ECC | Seeds | ACC mean ± SD | NMI mean ± SD | Masked ACC mean ± SD |
|---|---|---:|---:|---:|---:|
| coded_semi_amortized | random_gaussian L=2 | 3 | 0.3518 ± 0.0319 | 0.2106 ± 0.0240 | 0.3832 ± 0.0309 |
| coded_semi_amortized | repetition L=2 | 3 | 0.2968 ± 0.0265 | 0.1534 ± 0.0292 | 0.3779 ± 0.0166 |
| semi_amortized | none L=1 | 3 | 0.3643 ± 0.0727 | 0.1984 ± 0.0678 | 0.4088 ± 0.0297 |
| semi_amortized | random_gaussian L=2 | 3 | 0.3602 ± 0.0698 | 0.2037 ± 0.0563 | 0.4110 ± 0.0292 |
| semi_amortized | repetition L=2 | 3 | 0.3397 ± 0.0618 | 0.1938 ± 0.0644 | 0.4054 ± 0.0382 |

Per-run metrics, missing-view details, metric definitions and the source/data/version manifest
are stored alongside this report. Checkpoints and training configs were written locally to experiments/reproducibility_cloud_review (ignored by Git); use the command below to recreate them.

Reproduce in fresh directories:

```bash
python run_reproducibility.py --dataset mfeat --epochs 20 --seeds 42 43 44 --device cpu --run-dir experiments/repro-repeat --output reports/repro-repeat
```

## Regression and integration validation

- 29 regression tests passed locally, including CUDA/BF16, GPU scheduling, and generated-script execution.
- Real 100Leaves and Caltech101-7 one-epoch training plus data-reload validation passed.
- Real Mfeat CUDA/BF16 one-epoch training plus strict checkpoint evaluation passed.
- Full real ALOI training was not run; image-shape and loader/cache tests use small fixtures.
- CPU CI configuration was added; check the pull request for current remote CI status.
- See `validation.json`, `feature_loader_smokes.json`, and `source_snapshot.tar.gz`.
- The smoke records retain their own source hashes; they preceded the final scheduling and reporting changes.
