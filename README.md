# Coded Next-Gen Multi-View GPLVM (CNG-MV-GPLVM)

This repository implements the **Coded Next-Gen Multi-View GP-LVM**, a research framework aligning with **Yang et al. (2025)** while introducing structural regularization through **Linear Error Correcting Codes (ECC)** and robust amortized inference via **Product-of-Experts (PoE)**.

## 🚀 Workflow: Experiment-Centric Design

We use an on-demand experiment generation workflow to keep the repository clean and self-contained.

### 1. Create an Experiment
Use `create_experiment.py` to generate a dedicated folder for a specific run.

```bash
# Example: Create a Semi-Amortized experiment on 100Leaves
python create_experiment.py \
    --name "semi_mfeat_Z32_L10" \
    --dataset "mfeat" \
    --inference "semi_amortized" \
    --z_dim 32 \
    --redundancy 10 \
    --ecc_type "random_gaussian" \
    --epochs 500
```

This creates: `experiments/YYYYMonDD_HH-MM-SS_semi_mfeat_Z32_L10/`

### 2. Run the Experiment
Navigate to the generated folder and execute the `run.sh` script.

```bash
cd experiments/2026Jan27_..._semi_mfeat_Z32_L10
bash run.sh
```

### 3. Folder Structure
Each experiment is self-contained:
- `config.yaml`: Full hyperparameter set.
- `run.sh`: Automated execution script (handles PYTHONPATH and env vars).
- `logs/`: TensorBoard logs.
- `checkpoints/`: Model weights (`final_model.pth`).
- `README.md`: Metadata and description for tracking.

---

## 🏗️ Architecture Key Features

### 1. Next-Gen Spectral Mixture (NG-SM) Kernel
Aligned with Yang (2025), implementing a Bivariate Gaussian spectral density with dynamic RFF sampling:
- **Two-step reparameterization** for sampling frequencies ($\omega_1, \omega_2$).
- **Woodbury Identity** for efficient GP Marginal Likelihood calculation.

### 2. Product-of-Experts (PoE) Encoder
For `amortized` and `semi_amortized` modes, we use a PoE architecture:
- Independent MLP encoders per view.
- Joint posterior fusion: $q(z|Y) \propto p(z) \prod_v q(z|y^{(v)})$.
- Natively handles missing views and integrates information efficiently.

### 3. Linear Error Correcting Codes (ECC)
Regularizes the latent space by projecting $Z \in \mathbb{R}^{d_z}$ to $X \in \mathbb{R}^{L \cdot d_z}$ using structured matrices (Repetition or Random Gaussian).

---

## 📊 Evaluation
Use the root level `quick_eval.py` for standard clustering metrics (NMI, ACC) and Latent Shift analysis.

```bash
python quick_eval.py
```

## 📂 Repository Layout
- `src/models/`: Architecture definitions (CNG, Kernels, Encoders).
- `src/modules/`: ECC and specialized neural modules.
- `src/trainer/`: Training engine.
- `src/utils/`: Data loaders and utility functions.
- `experiments/`: Root for all generated runs.
- `results/`: Processed analysis and figures.
