# WGANO: Wavelet-Guided Attention Neural Operator

Official implementation of WGANO: Wavelet-Guided Attention Neural Operator for Multi-Physics Problems.

**Author:** Bo Fang  
**Affiliation:** China University of Petroleum (East China)

---

## Overview

WGANO introduces wavelet-guided attention mechanisms into neural operators for solving PDEs. C-WGANO extends this with physics-coupled decoders for multi-physics problems with bidirectional field dependencies.

**Key Features:**
- Wavelet-guided attention with high-frequency gating
- Physics-coupled decoder for multi-field interactions
- State-of-the-art results on 10 PDE benchmarks

---

## Installation

```bash
pip install torch torchvision numpy scipy matplotlib
pip install timm  # for trunc_normal_ initialization
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

---

## Project Structure

```
WGANO/
├── wgano/                      # Core models
│   ├── model.py                # WGANO for single-physics
│   └── coupled_model.py        # C-WGANO for multi-physics
├── utils/                      # Utilities
│   ├── normalizer.py           # Data normalization
│   ├── testloss.py             # Relative L2 loss
│   └── wavelets.py             # Wavelet transforms
├── experiments/                # Training scripts
│   ├── single_physics/         # 6 single-physics tasks
│   │   ├── darcy.py
│   │   ├── ns.py
│   │   ├── pipe.py
│   │   ├── airfoil.py
│   │   ├── elasticity.py
│   │   └── plasticity.py
│   └── multi_physics/          # 4 multi-physics tasks
│       ├── mhd.py
│       ├── eflow.py
│       ├── nsheat.py
│       └── va.py
├── figures/                    # Visualization results
└── README.md                   # This file
```

---

## Datasets

Download datasets from:
- **Single-physics:** [PDEBench](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
- **Multi-physics:** [Zenodo DOI: 10.5281/zenodo.24132237](https://doi.org/10.5281/zenodo.24132237)

**Data format:**
- Single-physics: `.mat` or `.npy` files
- Multi-physics: `.npy` files with shape `(128, 128, C)` where C varies by task

---

## Usage

### Single-Physics Tasks

**Darcy Flow:**
```bash
python experiments/single_physics/darcy.py \
  --data_path /path/to/darcy \
  --epochs 650 \
  --batch_size 8 \
  --save_name darcy_wgano
```

**Navier-Stokes:**
```bash
python experiments/single_physics/ns.py \
  --data_path /path/to/ns \
  --epochs 500 \
  --batch_size 2 \
  --save_name ns_wgano
```

**Pipe Flow:**
```bash
python experiments/single_physics/pipe.py \
  --data_path /path/to/pipe \
  --epochs 500 \
  --batch_size 4 \
  --save_name pipe_wgano
```

### Multi-Physics Tasks (C-WGANO)

**MHD (Magnetohydrodynamics):**
```bash
python experiments/multi_physics/mhd.py \
  --data_dir /path/to/dataset \
  --epochs 500 \
  --batch_size 4 \
  --coupling_iter 2 \
  --save_name mhd_cwgano
```

**Data format for multi-physics:**
```
dataset/
├── MHD/
│   ├── train/ (1.npy, 2.npy, ..., 1000.npy)
│   └── test/  (1.npy, 2.npy, ..., 200.npy)
├── E_flow/
├── NS_heat/
└── VA/
```

Each `.npy` file shape: `(128, 128, channels)`
- MHD: 6 channels (Br input, Jx/Jy/Jz + u_u/u_v outputs)
- E_flow: 4 channels (κ input, φ + u_u/u_v outputs)
- NS_heat: 4 channels (Q input, u_u/u_v + T outputs)
- VA: 13 channels (density input, 6 acoustic + 6 structural outputs, complex-valued)

### Evaluation

Add `--eval` flag to evaluate a trained model:
```bash
python experiments/single_physics/darcy.py \
  --data_path /path/to/darcy \
  --eval \
  --save_name darcy_wgano
```

---

## Model Architecture

### WGANO (Single-Physics)

```python
from wgano import WGANOModel

model = WGANOModel(
    space_dim=2,
    n_layers=4,
    n_hidden=64,
    n_head=4,
    mlp_ratio=4,
    fun_dim=1,
    out_dim=1,
    H=64,
    W=64
)
```

### C-WGANO (Multi-Physics)

```python
from wgano import CoupledWGANO

model = CoupledWGANO(
    n_layers=4,
    n_hidden=64,
    n_head=4,
    head_hidden=64,
    out_dims=[3, 2],      # field1: 3 channels, field2: 2 channels
    coupling_iter=2,       # K=2 coupling iterations
    H=128,
    W=128
)
```

---

## Hyperparameters

Optimal configurations used in experiments:

| Task       | n_hidden | n_layers | n_head | epochs | batch_size |
|------------|----------|----------|--------|--------|------------|
| Darcy      | 156      | 9        | 6      | 650    | 8          |
| NS         | 328      | 8        | 8      | 500    | 2          |
| Pipe       | 184      | 7        | 8      | 500    | 4          |
| Airfoil    | 320      | 10       | 8      | 800    | 4          |
| Elasticity | 144      | 10       | 8      | 500    | 8          |
| Plasticity | 240      | 8        | 8      | 500    | 8          |

Multi-physics (C-WGANO): n_hidden=64, n_layers=4, head_hidden=64, coupling_iter=2

---

## Citation

```bibtex
@misc{fang2026wgano,
  title={WGANO: Wavelet-Guided Attention Neural Operator for Multi-Physics Problems},
  author={Fang, Bo},
  year={2026},
  institution={China University of Petroleum (East China)}
}
```

---

## Figures

Visualization results are in `figures/`:
- `figure2.pdf` - WGANO architecture
- `figure3-11.pdf` - Qualitative comparisons on all 10 tasks

---

## License

MIT License

---

## Contact

For questions or issues:
- Email: s25070074@s.upc.edu.cn
- GitHub Issues: https://github.com/fangbo666/WGANO/issues
