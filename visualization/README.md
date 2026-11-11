# Visualization Scripts

This directory contains scripts for generating figures from the paper.

## Scripts

- `vis_framework.py` - Architecture visualization (Figure 2)
- `vis_darcy*.py` - Darcy flow results
- `vis_ns*.py` - Navier-Stokes results
- `vis_pipe*.py` - Pipe flow results
- `vis_airfoil*.py` - Airfoil results
- `vis_elasticity*.py` - Elasticity results
- `vis_plasticity*.py` - Plasticity results
- `vis_mhd*.py` - MHD results
- `vis_eflow*.py` - E-flow results
- `vis_nsheat*.py` - NS-heat results
- `vis_va*.py` - VA results

## Usage

These scripts require trained model checkpoints and test data.

Example:
```bash
python vis_darcy.py --checkpoint path/to/checkpoint.pt --data_path path/to/data
```

Refer to individual scripts for specific arguments.
