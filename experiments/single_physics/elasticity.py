"""
Elasticity Experiment

Training WGANO on irregular mesh elasticity dataset.
Model: n_hidden=144, n_layers=10, n_head=8

NOTE: This task uses irregular mesh (972 points) and requires 1D wavelet architecture.
The full implementation requires custom 1D wavelet attention layers.
See the paper supplementary materials for architectural details.

This script provides the data loading pattern. For the complete 1D architecture,
refer to the original codebase or contact the authors.
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import WGANOModel
from utils import TestLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('WGANO Elasticity')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_hidden', type=int, default=144)
parser.add_argument('--n_layers', type=int, default=10)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_name', type=str, default='elasticity_wgano')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

ntrain, ntest = 1000, 200


def main():
    print("Loading elasticity data...")
    PATH_Sigma = os.path.join(args.data_path, 'Random_UnitCell_sigma_10.npy')
    PATH_XY = os.path.join(args.data_path, 'Random_UnitCell_XY_10.npy')

    sigma = torch.from_numpy(np.load(PATH_Sigma)).float()
    xy = torch.from_numpy(np.load(PATH_XY)).float()

    x_train = sigma[:ntrain]
    x_test = sigma[ntrain:ntrain+ntest]
    y_train = xy[:ntrain]
    y_test = xy[ntrain:ntrain+ntest]

    print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Mesh points: {x_train.shape[1]}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    print("\nNOTE: Elasticity requires 1D wavelet architecture for irregular mesh.")
    print("This script provides the data loading pattern.")
    print("For the complete implementation with 1D wavelet layers,")
    print("refer to the paper supplementary materials.\n")

    print(f"Expected model: n_hidden={args.n_hidden}, n_layers={args.n_layers}")
    print(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("\nTo run this experiment, implement the 1D wavelet attention architecture")
    print("as described in Section 3.3 of the paper (irregular mesh variant).")


if __name__ == '__main__':
    main()
