"""
VA (Vibroacoustics) Experiment

Training C-WGANO on coupled acoustic-structural system.
Physics coupling: Structural vibration and acoustic radiation via fluid-structure interaction.

Data format:
  Input:  Water density distribution
  Output: Acoustic pressure (6 complex channels) + Structural stress (6 complex channels)
  Note: Complex-valued fields stored as [real, imag] pairs
"""
import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import CoupledWGANO

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('C-WGANO VA')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_hidden', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--head_hidden', type=int, default=64)
parser.add_argument('--coupling_iter', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_name', type=str, default='va_cwgano')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = 'cuda'


def load_split(data_dir, task, split):
    folder = os.path.join(data_dir, task, split)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    return np.stack([np.load(os.path.join(folder, f)) for f in files])


def nrmse_per_channel(pred, target):
    B, C = pred.shape[0], pred.shape[-1]
    total = 0.0
    for c in range(C):
        p = pred[..., c].reshape(B, -1)
        t = target[..., c].reshape(B, -1)
        nrmse = (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).mean()
        total += nrmse
    return total / C


def main():
    print('Loading data...')
    train_data = load_split(args.data_dir, 'VA', 'train')
    test_data = load_split(args.data_dir, 'VA', 'test')
    ntrain, ntest = len(train_data), len(test_data)

    x_train = torch.from_numpy(train_data[..., 0:1]).float()
    ac_train = torch.from_numpy(train_data[..., 1:7]).float()
    st_train = torch.from_numpy(train_data[..., 7:13]).float()
    x_test = torch.from_numpy(test_data[..., 0:1]).float()
    ac_test = torch.from_numpy(test_data[..., 1:7]).float()
    st_test = torch.from_numpy(test_data[..., 7:13]).float()

    train_loader = DataLoader(
        TensorDataset(x_train, ac_train, st_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(x_test, ac_test, st_test),
        batch_size=args.batch_size, shuffle=False)
    print(f'Train={ntrain}  Test={ntest}')
    print('Note: Acoustic and structural fields are complex-valued (6 channels each: 3 real + 3 imag)')

    model = CoupledWGANO(
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        head_hidden=args.head_hidden,
        out_dims=[6, 6],
        coupling_iter=args.coupling_iter,
        H=128,
        W=128
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    if args.eval:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name + '_best.pt')))
        model.eval()
        ac_err, st_err = 0.0, 0.0
        with torch.no_grad():
            for x, ac, st in test_loader:
                x, ac, st = x.to(device), ac.to(device), st.to(device)
                pred_ac, pred_st = model(x)
                ac_err += nrmse_per_channel(pred_ac, ac).item() * x.shape[0]
                st_err += nrmse_per_channel(pred_st, st).item() * x.shape[0]
        print(f'Acoustic error: {ac_err / ntest:.6f}')
        print(f'Structural error: {st_err / ntest:.6f}')
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)

    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_acoustic', 'test_structural', 'lr'])

    config = dict(
        arch='C-WGANO', task='VA',
        n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_head,
        head_hidden=args.head_hidden, coupling_iter=args.coupling_iter,
        total_params=total_params, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay
    )
    with open(os.path.join(args.save_dir, args.save_name + '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_sum = float('inf')
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, ac, st in train_loader:
            x, ac, st = x.to(device), ac.to(device), st.to(device)
            pred_ac, pred_st = model(x)
            loss = nrmse_per_channel(pred_ac, ac) + nrmse_per_channel(pred_st, st)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        ac_err, st_err = 0.0, 0.0
        with torch.no_grad():
            for x, ac, st in test_loader:
                x, ac, st = x.to(device), ac.to(device), st.to(device)
                pred_ac, pred_st = model(x)
                ac_err += nrmse_per_channel(pred_ac, ac).item() * x.shape[0]
                st_err += nrmse_per_channel(pred_st, st).item() * x.shape[0]
        ac_err /= ntest
        st_err /= ntest

        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {ep+1:03d}  train: {train_loss:.6f}  '
              f'Acoustic: {ac_err:.6f}  Structural: {st_err:.6f}  '
              f'lr: {cur_lr:.2e}  time: {elapsed:.1f}s')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep+1, f'{train_loss:.6f}',
                                   f'{ac_err:.6f}', f'{st_err:.6f}', f'{cur_lr:.2e}'])

        error_sum = ac_err + st_err
        if error_sum < best_sum:
            best_sum = error_sum
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name + '_best.pt'))

    print(f'Best combined error: {best_sum:.6f}')
    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name + '_final.pt'))


if __name__ == '__main__':
    main()
