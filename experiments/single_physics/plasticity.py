"""
Plasticity Experiment

Training WGANO on plasticity dataset with temporal dynamics.
Model: n_hidden=240, n_layers=8, n_head=8
Data: Material deformation with 20 time steps
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.io as scio
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import WGANOModel
from utils import TestLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('WGANO Plasticity')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_hidden', type=int, default=240)
parser.add_argument('--n_layers', type=int, default=8)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_name', type=str, default='plasticity_wgano')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

s1, s2 = 101, 31
T = 20
Deformation = 4
ntrain, ntest = 900, 80


def random_collate(batch):
    """Per-sample random temporal permutation for augmentation."""
    x_list, y_list = [], []
    for x_sample, y_sample in batch:
        perm = torch.randperm(T)
        x_list.append(x_sample)
        y_list.append(y_sample[..., perm])
    return torch.stack(x_list), torch.stack(y_list)


def main():
    data = scio.loadmat(args.data_path)
    inp_raw = torch.tensor(data['input'], dtype=torch.float)
    out_raw = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)

    x_train = inp_raw[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, s1, s2, 1)
    x_test = inp_raw[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, s1, s2, 1)

    fx_mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    fx_std = x_train.std(dim=(0, 1, 2), keepdim=True) + 1e-8
    x_train = (x_train - fx_mean) / fx_std
    x_test = (x_test - fx_mean) / fx_std

    y_train = out_raw[:ntrain, :s1, :s2].reshape(ntrain, s1, s2, Deformation, T)
    y_test = out_raw[-ntest:, :s1, :s2].reshape(ntest, s1, s2, Deformation, T)

    gx = np.linspace(0, 1, s1)
    gy = np.linspace(0, 1, s2)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing='ij')
    pos = torch.stack([torch.tensor(grid_x, dtype=torch.float),
                       torch.tensor(grid_y, dtype=torch.float)], dim=-1).unsqueeze(0)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, collate_fn=random_collate)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    model = WGANOModel(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        fun_dim=1,
        out_dim=Deformation,
        use_qk_norm=False,
        unified_pos=False,
        ref=8,
        H=s1,
        W=s2
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    pos = pos.cuda()
    myloss = TestLoss(size_average=False)

    if args.eval:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name + '_best.pt')))
        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                B = x.shape[0]
                err_sample = 0.0
                for t in range(T):
                    pred, _, _ = model(pos.repeat(B, 1, 1, 1), x)
                    err_sample += myloss(pred.reshape(B, -1), y[..., t].reshape(B, -1)).item()
                rel_err += err_sample / T
        print(f'Test error: {rel_err / ntest:.6f}')
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_error', 'lr'])

    best_err = float('inf')
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            B = x.shape[0]
            loss_batch = 0.0
            for t in range(T):
                pred, _, _ = model(pos.repeat(B, 1, 1, 1), x)
                loss_batch += myloss(pred.reshape(B, -1), y[..., t].reshape(B, -1))
            loss = loss_batch / T
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                B = x.shape[0]
                err_sample = 0.0
                for t in range(T):
                    pred, _, _ = model(pos.repeat(B, 1, 1, 1), x)
                    err_sample += myloss(pred.reshape(B, -1), y[..., t].reshape(B, -1)).item()
                rel_err += err_sample / T
        rel_err /= ntest

        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {ep+1:03d}  train: {train_loss:.6f}  test: {rel_err:.6f}  '
              f'lr: {cur_lr:.2e}  time: {elapsed:.1f}s')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep+1, f'{train_loss:.6f}', f'{rel_err:.6f}', f'{cur_lr:.2e}'])

        if rel_err < best_err:
            best_err = rel_err
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name + '_best.pt'))

    print(f'Best test error: {best_err:.6f}')
    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name + '_final.pt'))


if __name__ == '__main__':
    main()
