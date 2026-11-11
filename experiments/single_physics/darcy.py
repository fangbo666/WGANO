"""
Darcy Flow Experiment

Training WGANO on Darcy flow dataset.
Model: n_hidden=156, n_layers=9, n_head=6
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
from utils import UnitTransformer, TestLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('WGANO Darcy Flow')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=650)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_hidden', type=int, default=156)
parser.add_argument('--n_layers', type=int, default=9)
parser.add_argument('--n_head', type=int, default=6)
parser.add_argument('--mlp_ratio', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=3e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--ntest', type=int, default=200)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_name', type=str, default='darcy_wgano')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

r = args.downsample
s = int(((421 - 1) / r) + 1)

train_path = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth1.mat')
test_path = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth2.mat')


def load_data():
    td = scio.loadmat(train_path)
    x_tr = torch.from_numpy(td['coeff'][:args.ntrain, ::r, ::r][:, :s, :s]).float()
    y_tr = torch.from_numpy(td['sol'][:args.ntrain, ::r, ::r][:, :s, :s]).float()

    if os.path.exists(test_path):
        vd = scio.loadmat(test_path)
        x_te = torch.from_numpy(vd['coeff'][:args.ntest, ::r, ::r][:, :s, :s]).float()
        y_te = torch.from_numpy(vd['sol'][:args.ntest, ::r, ::r][:, :s, :s]).float()
    else:
        x_te = torch.from_numpy(td['coeff'][args.ntrain:args.ntrain+args.ntest, ::r, ::r][:, :s, :s]).float()
        y_te = torch.from_numpy(td['sol'][args.ntrain:args.ntrain+args.ntest, ::r, ::r][:, :s, :s]).float()

    return x_tr, y_tr, x_te, y_te


def main():
    x_tr, y_tr, x_te, y_te = load_data()
    ntrain, ntest = args.ntrain, args.ntest

    x_normalizer = UnitTransformer(x_tr.reshape(ntrain, -1))
    y_normalizer = UnitTransformer(y_tr.reshape(ntrain, -1))
    x_tr = x_normalizer.encode(x_tr.reshape(ntrain, -1)).reshape(ntrain, s, s, 1)
    x_te = x_normalizer.encode(x_te.reshape(ntest, -1)).reshape(ntest, s, s, 1)
    y_tr = y_normalizer.encode(y_tr.reshape(ntrain, -1)).reshape(ntrain, s, s, 1)
    y_te = y_te.reshape(ntest, s, s, 1)

    x_normalizer.cuda()
    y_normalizer.cuda()

    gx, gy = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing='ij')
    pos = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos.repeat(ntrain, 1, 1, 1), x_tr, y_tr),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos.repeat(ntest, 1, 1, 1), x_te, y_te),
        batch_size=args.batch_size, shuffle=False)

    model = WGANOModel(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        fun_dim=1,
        out_dim=1,
        unified_pos=True,
        ref=8,
        H=s,
        W=s
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    if args.eval:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name + '_best.pt')))
        model.eval()
        myloss = TestLoss(size_average=False)
        rel_err = 0.0
        with torch.no_grad():
            for pos_batch, x, y in test_loader:
                pos_batch, x, y = pos_batch.cuda(), x.cuda(), y.cuda()
                pred, _, _ = model(pos_batch, x)
                pred = y_normalizer.decode(pred.reshape(x.shape[0], -1)).reshape(x.shape[0], s, s, 1)
                rel_err += myloss(pred.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1)).item()
        print(f'Test error: {rel_err / ntest:.6f}')
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.25)
    myloss = TestLoss(size_average=False)

    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_error', 'lr'])

    best_err = float('inf')
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0

        for pos_batch, x, y in train_loader:
            pos_batch, x, y = pos_batch.cuda(), x.cuda(), y.cuda()
            pred, _, _ = model(pos_batch, x)
            loss = myloss(pred.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
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
            for pos_batch, x, y in test_loader:
                pos_batch, x, y = pos_batch.cuda(), x.cuda(), y.cuda()
                pred, _, _ = model(pos_batch, x)
                pred = y_normalizer.decode(pred.reshape(x.shape[0], -1)).reshape(x.shape[0], s, s, 1)
                rel_err += myloss(pred.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1)).item()
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
