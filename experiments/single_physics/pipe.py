"""
Pipe Flow Experiment

Training WGANO on pipe flow dataset.
Model: n_hidden=184, n_layers=7, n_head=8
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import WGANOModel
from utils import UnitTransformer, TestLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('WGANO Pipe Flow')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_hidden', type=int, default=184)
parser.add_argument('--n_layers', type=int, default=7)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_name', type=str, default='pipe_wgano')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
ntrain, ntest = 1000, 200


def main():
    inputX = torch.tensor(np.load(os.path.join(args.data_path, 'Pipe_X.npy')), dtype=torch.float)
    inputY = torch.tensor(np.load(os.path.join(args.data_path, 'Pipe_Y.npy')), dtype=torch.float)
    inp = torch.stack([inputX, inputY], dim=-1)
    output = torch.tensor(np.load(os.path.join(args.data_path, 'Pipe_Q.npy'))[:, 0], dtype=torch.float)

    s1, s2 = 129, 129
    x_train = inp[:ntrain, :s1, :s2]
    y_train = output[:ntrain, :s1, :s2]
    x_test = inp[-ntest:, :s1, :s2]
    y_test = output[-ntest:, :s1, :s2]

    x_normalizer = UnitTransformer(x_train.reshape(ntrain, -1, 2))
    y_normalizer = UnitTransformer(y_train.reshape(ntrain, -1))
    x_train = x_normalizer.encode(x_train.reshape(ntrain, -1, 2)).reshape(ntrain, s1, s2, 2)
    x_test = x_normalizer.encode(x_test.reshape(ntest, -1, 2)).reshape(ntest, s1, s2, 2)
    y_train_enc = y_normalizer.encode(y_train.reshape(ntrain, -1)).reshape(ntrain, s1, s2, 1)
    y_test = y_test.reshape(ntest, s1, s2, 1)

    x_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train_enc),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    model = WGANOModel(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        use_qk_norm=False,
        H=s1,
        W=s2
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    if args.eval:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name + '_best.pt')))
        model.eval()
        myloss = TestLoss(size_average=False)
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                pred, _, _ = model(x, fx=None)
                pred = y_normalizer.decode(pred.squeeze(-1))
                rel_err += myloss(pred.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1)).item()
        print(f'Test error: {rel_err / ntest:.6f}')
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

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
            pred, _, _ = model(x, fx=None)
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
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                pred, _, _ = model(x, fx=None)
                pred = y_normalizer.decode(pred.squeeze(-1))
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
