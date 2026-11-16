"""ClaimNet model, training loop and metrics used by the Optuna tuning job.

Same definitions as Step 3 and Step 4 of 05-Model-Training.ipynb, so the network tuned here is the one
05 rebuilds in Step 4c from the best trial's settings.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

GROUPS = ['mean', 'max', 'min', 'sum', 'max_div_min', 'max_minus_min', 'first_div_last', 'first_minus_last']
EMB_DIM = 1536


def gini(y, p):
    """Gini = 2 * AUC - 1."""
    return 2 * roc_auc_score(y, p) - 1


def _block(i, o, p):
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(p))


class ClaimNet(nn.Module):
    """One encoder per running summary -> concat -> fusion -> MLP head -> 1 logit."""

    def __init__(self, group_dims, fuse_dim=256, head_dims=(128,), in_dim=EMB_DIM, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([_block(in_dim, d, dropout) for d in group_dims])
        self.fuse = _block(sum(group_dims), fuse_dim, dropout)
        layers, prev = [], fuse_dim
        for h in head_dims:
            layers.append(_block(prev, h, dropout))
            prev = h
        self.head = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x):                      # x: (B, 8, 1536)
        z = torch.cat([enc(x[:, i]) for i, enc in enumerate(self.encoders)], dim=1)
        return self.head(self.fuse(z)).squeeze(-1)


def sample_cfg(trial):
    """Optuna search space (same as 05, Step 4)."""
    return {'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.7),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 3e-2, log=True),
            'noise_std': trial.suggest_float('noise_std', 0.0, 0.05),
            'batch': trial.suggest_categorical('batch', [32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]),
            'fuse_dim': trial.suggest_categorical('fuse_dim', [128, 256, 384, 512]),
            'group_dims': [trial.suggest_categorical(f'dim_{g}', [64, 96, 128, 192, 256]) for g in GROUPS]}


@torch.no_grad()
def predict(net, X, device='cuda', bs=32768):
    net.eval()
    return np.concatenate([torch.sigmoid(net(torch.as_tensor(X[i:i + bs], device=device).float())).cpu().numpy()
                           for i in range(0, len(X), bs)])


def train_one(cfg, Xtr, ytr, Xva, yva, max_epochs=60, patience=8, device='cuda'):
    """BCEWithLogits (pos_weight = neg/pos), AdamW, cosine LR, early stop on validation Gini.

    Returns (best model, best validation Gini, best epoch counted from 1).
    """
    net = ClaimNet(cfg['group_dims'], cfg['fuse_dim'], dropout=cfg['dropout']).to(device)
    pos = float(ytr.sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ytr) - pos) / pos], device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best, best_state, best_epoch, since = -1, None, 0, 0
    for ep in range(max_epochs):
        net.train()
        for i in torch.randperm(len(ytr)).split(cfg['batch']):
            idx = np.sort(i.numpy())
            xb = torch.as_tensor(Xtr[idx], device=device).float()
            xb = xb + cfg['noise_std'] * torch.randn_like(xb)
            loss = crit(net(xb), torch.as_tensor(ytr[idx], device=device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        g = gini(yva, predict(net, Xva, device))
        if g > best:
            best, best_epoch, since = g, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, best_epoch


def last_snapshot_gini(y, p, cvg):
    """Gini on each coverage's latest snapshot (rows are in snapshot order within a coverage)."""
    last = pd.Series(np.arange(len(cvg))).groupby(cvg).last().to_numpy()
    return gini(y[last], p[last])
