"""
Reusable training loop, shared across Attorney/Litigation (binary) and the 2
Severity variants (5-class, same data/architecture, different PRIMARY metric):
  * severity-gini: selection uses support-weighted ordinal cumulative Gini
  * severity-qwk:  selection uses quadratic-weighted-kappa (QWK)
Both numbers (plus F1) are always computed and logged every epoch for every
split regardless of which one is primary -- `primary_metric` in cfg only
decides what early-stopping/Optuna-pruning/best-checkpoint selection watches.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader

from data import MemmapFeatures
from model import CutoffNet
import optuna

EVAL_BATCH = 16384


def binary_gini(y, p):
    return 2.0 * roc_auc_score(y, p) - 1.0


def ordinal_cumulative_gini(y, proba, num_classes):
    """Buckets are ORDERED (0<1<2<3<4), so instead of nominal one-vs-rest, use
    the standard ordinal "cumulative threshold" decomposition: for
    k=1..num_classes-1, collapse to the binary problem (bucket >= k) vs
    (bucket < k) -- e.g. for 5 buckets that's {4} vs {0,1,2,3}, {3,4} vs
    {0,1,2}, {2,3,4} vs {0,1}, {1,2,3,4} vs {0}. Each threshold's score is
    P(bucket >= k) = sum of proba[:, k:].

    SUPPORT-WEIGHTED average across thresholds (not plain mean): the extreme
    thresholds (e.g. {4} vs rest) have far fewer positives than the middle
    ones, so their Gini estimate is noisier -- weighting by min(n_pos,n_neg)
    lets the well-populated thresholds dominate instead of every threshold
    counting equally regardless of how reliable its estimate is."""
    ginis, weights = [], []
    for k in range(1, num_classes):
        yk = (y >= k).astype(np.float32)
        n_pos, n_neg = yk.sum(), len(yk) - yk.sum()
        if 0 < n_pos < len(yk):
            score_k = proba[:, k:].sum(axis=1)
            ginis.append(binary_gini(yk, score_k))
            weights.append(min(n_pos, n_neg))
    if not ginis:
        return float("nan")
    return float(np.average(ginis, weights=weights))


@torch.no_grad()
def predict(net, loader, device, is_multiclass):
    net.eval(); ys, ps = [], []
    for xb, yb in loader:
        logits = net(xb.to(device).float())
        if is_multiclass:
            p = torch.softmax(logits, dim=1).float().cpu().numpy()
        else:
            p = torch.sigmoid(logits).float().cpu().numpy()
        ps.append(p); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(net, loader, device, is_multiclass, num_classes=1):
    y, p = predict(net, loader, device, is_multiclass)
    out = {"n": int(len(y))}
    if not is_multiclass:
        out["pos_rate"] = float(y.mean())
        if 0 < y.sum() < len(y):
            out["gini"] = float(binary_gini(y, p))
        else:
            out["gini"] = float("nan")
        return out
    # multiclass (Severity)
    pred_cls = p.argmax(axis=1)
    out["gini"] = ordinal_cumulative_gini(y, p, num_classes)        # primary
    out["f1_macro"] = float(f1_score(y, pred_cls, average="macro", zero_division=0))
    out["qwk"] = float(cohen_kappa_score(y.astype(int), pred_cls, weights="quadratic"))
    return out


def make_loader(X, y, idx, batch, train, device):
    ds = MemmapFeatures(X, y, idx)
    nw = 4 if device == "cuda" else 0
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=nw,
                       pin_memory=(device == "cuda"), drop_last=False,
                       persistent_workers=(nw > 0))


def run_training(X, y, tr_idx, va_idx, ho_idx, cfg, device, groups, log=print, trial=None,
                  last_idx=None, num_classes=1):
    """Train one config. Logs ROW-level train/val/holdout(=test) metrics every epoch
    (gini always; f1/qwk too if multiclass), plus CVG-level (final-snapshot-only)
    metrics if `last_idx` given.

    `cfg["primary_metric"]` ("gini" or "qwk", default "gini") selects what
    early-stopping/Optuna-pruning/best-checkpoint watches on the VAL split --
    "qwk" is only valid for multiclass (Severity). Both gini and qwk are always
    computed and returned regardless of which one was primary.

    Returns dict(best_metric[val, selected by primary_metric], best_epoch,
    best_holdout_gini, best_holdout_qwk[multiclass only], best_cvg[dict],
    best_state, history)."""
    is_multiclass = num_classes > 1
    primary_metric = cfg.get("primary_metric", "gini")
    assert primary_metric in ("gini", "qwk")
    assert is_multiclass or primary_metric == "gini", "qwk is only defined for multiclass models"
    torch.manual_seed(cfg.get("seed", 42))
    dims = [cfg["group_dims"][g] for g in groups]
    net = CutoffNet(groups, dims, fuse_dim=cfg["fuse_dim"], dropout=cfg["dropout"],
                    num_classes=num_classes).to(device)

    tr_loader = make_loader(X, y, tr_idx, cfg["batch"], True, device)
    tr_eval = make_loader(X, y, tr_idx, EVAL_BATCH, False, device)
    va_loader = make_loader(X, y, va_idx, EVAL_BATCH, False, device)
    ho_loader = make_loader(X, y, ho_idx, EVAL_BATCH, False, device)
    cvg_loaders = {}
    if last_idx is not None:
        for k, idx in last_idx.items():
            cvg_loaders[k] = make_loader(X, y, idx, EVAL_BATCH, False, device)

    if is_multiclass:
        counts = np.array([max((y[tr_idx] == c).sum(), 1.0) for c in range(num_classes)])
        class_weight = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)
        crit = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
        pos = float(y[tr_idx].sum()); neg = float((y[tr_idx] == 0).sum())
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    max_epochs = cfg["max_epochs"]
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
             if cfg.get("cosine", True) else None)
    noise_std = float(cfg.get("noise_std", 0.0))

    best_metric, best_epoch, best_ho, best_ho_qwk, best_state, history = -2.0, -1, float("nan"), float("nan"), None, []
    best_cvg = {}
    patience = cfg["patience"]
    since_improved = 0
    for ep in range(1, max_epochs + 1):
        net.train(); tl = tn = 0
        for xb, yb in tr_loader:
            xb = xb.to(device).float()
            if noise_std > 0:
                xb = xb + noise_std * torch.randn_like(xb)
            yb = yb.to(device)
            if is_multiclass:
                yb = yb.long()
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
            tl += loss.item() * len(yb); tn += len(yb)
        if sched is not None:
            sched.step()

        tr_m = evaluate(net, tr_eval, device, is_multiclass, num_classes)
        va_m = evaluate(net, va_loader, device, is_multiclass, num_classes)
        ho_m = evaluate(net, ho_loader, device, is_multiclass, num_classes)
        vg = va_m[primary_metric]           # what selection/pruning/early-stop watches
        cvg_m = {k: evaluate(net, ld, device, is_multiclass, num_classes) for k, ld in cvg_loaders.items()}

        rec = {"epoch": ep, "train_loss": tl / tn, "primary_metric": primary_metric,
               "train_gini": tr_m["gini"], "val_gini": va_m["gini"], "holdout_gini": ho_m["gini"],
               "val_primary": vg}
        if is_multiclass:
            rec.update({"train_f1_macro": tr_m["f1_macro"], "val_f1_macro": va_m["f1_macro"],
                        "holdout_f1_macro": ho_m["f1_macro"], "train_qwk": tr_m["qwk"],
                        "val_qwk": va_m["qwk"], "holdout_qwk": ho_m["qwk"]})
        if cvg_m:
            rec.update({f"cvg_{k}_gini": v["gini"] for k, v in cvg_m.items()})
            if is_multiclass:
                rec.update({f"cvg_{k}_qwk": v["qwk"] for k, v in cvg_m.items()})
        history.append(rec)

        msg = f"  ep{ep:>3} loss={tl/tn:.4f} | GINI tr={tr_m['gini']:.4f} va={va_m['gini']:.4f} ho={ho_m['gini']:.4f}"
        if is_multiclass:
            msg += f" | QWK va={va_m['qwk']:.4f} | F1 va={va_m['f1_macro']:.4f}"
        msg += f" | [selecting on {primary_metric}={vg:.4f}]"
        log(msg)

        if trial is not None:
            trial.report(vg, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if vg > best_metric + 1e-5:
            best_metric, best_epoch, best_ho, best_ho_qwk = vg, ep, ho_m["gini"], ho_m.get("qwk", float("nan"))
            best_cvg = dict(cvg_m)
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                log(f"  early stop at ep{ep} (no val gain for {patience}); "
                    f"best ep{best_epoch} val_{primary_metric}={best_metric:.4f} holdout_gini={best_ho:.4f}")
                break

    return {"best_metric": best_metric, "primary_metric": primary_metric, "best_epoch": best_epoch,
            "best_holdout_gini": best_ho, "best_holdout_qwk": best_ho_qwk,
            "best_cvg": best_cvg, "best_state": best_state, "history": history}
