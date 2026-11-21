"""
Shared architecture for the 3 cutoff-snapshot models (Attorney, Litigation,
Severity). Per-group encoder -> fusion -> MLP head, with a configurable number
of output logits (`num_classes`): 1 logit + BCEWithLogits for Attorney/Litigation, 5 logits +
CrossEntropy for Severity (SEV_BUCKET 0-4).
"""
import torch
import torch.nn as nn


def _block(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class CutoffNet(nn.Module):
    def __init__(self, group_names, group_dims, fuse_dim=256, head_dims=(128,),
                 in_dim=1536, dropout=0.1, num_classes=1):
        super().__init__()
        assert len(group_names) == len(group_dims), "one dim per group"
        self.group_names = list(group_names)
        self.group_dims = list(group_dims)
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.encoders = nn.ModuleList([_block(in_dim, d, dropout) for d in group_dims])
        self.fuse = _block(sum(group_dims), fuse_dim, dropout)
        layers, prev = [], fuse_dim
        for h in head_dims:
            layers.append(_block(prev, h, dropout)); prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, G, in_dim) float32
        codes = [enc(x[:, i, :]) for i, enc in enumerate(self.encoders)]
        z = torch.cat(codes, dim=1)
        z = self.fuse(z)
        out = self.head(z)
        return out.squeeze(-1) if self.num_classes == 1 else out   # (B,) or (B,num_classes)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
