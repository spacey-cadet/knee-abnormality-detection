"""KneeNet: multi-series knee MRI classifier with per-label attention.

input  x        [B, S, D, H, W] uint8/float (0-255)     S = series slots (plane x K)
       present  [B, S] 1 if the slot has a series
       plane    [B, S] 0/1/2     fluid [B, S] 0/1     fat [B, S] 0/1
output logits   [B, 12]

head="tokens" (v6): no per-series pooling. Every 2.5D window becomes a token (+ plane/fluid/fat + slice-index
embeddings), one transformer layer mixes all S*D tokens, and each label query cross-attends to the windows it needs.
Same encoder cost as head="series" (the default, identical to v5).
Pipeline: 2.5D slices (3 neighbouring slices as RGB) -> shared CNN -> attention pooling over slices ->
series token (+ plane/fluid/fat embeddings) -> transformer over series -> 12 learned label queries that
cross-attend to the series tokens (each label can pick the planes it needs) -> one logit per label.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LABELS


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x):                       # x [N, D, C]
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(1)


class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=False, weights_path=None, d_model=256,
                 n_layers=2, n_heads=4, dropout=0.2, drop_path=0.1, series_dropout=0.2, n_labels=len(LABELS),
                 grad_ckpt=False, train_slices=None, head="series", n_pos=32):
        super().__init__()
        assert head in ("series", "tokens"), head
        self.head, self.n_pos = head, n_pos
        if head == "tokens":
            n_layers = 1                     # attention is over every window token (S*D), so one layer is enough
        import timm
        kw = dict(pretrained=pretrained and weights_path is None, num_classes=0, global_pool="avg")
        if drop_path and any(k in backbone for k in ("efficientnet", "convnext", "vit_", "deit_", "coatnet")):
            kw["drop_path_rate"] = drop_path
        # ViT backbones (DINO / DINOv2 / DeiT): fixed patch grid. Input stays 160 px in the cache; for patch-14 models
        # it is resized to 168 (12x12 tokens) inside the model, so train / infer / submission need no extra flag.
        self.vit_size = None
        if backbone.startswith(("vit_", "deit_")):
            self.vit_size = 168 if "patch14" in backbone else 160
            kw["img_size"] = self.vit_size
            kw["fc_norm"] = False          # keep the final norm BEFORE pooling: timm's avg-pool default renames it to fc_norm and the pretrained norm.* keys no longer load
        elif backbone.startswith("coatnet"):
            # CoAtNet (timm MaxxVit): window attention needs the grid fixed at construction. The cache is 160 px, so build for 160
            # (5x5 windows at the last stage; rel-pos tables are resized by timm when pretrained weights load).
            self.vit_size = 160
            kw["img_size"] = self.vit_size
        self.enc = timm.create_model(backbone, **kw)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            self.enc.load_state_dict(sd, strict=False)
        if grad_ckpt and hasattr(self.enc, "set_grad_checkpointing"):
            self.enc.set_grad_checkpointing(True)
        c = self.enc.num_features
        self.proj = nn.Linear(c, d_model)
        self.pool = AttnPool(d_model)
        self.e_pos = nn.Embedding(n_pos, d_model) if head == "tokens" else None     # original slice index 0..D-1
        self.e_plane, self.e_fluid, self.e_fat = nn.Embedding(3, d_model), nn.Embedding(2, d_model), nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(layer, n_layers)
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head_w = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_labels))
        self.series_dropout, self.train_slices = series_dropout, train_slices
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def _enc(self, tri):
        if self.vit_size and tri.shape[-1] != self.vit_size:
            tri = torch.nn.functional.interpolate(tri, size=(self.vit_size, self.vit_size), mode="bilinear", align_corners=False)
        return self.enc(tri)

    def encode_tokens(self, x):
        """x [N, D, H, W] -> window tokens [N, D', d_model] and their original slice indices [D'] (no pooling)."""
        N, D, H, W = x.shape
        idx = torch.arange(D, device=x.device)
        if self.training and self.train_slices and self.train_slices < D:
            idx = torch.sort(torch.randperm(D, device=x.device)[:self.train_slices]).values
            x = x[:, idx]
            D = x.shape[1]
        xp = torch.cat([x[:, :1], x, x[:, -1:]], 1)
        tri = torch.stack([xp[:, :-2], xp[:, 1:-1], xp[:, 2:]], 2).reshape(N * D, 3, H, W)
        tri = (tri - self.mean) / self.std
        return self.proj(self._enc(tri)).view(N, D, -1), idx

    def encode_series(self, x):
        """x [N, D, H, W] -> [N, d_model] (attention pooled over slices, 2.5D input)."""
        N, D, H, W = x.shape
        if self.training and self.train_slices and self.train_slices < D:
            idx = torch.sort(torch.randperm(D, device=x.device)[:self.train_slices]).values
            x = x[:, idx]
            D = x.shape[1]
        xp = torch.cat([x[:, :1], x, x[:, -1:]], 1)            # replicate-pad the slice axis
        tri = torch.stack([xp[:, :-2], xp[:, 1:-1], xp[:, 2:]], 2)        # [N, D, 3, H, W]
        tri = tri.reshape(N * D, 3, H, W)
        tri = (tri - self.mean) / self.std
        f = self.proj(self._enc(tri)).view(N, D, -1)
        return self.pool(f)

    def forward(self, x, present, plane, fluid, fat):
        B, S, D, H, W = x.shape
        x = x.float()
        present = present.bool().clone()
        if self.training and self.series_dropout > 0:
            drop = torch.rand(B, S, device=x.device) < self.series_dropout
            keep = present & ~drop
            none = keep.sum(1) == 0
            keep[none] = present[none]                       # never drop everything
            present = keep
        none = present.sum(1) == 0
        present[none, 0] = True                              # empty study: use slot 0 (zeros) to avoid NaN
        if self.head == "tokens":
            f, idx = self.encode_tokens(x.view(B * S, D, H, W))             # [B*S, D', d]
            Dp, d = f.shape[1], f.shape[2]
            f = f.view(B, S, Dp, d)
            ser = self.e_plane(plane.long()) + self.e_fluid(fluid.long()) + self.e_fat(fat.long())    # [B, S, d]
            tok = (f + ser.unsqueeze(2) + self.e_pos(idx).view(1, 1, Dp, d)).reshape(B, S * Dp, d)
            pad = (~present).unsqueeze(2).expand(B, S, Dp).reshape(B, S * Dp)
            tok = self.fuse(tok, src_key_padding_mask=pad)
        else:
            tok = self.encode_series(x.view(B * S, D, H, W)).view(B, S, -1)
            tok = tok + self.e_plane(plane.long()) + self.e_fluid(fluid.long()) + self.e_fat(fat.long())
            pad = ~present
            tok = self.fuse(tok, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        att, _ = self.xattn(q, tok, tok, key_padding_mask=pad)
        z = self.drop(self.norm(att + q))
        return (z * self.head_w).sum(-1) + self.head_b        # [B, 12]


class SoftBCE(nn.Module):
    """Weighted BCE with soft targets, label smoothing and optional per-label pos_weight."""

    def __init__(self, pos_weight=None, smoothing=0.05):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(len(LABELS)))
        self.s = smoothing

    def forward(self, logits, y, w):
        y = y * (1 - self.s) + 0.5 * self.s
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1e-6)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow
