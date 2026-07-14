#!/usr/bin/env python3
"""
train_example.py -- minimal, single-file training example for the
cell-type-resolved sequence-to-expression (S2E) model used in the CASCADE
paper (Farghadan, Schmitz, Jackson & Pickering).

This is a stripped-down illustration of the training procedure described in
the paper's Methods, NOT the full production pipeline (no distributed
training, cross-validation, learning-rate schedules, or checkpoint
resumption). It exists so a reader can see, in one file, how a frozen
per-base GPN embedding is turned into per-cell-type expression predictions.

Pipeline recap (see the paper for full detail):
  1. A soybean-adapted GPN (Genomic Pre-trained Network) backbone is
     fine-tuned by masked-language modeling on Glycine max sequence, then
     frozen. It is NOT part of this script -- see https://github.com/songlab-cal/gpn
     and the paper's Methods for the fine-tuning recipe.
  2. For each gene, the frozen backbone embeds a fixed window around the
     transcription start site (TSS +/- 2,000 bp in the paper) at
     single-nucleotide resolution, giving a precomputed embedding
     X_g in R^{L x D} (L = 4,000, D = 512 in the paper).
  3. This script trains a convolutional decoder that maps X_g to one
     predicted expression value per cell type, for all cell types at once
     ("multi-task" training).

Expected inputs
---------------
--embeddings_file : a single safetensors or .pt file holding a mapping
                     {gene_id: FloatTensor(L, D)} for every gene in
                     --expression_path. All L must match (pad/window
                     upstream if needed).
--expression_path  : a CSV with genes as rows (first column = gene_id)
                     and cell types as columns (already log1p(CPM) or
                     otherwise normalized).

Usage
-----
    python train_example.py \\
        --embeddings_file /path/to/embeddings.safetensors \\
        --expression_path /path/to/expression.csv \\
        --output_dir ./example_run

Requirements: torch, numpy, pandas, safetensors
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# Model: frozen-embedding -> per-cell-type expression
# --------------------------------------------------------------------------- #

class SparseCNNEncoder(nn.Module):
    """(B, L, D) -> (B, d_model // 2) gene-level encoder."""

    def __init__(self, input_dim, d_model, dropout=0.3, conv1_kernel=3, conv2_kernel=5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=conv1_kernel, padding=conv1_kernel // 2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=conv2_kernel, padding=conv2_kernel // 2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.bottleneck = nn.Linear(d_model, d_model // 2)

    def forward(self, x):
        x = x.transpose(1, 2)                     # (B, D, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)                        # dropout before pooling
        x = self.global_pool(x).squeeze(-1)        # (B, d_model)
        x = self.layer_norm(x)
        return self.bottleneck(x)                  # (B, d_model // 2)


class PromoterCellTypeModel(nn.Module):
    """Sequence embedding -> per-cell-type expression vector (B, num_cell_types)."""

    def __init__(self, emb_dim, num_cell_types, d_model=768, hidden_dim=1024,
                 encoder_dropout=0.3, decoder_dropout=0.3, conv1_kernel=3, conv2_kernel=5):
        super().__init__()
        self.encoder = SparseCNNEncoder(emb_dim, d_model, encoder_dropout, conv1_kernel, conv2_kernel)
        self.latent_proj = nn.Linear(d_model // 2, d_model // 4)
        self.decoder = nn.Sequential(
            nn.Linear(d_model // 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(hidden_dim, num_cell_types),
        )

    def forward(self, x):
        latent = self.latent_proj(self.encoder(x))     # (B, d_model // 4)
        return self.decoder(latent)                     # (B, num_cell_types)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class GeneExpressionDataset(Dataset):
    """One sample per gene; target is its full vector of per-cell-type expression."""

    def __init__(self, gene_ids, embeddings, expr):
        self.gene_ids = gene_ids
        self.embeddings = embeddings
        self.expr = expr  # numpy array (num_genes, num_cell_types), aligned to gene_ids

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gene_id = self.gene_ids[idx]
        x = self.embeddings[gene_id]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        y = torch.as_tensor(self.expr[idx], dtype=torch.float32)
        return x.float(), y


def load_embeddings(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def per_cell_pearson(pred, true):
    """Mean across-gene Pearson correlation, one value per cell type, then averaged."""
    pred = pred.numpy()
    true = true.numpy()
    corrs = []
    for c in range(pred.shape[1]):
        if np.std(true[:, c]) == 0 or np.std(pred[:, c]) == 0:
            continue
        corrs.append(np.corrcoef(pred[:, c], true[:, c])[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())
    pcc = per_cell_pearson(torch.cat(all_pred), torch.cat(all_true))
    return total_loss / n, pcc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings_file", required=True, help="safetensors or .pt file: {gene_id: Tensor(L, D)}")
    p.add_argument("--expression_path", required=True, help="CSV, genes as rows (col 0 = gene_id), cell types as columns")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction of genes held out for validation.")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--encoder_dropout", type=float, default=0.3)
    p.add_argument("--decoder_dropout", type=float, default=0.3)
    p.add_argument("--conv1_kernel", type=int, default=3)
    p.add_argument("--conv2_kernel", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = load_embeddings(args.embeddings_file)
    expr_df = pd.read_csv(args.expression_path, index_col=0)

    gene_ids = [g for g in expr_df.index if g in embeddings]
    dropped = len(expr_df) - len(gene_ids)
    if dropped:
        print(f"Warning: {dropped} genes in {args.expression_path} have no matching embedding and are skipped.")
    expr = expr_df.loc[gene_ids].to_numpy(dtype=np.float32)
    emb_dim = embeddings[gene_ids[0]].shape[-1]
    num_cell_types = expr.shape[1]

    rng = random.Random(args.seed)
    shuffled = gene_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_ids, train_ids = shuffled[:n_val], shuffled[n_val:]
    id_to_row = {g: i for i, g in enumerate(gene_ids)}

    train_ds = GeneExpressionDataset(train_ids, embeddings, expr[[id_to_row[g] for g in train_ids]])
    val_ds = GeneExpressionDataset(val_ids, embeddings, expr[[id_to_row[g] for g in val_ids]])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"genes: {len(gene_ids)} total -> {len(train_ids)} train / {len(val_ids)} val, "
          f"cell types: {num_cell_types}, embedding dim: {emb_dim}")

    model = PromoterCellTypeModel(
        emb_dim=emb_dim, num_cell_types=num_cell_types, d_model=args.d_model,
        hidden_dim=args.hidden_dim, encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout, conv1_kernel=args.conv1_kernel,
        conv2_kernel=args.conv2_kernel,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pcc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_pcc = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"epoch {epoch:4d}  train_loss={train_loss:.4f} train_pcc={train_pcc:.3f}  "
              f"val_loss={val_loss:.4f} val_pcc={val_pcc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
                break

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
