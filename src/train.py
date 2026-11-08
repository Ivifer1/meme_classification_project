"""Paso 2: entrena la red de fusión (y los dos modelos de una sola modalidad)
sobre los embeddings precalculados.

- Separa un 10 % de train.jsonl como validación (estratificado)
- BCEWithLogitsLoss con pos_weight para compensar que hay menos memes de odio
- AdamW con weight decay, dropout y early stopping según el ROC-AUC de validación
- Guarda el mejor modelo de cada modo en models/<modo>.pth

Uso:  python -m src.train            (entrena imagen, texto y fusion)
      python -m src.train --modes fusion
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import src.config as config
from src.dataset import load_feature_split, make_tensor_dataset
from src.models.fusion_mlp import FusionMLP


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for v, t, y in loader:
        logits = model(v.to(device), t.to(device))
        probs.append(torch.sigmoid(logits).squeeze(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def split_indices(labels):
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=config.VAL_FRACTION,
                            stratify=labels, random_state=config.SEED)


def train_mode(mode, feats, train_idx, val_idx, device):
    set_seed(config.SEED)
    train_loader = DataLoader(make_tensor_dataset(feats, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_tensor_dataset(feats, val_idx), batch_size=256)

    y_train = feats["labels"][train_idx]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = FusionMLP(mode).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_auc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total = 0.0
        for v, t, y in train_loader:
            v, t, y = v.to(device), t.to(device), y.unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(v, t), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_loader.dataset)

        val_probs, val_labels = predict_proba(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_auc": round(val_auc, 4)})
        print(f"[{mode}] época {epoch:02d} | loss {train_loss:.4f} | ROC-AUC val {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print(f"[{mode}] early stopping en la época {epoch}")
                break

    # Umbral de decisión: el que maximiza F1 en validación (no se toca el set de evaluación)
    model.load_state_dict(best_state)
    val_probs, val_labels = predict_proba(model, val_loader, device)
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_at(val_labels, val_probs, th) for th in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save({"mode": mode, "state_dict": best_state, "threshold": threshold},
               os.path.join(config.MODELS_DIR, f"{mode}.pth"))
    print(f"[{mode}] mejor ROC-AUC val {best_auc:.4f} | umbral {threshold:.2f} | guardado en models/{mode}.pth\n")
    return {"best_val_auc": round(best_auc, 4), "threshold": threshold, "history": history}


def f1_at(y, p, th):
    pred = p >= th
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=config.MODES)
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Entrenando en: {device}\n")
    feats = load_feature_split("train")
    train_idx, val_idx = split_indices(feats["labels"].numpy())
    print(f"Entrenamiento: {len(train_idx)} memes | validación: {len(val_idx)} memes\n")

    summary = {mode: train_mode(mode, feats, train_idx, val_idx, device) for mode in args.modes}

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
