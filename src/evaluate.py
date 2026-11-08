"""Paso 3: evalúa los modelos entrenados en el split dev (500 memes etiquetados
que no se usaron para entrenar ni para elegir hiperparámetros).

Compara: solo imagen, solo texto, early fusion y late fusion (promedio de las
probabilidades de los modelos de imagen y texto).

Genera en results/:
  metrics.json, metrics.md, roc_curves.png, confusion_matrix_fusion.png

Uso:  python -m src.evaluate
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

import src.config as config
from src.dataset import load_feature_split, make_tensor_dataset
from src.models.fusion_mlp import FusionMLP
from src.train import predict_proba

NAMES = {
    "imagen": "Solo imagen (ResNet50)",
    "texto": "Solo texto (DistilBERT)",
    "late_fusion": "Late fusion (promedio)",
    "fusion": "Early fusion (concatenación)",
}


def load_model(mode, device):
    ckpt = torch.load(os.path.join(config.MODELS_DIR, f"{mode}.pth"), map_location="cpu")
    model = FusionMLP(mode)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt["threshold"]


def compute_metrics(y, p, th):
    pred = (p >= th).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "threshold": th,
    }


def main():
    device = config.DEVICE
    feats = load_feature_split("dev")
    loader = DataLoader(make_tensor_dataset(feats), batch_size=256)

    probs, thresholds = {}, {}
    for mode in ["imagen", "texto", "fusion"]:
        model, th = load_model(mode, device)
        probs[mode], y = predict_proba(model, loader, device)
        thresholds[mode] = th
    y = y.astype(int)
    probs["late_fusion"] = (probs["imagen"] + probs["texto"]) / 2
    thresholds["late_fusion"] = (thresholds["imagen"] + thresholds["texto"]) / 2

    results = {m: {k: round(float(v), 4) for k, v in compute_metrics(y, probs[m], thresholds[m]).items()}
               for m in NAMES}

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"split": "dev", "n": int(len(y)), "positivos": int(y.sum()), "modelos": results}, f, indent=2)

    lines = ["| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|"]
    for m, name in NAMES.items():
        r = results[m]
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} |")
    table = "\n".join(lines)
    with open(os.path.join(config.RESULTS_DIR, "metrics.md"), "w") as f:
        f.write(f"Evaluación en dev ({len(y)} memes, {int(y.sum())} con odio)\n\n{table}\n")
    print(table)

    # Curvas ROC
    plt.figure(figsize=(6, 5))
    for m, name in NAMES.items():
        fpr, tpr, _ = roc_curve(y, probs[m])
        plt.plot(fpr, tpr, label=f"{name} (AUC {results[m]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curvas ROC en dev")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # Matriz de confusión del modelo multimodal
    cm = confusion_matrix(y, (probs["fusion"] >= thresholds["fusion"]).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Sin odio", "Odio"])
    ax.set_yticks([0, 1], ["Sin odio", "Odio"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Early fusion en dev")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrix_fusion.png"), dpi=150)
    plt.close(fig)
    print(f"\nResultados guardados en {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
