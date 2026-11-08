import os
import json
import torch
from torch.utils.data import Dataset, TensorDataset

import src.config as config


def resolve_image_path(img_field):
    """Devuelve la ruta de la imagen. Acepta la versión original (.png)
    y la versión reducida para Colab (.jpg)."""
    path = os.path.join(config.RAW_DATA_DIR, img_field)
    if os.path.exists(path):
        return path
    alt = os.path.splitext(path)[0] + ".jpg"
    return alt if os.path.exists(alt) else path


class HatefulMemesDataset(Dataset):
    """Lee los registros de un split (train, dev o test) del dataset Hateful Memes.
    Cada elemento es (ruta_imagen, texto, etiqueta)."""

    def __init__(self, split="train"):
        jsonl_path = os.path.join(config.RAW_DATA_DIR, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"No se encontró {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        print(f"Split '{split}': {len(self.data)} memes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item.get("label", -1)   # el test público no trae etiqueta
        return resolve_image_path(item["img"]), item.get("text", ""), torch.tensor(label, dtype=torch.float32)


def load_feature_split(split):
    """Carga los embeddings precalculados por src/extract_features.py"""
    path = os.path.join(config.FEATURES_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. Primero ejecuta: python -m src.extract_features")
    return torch.load(path)


def make_tensor_dataset(feats, idx=None):
    v, t, y = feats["vision"], feats["text"], feats["labels"]
    if idx is not None:
        v, t, y = v[idx], t[idx], y[idx]
    return TensorDataset(v, t, y)
