"""Paso 1: extrae una sola vez los embeddings de imagen (ResNet50) y texto (DistilBERT)
de cada split y los guarda en data/features/<split>.pt

Como los extractores están congelados, no hace falta recalcularlos en cada época:
el entrenamiento de la red de fusión pasa de horas a segundos.

Uso:  python -m src.extract_features --splits train dev
"""
import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm

import src.config as config
from src.dataset import HatefulMemesDataset
from src.features.vision_extractor import MCVisionExtractor
from src.features.text_extractor import MCTextExtractor


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        print(f"Aviso: no se pudo abrir {path}, se usa una imagen negra")
        return Image.new("RGB", config.IMAGE_SIZE)


def extract_split(split, vision, text):
    ds = HatefulMemesDataset(split)
    vis, txt = [], []
    bs = config.EXTRACT_BATCH_SIZE
    for start in tqdm(range(0, len(ds), bs), desc=f"Extrayendo {split}"):
        items = ds.data[start:start + bs]
        imgs = [load_image(ds[start + i][0]) for i in range(len(items))]
        vis.append(vision.extract_batch(imgs))
        txt.append(text.extract_batch([it.get("text", "") for it in items]))

    feats = {
        "ids": torch.tensor([int(it["id"]) for it in ds.data]),
        "texts": [it.get("text", "") for it in ds.data],
        "vision": torch.cat(vis),
        "text": torch.cat(txt),
        "labels": torch.tensor([float(it.get("label", -1)) for it in ds.data]),
    }
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    out = os.path.join(config.FEATURES_DIR, f"{split}.pt")
    torch.save(feats, out)
    print(f"Guardado {out}: imagen {tuple(feats['vision'].shape)}, texto {tuple(feats['text'].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    args = parser.parse_args()

    vision = MCVisionExtractor()
    text = MCTextExtractor()
    for split in args.splits:
        extract_split(split, vision, text)


if __name__ == "__main__":
    main()
