"""Clasifica un meme nuevo con el modelo multimodal entrenado.

Uso:  python -m src.predict --image ruta/al/meme.png --text "texto del meme"
"""
import argparse
import torch
from PIL import Image

import src.config as config
from src.features.vision_extractor import MCVisionExtractor
from src.features.text_extractor import MCTextExtractor
from src.evaluate import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--mode", default="fusion", choices=["imagen", "texto", "fusion"])
    args = parser.parse_args()

    device = config.DEVICE
    v = MCVisionExtractor().extract_batch([Image.open(args.image)]).to(device)
    t = MCTextExtractor().extract_batch([args.text]).to(device)
    model, th = load_model(args.mode, device)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(v, t)).item()

    label = "CONTIENE ODIO" if p >= th else "sin odio"
    print(f"Probabilidad de odio: {p:.1%} (umbral {th:.2f}) -> {label}")


if __name__ == "__main__":
    main()
