import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import src.config as config


class MCVisionExtractor:
    """ResNet50 preentrenada en ImageNet, sin la capa final.
    Devuelve un embedding de 2048 dimensiones por imagen."""

    def __init__(self, pretrained=True):
        print(f"[{config.DEVICE}] Cargando extractor de visión ({config.VISION_MODEL_NAME})...")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()          # quitamos el clasificador de ImageNet

        for param in self.model.parameters():  # extractor congelado
            param.requires_grad = False

        self.device = config.DEVICE
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_batch(self, pil_images):
        """Lista de imágenes PIL -> tensor [N, 2048]"""
        batch = torch.stack([self.transform(img.convert("RGB")) for img in pil_images]).to(self.device)
        return self.model(batch).cpu()

    def extract_features(self, pil_image):
        """Una imagen PIL -> tensor [2048]"""
        return self.extract_batch([pil_image])[0]


if __name__ == "__main__":
    extractor = MCVisionExtractor()
    vector = extractor.extract_features(Image.new("RGB", (300, 300), color="blue"))
    print(f"Embedding visual: {tuple(vector.shape)} (esperado: (2048,))")
