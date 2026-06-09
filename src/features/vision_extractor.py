import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import src.config as config

class MCVisionExtractor:
    def __init__(self):
        print(f"[{config.DEVICE}] Cargando Extractor de Visión ({config.VISION_MODEL_NAME})...")
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # RÚBRICA: Reemplazar la capa fc para extraer embeddings limpios
        self.model.fc = nn.Identity()
        
        # RÚBRICA: Congelar parámetros
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.device = config.DEVICE
        self.model.to(self.device)
        self.model.eval()

        # Transformaciones estándar de ResNet
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        tensor_img = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor_img)
        return embedding.squeeze(0)

if __name__ == "__main__":
    # Autoprueba del módulo
    extractor = MCVisionExtractor()
    dummy_img = Image.new("RGB", (300, 300), color="blue")
    vector = extractor.extract_features(dummy_img)
    print(f"-> Extracción visual exitosa. Tamaño del embedding: {vector.shape} (Esperado: [2048])")