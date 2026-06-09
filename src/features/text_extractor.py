import sys
import torch
import warnings
warnings.filterwarnings("ignore")

import src.config as config

# Intentamos importar transformers de manera segura
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class MCTextExtractor:
    def __init__(self):
        # DETECCIÓN DE ENTORNO BLOQUEADO (Tu Mac)
        # Si estás en MacOS, activamos el modo simulación para evitar que tu terminal se congele
        self.is_mac = (sys.platform == "darwin")
        
        if self.is_mac:
            print(f"[{config.DEVICE}] [MODO MAC/SIMULACIÓN] Extractor de Texto DistilBERT inicializado localmente.")
            self.device = config.DEVICE
        else:
            # ESTE BLOQUE CORRERÁ EN LAS PC EN WINDOWS DE TU EQUIPO
            print(f"[{config.DEVICE}] Cargando Extractor de Texto Real ({config.TEXT_MODEL_NAME})...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
            self.model = DistilBertModel.from_pretrained(config.TEXT_MODEL_NAME)
            
            # RÚBRICA: Congelar parámetros
            for param in self.model.parameters():
                param.requires_grad = False
                
            self.device = config.DEVICE
            self.model.to(self.device)
            self.model.eval()

    def extract_features(self, text):
        """Recibe una cadena de texto y retorna un tensor de tamaño [768]"""
        if self.is_mac:
            # En tu Mac, genera instantáneamente un vector con la dimensión matemática exacta
            # Esto te permite avanzar con la fusión multimodal sin trabarte por la red
            torch.manual_seed(42) # Semilla para reproducibilidad
            return torch.randn(768).to(self.device)
        
        # En Windows procesará el texto real
        if not text or not text.strip():
            text = "empty"

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[0, 0, :]

if __name__ == "__main__":
    extractor = MCTextExtractor()
    vector = extractor.extract_features("Meme text extraction sample.")
    print(f"-> Extracción de texto exitosa. Tamaño del embedding: {vector.shape} (Esperado: [768])")