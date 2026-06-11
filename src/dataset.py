import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import src.config as config

class HatefulMemesDataset(Dataset):
    def __init__(self, split="train"):
        """
        Dataset de PyTorch que usa tu config.py para cargar los memes.
        split: Puede ser "train", "dev" o "test".
        """
        # Usamos tu configuración para armar la ruta
        jsonl_path = os.path.join(config.RAW_DATA_DIR, f"{split}.jsonl")
        self.data = []
        
        if not os.path.exists(jsonl_path):
            print(f"[Error] No se encontró el archivo de datos en: {jsonl_path}")
            return

        print(f"📂 Leyendo dataset desde: {jsonl_path}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
        print(f"✅ ¡Dataset cargado correctamente! Total de registros: {len(self.data)}")

    def __len__(self):
        # PyTorch necesita saber cuántos memes hay en total
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch usará esta función para pedir los memes uno por uno
        item = self.data[idx]
        
        # 1. Ruta de la imagen (usando config.RAW_DATA_DIR como base)
        img_filename = item.get("img", "")
        img_path = os.path.join(config.RAW_DATA_DIR, img_filename)
        
        # 2. Texto
        text = item.get("text", "")
        
        # 3. Etiqueta (Label) -> Lo convertimos a un Tensor matemático
        label = item.get("label", 0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_path, text, label_tensor

# ==========================================
# BLOQUE DE PRUEBA DEL DATALOADER
# ==========================================
if __name__ == "__main__":
    print("🛠️ Inicializando prueba del DataLoader de PyTorch...")
    
    # 1. Instanciamos tu nuevo Dataset
    dataset = HatefulMemesDataset(split="train")
    
    if len(dataset) > 0:
        # 2. Creamos el DataLoader (agrupa en lotes de 4 y los mezcla)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 3. Sacamos un lote de prueba
        batch_img_paths, batch_texts, batch_labels = next(iter(dataloader))
        
        print("\n📦 LOTE DE PRUEBA EXITOSO (BATCH SIZE = 4):")
        print(f"-> Rutas extraídas: {batch_img_paths}")
        print(f"-> Etiquetas matemáticas: {batch_labels}")
        print("\n✅ El Dataset está listo para inyectar datos a la Red Neuronal.")