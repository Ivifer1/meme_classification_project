import os
import json
import torch
from PIL import Image

# Importamos tus clases reales
from src.features.vision_extractor import MCVisionExtractor 
from src.features.text_extractor import MCTextExtractor # ⚠️ CAMBIA ESTO por el nombre real de tu clase de texto
from src.models.fusion_mlp import FusionMLP

def run_real_pipeline():
    print("🚀 Iniciando Pipeline Multimodal con DATOS REALES...\n")
    
    # 1. Rutas a tus datos reales (Las que ocultamos de GitHub pero existen en tu Mac)
    jsonl_path = "data/raw/data/train.jsonl"
    img_dir = "data/raw/data/img/"
    
    # 2. Leer el primer meme de tu dataset real
    print(f"📂 Abriendo la base de datos: {jsonl_path}")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_meme = json.loads(f.readline())
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo train.jsonl. Verifica la ruta.")
        return
        
    img_filename = first_meme.get("img", "") # El campo donde dice "img/01234.png"
    real_text = first_meme.get("text", "")   
    
    # Ajustar la ruta de la imagen dependiendo de cómo viene en tu jsonl
    if img_filename.startswith("img/"):
        img_filename = img_filename.replace("img/", "")
    img_path = os.path.join(img_dir, img_filename)
    
    print(f"🖼️ Imagen real cargada: {img_path}")
    print(f"📝 Texto real cargado: '{real_text}'\n")
    
    # Abrimos la imagen de verdad usando la ruta
    try:
        real_image = Image.open(img_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró la imagen en {img_path}")
        return
    
    # 3. Instanciar los modelos
    print("⚙️ Cargando modelos (Visión, Texto y Fusión MLP)...")
    vision_extractor = MCVisionExtractor()
    text_extractor = MCTextExtractor() 

    # DETECCIÓN DE DISPOSITIVO MULTIPLATAFORMA (Windows / Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda") # Para tus compañeros en Windows con Nvidia
        print("🖥️ Dispositivo detectado: CUDA (Nvidia GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Para ti en Mac (Apple Silicon)
        print("🖥️ Dispositivo detectado: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")  # Fallback universal
        print("🖥️ Dispositivo detectado: CPU")

    # Enviamos la red al dispositivo correcto
    fusion_network = FusionMLP().to(device)
    fusion_network.eval()
    
    # 4. Procesamiento Matemático
    print("🧠 Procesando las modalidades a través de la Red Neuronal...")
    with torch.no_grad():
        vision_features = vision_extractor.extract_features(real_image)
        text_features = text_extractor.extract_features(real_text)
        
        if len(vision_features.shape) == 1:
            vision_features = vision_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)
            
        # 🌟 Aseguramos que los datos viajen al mismo chip que la red
        vision_features = vision_features.to(device)
        text_features = text_features.to(device)
            
        logits = fusion_network(vision_features, text_features)
        probability = torch.sigmoid(logits).item()
        
    print("\n===========================================")
    print(f"✅ RESULTADO DE LA RED NEURONAL:")
    print(f"-> Probabilidad de Hate Speech: {probability * 100:.2f}%")
    print("===========================================")

if __name__ == "__main__":
    run_real_pipeline()