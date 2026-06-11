import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

# 🌟 AGREGA ESTA LÍNEA AQUÍ:
import src.config as config

# Tus otros imports...
from src.dataset import HatefulMemesDataset
from src.features.vision_extractor import MCVisionExtractor 
from src.features.text_extractor import MCTextExtractor
from src.models.fusion_mlp import FusionMLP

def train():
    print("🚀 INICIANDO FASE DE ENTRENAMIENTO MULTIMODAL...\n")

    # 1. Detectar Dispositivo (Mac/Windows/CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️ Entrenando en dispositivo: {device}\n")

    # 2. Preparar los Datos
    print("📦 Cargando Dataset de Entrenamiento...")
    train_dataset = HatefulMemesDataset(split="train")
    # Batch size de 16 es un buen balance para que la Mac no se quede sin memoria
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. Inicializar Modelos
    print("⚙️ Inicializando Extractores y Red de Fusión...")
    vision_extractor = MCVisionExtractor()
    text_extractor = MCTextExtractor()
    
    fusion_network = FusionMLP().to(device)
    fusion_network.train() # Ponemos la red en MODO ENTRENAMIENTO

    # 4. Configurar Función de Pérdida y Optimizador (Rúbrica Días 3-5)
    criterion = nn.BCEWithLogitsLoss() # Ideal para clasificar 0 o 1
    optimizer = optim.Adam(fusion_network.parameters(), lr=0.001) # Optimizador estándar

    # 5. CICLO DE ENTRENAMIENTO (Rúbrica Días 6-7)
    num_epochs = 3 # Número de pasadas completas al dataset
    
    for epoch in range(num_epochs):
        print(f"\n=====================================")
        print(f"🔄 ÉPOCA {epoch + 1}/{num_epochs}")
        print(f"=====================================")
        
        running_loss = 0.0
        
        for batch_idx, (img_paths, texts, labels) in enumerate(train_loader):
            # Mover las etiquetas al dispositivo
            labels = labels.unsqueeze(1).to(device) # Ajustar forma a [batch, 1]
            
            # --- FASE 1: EXTRAER CARACTERÍSTICAS (congeladas, sin gradiente) ---
            vision_features_list = []
            with torch.no_grad(): 
                for img_path in img_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        img = Image.new('RGB', (224, 224), color='black')
                    
                    v_feat = vision_extractor.extract_features(img)
                    vision_features_list.append(v_feat.view(-1)) 
                
                # Apilamos las imágenes [batch_size, 2048]
                batch_vision_features = torch.stack(vision_features_list, dim=0).to(device)
                
                # Extraemos los textos
                batch_text_features = text_extractor.extract_features(texts).to(device)
                
                # 🌟 PARCHE PARA EL ÚLTIMO LOTE 🌟
                if len(batch_text_features.shape) == 1:
                    # Si el simulador da un vector 1D (768), lo clonamos para la cantidad exacta de memes (16 o 4)
                    batch_text_features = batch_text_features.unsqueeze(0).expand(len(texts), -1)
                elif len(batch_text_features.shape) == 2 and batch_text_features.shape[0] != len(texts):
                    # Solo por seguridad, si la forma viene rara, forzamos su estructura
                    batch_text_features = batch_text_features.view(len(texts), -1)

            # --- FASE 2: RED DE FUSIÓN Y PREDICCIÓN ---
            optimizer.zero_grad() # Limpiamos la memoria del optimizador
            
            # Pasamos los vectores por la red
            logits = fusion_network(batch_vision_features, batch_text_features)
            
            # --- FASE 3: CALCULAR ERROR (Loss) Y AJUSTAR PESOS ---
            loss = criterion(logits, labels)
            loss.backward() # Propagación hacia atrás (Backpropagation)
            optimizer.step() # Actualizar pesos
            
            # Monitoreo de la curva de pérdida
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"   [Batch {batch_idx + 1}/{len(train_loader)}] Training Loss: {loss.item():.4f}")

        # Promedio de pérdida de la época
        epoch_loss = running_loss / len(train_loader)
        print(f"✅ FIN ÉPOCA {epoch + 1} | Loss Promedio: {epoch_loss:.4f}")

    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")

    # ==========================================
    # 💾 GUARDAR EL MODELO ENTRENADO
    # ==========================================
    # Creamos la ruta usando tu config (ej: models/fusion_mlp_model.pth)
    model_dir = os.path.dirname(config.MODEL_PATH) if hasattr(config, 'MODEL_PATH') else "models"
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, "fusion_mlp_model.pth")
    print(f"💾 Guardando los pesos del modelo en: {save_path}...")
    
    # Guardamos el diccionario de estado del modelo
    torch.save(fusion_network.state_dict(), save_path)
    print("✅ ¡Modelo guardado correctamente y listo para ser evaluado!")

if __name__ == "__main__":
    train()
