import os
import torch

# --- Rutas del Proyecto ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Añadimos la subcarpeta 'data' que está dentro de 'raw' según tu captura
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "data") 
SPLITS_DATA_DIR = os.path.join(DATA_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fusion_mlp_model.pth")

# --- Configuración de Hardware Inteligente (Windows / Mac) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")          # GPUs Nvidia en Windows
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")           # GPU de tu MacBook
else:
    DEVICE = torch.device("cpu")           # CPU por defecto

# --- Hiperparámetros de la Semana 1 ---
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)

# --- Modelos Base ---
VISION_MODEL_NAME = "resnet50"
TEXT_MODEL_NAME = "distilbert-base-uncased"