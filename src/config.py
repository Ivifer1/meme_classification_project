import os
import torch

# --- Rutas del proyecto ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# La carpeta de datos se puede cambiar con la variable de entorno MEME_DATA_DIR
# (útil en Colab, donde los datos se descomprimen en otra ruta).
# Debe contener train.jsonl, dev.jsonl y la carpeta img/
RAW_DATA_DIR = os.environ.get("MEME_DATA_DIR", os.path.join(BASE_DIR, "data", "raw", "data"))

FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")   # embeddings precalculados
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Hardware (Windows con NVIDIA, Mac con Apple Silicon o CPU) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Modelos base (congelados, solo se usan para extraer características) ---
VISION_MODEL_NAME = "resnet50"
TEXT_MODEL_NAME = "distilbert-base-uncased"
IMAGE_SIZE = (224, 224)
TEXT_MAX_LENGTH = 64            # los textos de los memes son cortos
EXTRACT_BATCH_SIZE = 64

# --- Entrenamiento de la red de fusión ---
SEED = 42
VAL_FRACTION = 0.1              # parte de train.jsonl reservada para validación
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
HIDDEN_DIM = 512
PATIENCE = 6                    # early stopping según el ROC-AUC de validación

# Modos que se entrenan y comparan
MODES = ["imagen", "texto", "fusion"]
