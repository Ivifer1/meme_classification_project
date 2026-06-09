import os
import json
import src.config as config

def load_meme_dataset():
    # Cambiado de tren.jsonl a train.jsonl para que coincida con tu captura
    jsonl_path = os.path.join(config.RAW_DATA_DIR, "train.jsonl")
    
    if not os.path.exists(jsonl_path):
        print(f"[Error] No se encontró el archivo de datos en: {jsonl_path}")
        return []

    data = []
    print(f"Leyendo dataset desde: {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    print(f"¡Dataset cargado correctamente! Total de registros: {len(data)}")
    return data

if __name__ == "__main__":
    load_meme_dataset()