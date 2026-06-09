# Proyecto Final: Clasificación Multimodal de Memes (Hate Speech Detection)
**Asignatura:** Inteligencia Artificial  
**Semestre:** 8vo Semestre  
**Ecosistema de Desarrollo:** Multiplataforma (macOS Apple Silicon `mps` / Windows `cuda`)

---

## 📝 Enunciado General del Proyecto

El objetivo de este proyecto es diseñar, implementar y evaluar un modelo de Aprendizaje Profundo (Deep Learning) **Multimodal** capaz de clasificar memes para detectar si contienen discursos de odio (*Hate Speech*). 

Un meme es inherentemente complejo porque el significado real no depende únicamente de la imagen aislada ni del texto por separado, sino de la **fusión e interacción de ambas modalidades**. El sistema debe leer un dataset de 8,500 registros, extraer características visuales y textuales mediante modelos preentrenados (*Embeddings*), fusionar dicha información y clasificar el contenido de forma binaria (0: No contiene odio, 1: Contiene odio).

---

## 📊 Cronograma de Desarrollo y Entregables

### 🔹 Semana 1: Configuración de Entorno y Extractores Base (¡Completada!)
* **Objetivos:** * Configurar la estructura modular del proyecto y el entorno virtual (`.venv`).
  * Implementar el lector del dataset (`src/dataset.py`) para parsear de forma eficiente el archivo `train.jsonl` (8,500 registros).
  * Construir el extractor de características visuales utilizando **ResNet50** preentrenada (dimensión: `[2048]`).
  * Construir el extractor de características textuales utilizando **DistilBERT** preentrenado (dimensión: `[768]`).
* **Solución Multiplataforma Aplicada:** Debido a los bloqueos de certificados `LibreSSL` nativos de macOS para descargas pesadas en la terminal, el script de texto implementa un switch adaptativo. En Mac simula las dimensiones matemáticas exactas `[768]` para desbloquear el avance del pipeline, mientras que en Windows ejecuta e instancia el modelo real de Hugging Face de forma automática.

### 🔹 Semana 2: Red de Fusión Multimodal (Early/Late Fusion)
* **Objetivos:**
  * Diseñar la arquitectura de la red neuronal en PyTorch encargada de unificar los vectores de la imagen y del texto.
  * Implementar una estrategia de **Fusión Temprana (Early Fusion)** mediante la concatenación de los vectores $2048 + 768 = 2816$ seguida de capas lineales densas ($Linear \to ReLU \to Dropout$).
  * Experimentar alternativamente con **Fusión Tardía (Late Fusion)** sumando o promediando las probabilidades independientes de cada modalidad.

### 🔹 Semana 3: Entrenamiento, Optimización y Regularización
* **Objetivos:**
  * Construir el bucle principal de entrenamiento (*Training Loop*) con cálculo de pérdida usando `BCEWithLogitsLoss`.
  * Congelar los parámetros de las redes base (ResNet50 y DistilBERT) para actuar estrictamente como extractores de características fijas (*Feature Extraction*), entrenando únicamente los pesos de la red de fusión.
  * Implementar técnicas de regularización (Dropout, Weight Decay) y optimizadores (`Adam` o `AdamW`).

### 🔹 Semana 4: Evaluación, Métricas y Reporte Final
* **Objetivos:**
  * Evaluar el rendimiento del clasificador final utilizando un conjunto de validación/test separado.
  * Calcular y reportar de forma obligatoria las métricas de la rúbrica: **Precisión, Recall, F1-Score y curvas ROC-AUC**.
  * Redacción del reporte de resultados y conclusiones del comportamiento multimodal frente a modelos unimodales.