Evaluación en dev (500 memes, 250 con odio)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Solo imagen (ResNet50) | 0.542 | 0.527 | 0.816 | 0.640 | 0.574 |
| Solo texto (DistilBERT) | 0.580 | 0.608 | 0.452 | 0.518 | 0.631 |
| Late fusion (promedio) | 0.590 | 0.575 | 0.688 | 0.627 | 0.643 |
| Early fusion (concatenación) | 0.626 | 0.657 | 0.528 | 0.585 | 0.672 |
