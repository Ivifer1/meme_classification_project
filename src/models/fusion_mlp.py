import torch
import torch.nn as nn

import src.config as config

VISION_DIM = 2048
TEXT_DIM = 768


class FusionMLP(nn.Module):
    """Clasificador binario sobre los embeddings congelados.

    mode = "imagen": usa solo el vector de ResNet50 (2048)
    mode = "texto":  usa solo el vector de DistilBERT (768)
    mode = "fusion": early fusion, concatena ambos (2048 + 768 = 2816)

    Cada modalidad se normaliza con LayerNorm antes de concatenar para que
    ninguna domine solo por la escala de sus valores.
    """

    def __init__(self, mode="fusion", hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT):
        super().__init__()
        if mode not in ("imagen", "texto", "fusion"):
            raise ValueError(f"Modo desconocido: {mode}")
        self.mode = mode

        self.norm_v = nn.LayerNorm(VISION_DIM)
        self.norm_t = nn.LayerNorm(TEXT_DIM)
        in_dim = {"imagen": VISION_DIM, "texto": TEXT_DIM, "fusion": VISION_DIM + TEXT_DIM}[mode]

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),   # un logit: >0 tiende a "odio"
        )

    def forward(self, vision_features, text_features):
        if self.mode == "imagen":
            x = self.norm_v(vision_features)
        elif self.mode == "texto":
            x = self.norm_t(text_features)
        else:
            x = torch.cat((self.norm_v(vision_features), self.norm_t(text_features)), dim=1)
        return self.classifier(x)
