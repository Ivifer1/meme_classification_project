import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super(FusionMLP, self).__init__()
        
        # El Perceptrón Multicapa (MLP)
        self.classifier = nn.Sequential(
            # ✨ MAGIA: LazyLinear calcula la dimensión de entrada automáticamente
            nn.LazyLinear(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1) # Capa final de salida binaria
        )

    def forward(self, vision_features, text_features):
        # Unimos visión y texto
        fused_features = torch.cat((vision_features, text_features), dim=1)
        
        # Pasamos por la red neuronal
        logits = self.classifier(fused_features)
        return logits