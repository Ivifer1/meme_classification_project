import torch
from transformers import AutoTokenizer, AutoModel

import src.config as config


class MCTextExtractor:
    """DistilBERT preentrenado y congelado.
    Devuelve un embedding de 768 dimensiones por texto (mean pooling sobre los tokens)."""

    def __init__(self):
        print(f"[{config.DEVICE}] Cargando extractor de texto ({config.TEXT_MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)

        for param in self.model.parameters():  # extractor congelado
            param.requires_grad = False

        self.device = config.DEVICE
        self.model.to(self.device).eval()

    @torch.no_grad()
    def extract_batch(self, texts):
        """Lista de textos -> tensor [N, 768]"""
        texts = [t if t and t.strip() else "[empty]" for t in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=config.TEXT_MAX_LENGTH).to(self.device)
        hidden = self.model(**inputs).last_hidden_state          # [N, tokens, 768]

        # Promedio de los tokens reales de cada texto (sin contar el padding)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return embeddings.cpu()

    def extract_features(self, text):
        """Un texto -> tensor [768]"""
        return self.extract_batch([text])[0]


if __name__ == "__main__":
    extractor = MCTextExtractor()
    vectors = extractor.extract_batch(["meme text sample", "another different meme"])
    print(f"Embeddings de texto: {tuple(vectors.shape)} (esperado: (2, 768))")
