from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Clase para generar embeddings de texto usando un modelo local de SentenceTransformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: nombre del modelo local para embeddings.
        """
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Devuelve el embedding para un texto dado.

        Args:
            text: Texto a convertir en embedding.
        Returns:
            Un array de numpy con floats representando el embedding del texto.
        """
        vector: np.ndarray = self.model.encode(text, convert_to_numpy=True)
        return vector

    def get_embeddings(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Devuelve embeddings para una lista de textos en batches.

        Args:
            texts: Lista de textos a convertir en embeddings.
            batch_size: Tamaño del batch para procesamiento por lotes.
        Returns:
            Un array de numpy de dimensión (n_texts, embedding_dim) representando los embeddings.
        """
        vectors: np.ndarray = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True
        )
        return vectors
