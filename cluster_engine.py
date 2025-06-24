from typing import Optional
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ClusterEngine:
    """
    Agrupa vectores usando el algoritmo HDBSCAN y evalúa la calidad del clustering.
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        metric: str = "euclidean",
        cluster_selection_method: str = "leaf",
        cluster_selection_epsilon: float = 0.02,
        **kwargs
    ):
        self.model: Optional[hdbscan.HDBSCAN] = None

        self.params = dict(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            **kwargs
        )

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """
        Aplica PCA, normaliza y ajusta HDBSCAN.
        También calcula métricas de evaluación del clustering.
        """
        assert vectors.ndim == 2, "Esperá un array de forma (N, dim)"

        # PCA
        pca = PCA(n_components=3)
        vectors_pca = pca.fit_transform(vectors)
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        print(f"Varianza acumulada por PCA (componentes 1-50): {explained_var[-1]:.4f}")

        # Normalización L2
        vectors_norm = normalize(vectors_pca, norm="l2", axis=1)

        # HDBSCAN
        self.model = hdbscan.HDBSCAN(**self.params).fit(vectors_norm)
        labels = self.model.labels_

        # Evaluación (solo si hay al menos 2 clusters válidos)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            silhouette = silhouette_score(vectors_norm, labels)
            ch_index = calinski_harabasz_score(vectors_norm, labels)
            db_index = davies_bouldin_score(vectors_norm, labels)

            print(f"Silhouette Score (más alto, mejor): {silhouette:.4f}")
            print(f"Calinski-Harabasz Index (más alto, mejor): {ch_index:.2f}")
            print(f"Davies-Bouldin Index (más bajo, mejor): {db_index:.4f}")
        else:
            print("No hay suficientes clusters válidos para calcular métricas.")

        return labels