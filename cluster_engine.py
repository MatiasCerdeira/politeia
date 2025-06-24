from typing import Optional
import numpy as np
import hdbscan
import numpy as np
from sklearn.preprocessing import normalize


class ClusterEngine:
    """
    Agrupa vectores usando el algoritmo HDBSCAN.

    Attributes:
        model (Optional[hdbscan.HDBSCAN]): Una vez que se llama a fit, aca queda la instancia entrenada de HDBSCAN.
        min_cluster_size (int): Tamaño mínimo de un cluster.
        min_samples (int): Controla la sensibilidad al ruido (por defecto igual a min_cluster_size).
        kwargs (dict): Otros parámetros de HDBSCAN (metric, cluster_selection_epsilon, etc.).
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
        """
        Args:
            min_cluster_size: tamaño mínimo de un cluster.
            min_samples: controla sensibilidad al ruido (por defecto igual a min_cluster_size).
            kwargs: otros parámetros de HDBSCAN (metric, cluster_selection_epsilon, etc.).
        """
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
        Ajusta HDBSCAN sobre los vectores y devuelve etiquetas de cluster:
        - etiqueta -1 = ruido
        - 0,1,2,... = clusters

        Args:
            vectors (np.ndarray):
                Arreglo de forma (N, dim) con los vectores a clusterizar.

        Returns:
            np.ndarray:
                Un array de numpy con las etiquetas de cluster para cada vector.
                Las etiquetas son enteros donde -1 indica ruido y 0, 1, 2,... indican clusters.
        """
        assert vectors.ndim == 2, "Esperá un array de forma (N, dim)"

        vectors_norm = normalize(vectors, norm="l2", axis=1)

        self.model = hdbscan.HDBSCAN(**self.params).fit(vectors_norm)
        return self.model.labels_
