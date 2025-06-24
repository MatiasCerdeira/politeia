from typing import List, Tuple, Optional
import faiss
import numpy as np


class VectorStoreManager:
    """
    Administra un índice FAISS para la búsqueda de similitud de vectores.

    Las responsabilidades incluyen crear un índice FAISS, agregar vectores con IDs asociados,
    realizar búsquedas de los k vecinos más cercanos y guardar el índice en disco.

    La clase mantiene un mapeo interno entre las posiciones de los vectores en el índice y sus
    IDs correspondientes. Este mapeo se almacena en memoria en la lista `ids` y debe
    persistirse por separado si es necesario.

    Atributos:
        dim (int): Dimensionalidad de los vectores.
        index (faiss.Index): Instancia del índice FAISS.
        ids (List[int]): Lista que mapea posiciones de vectores a sus IDs.
    """

    def __init__(self, dim: int):
        """
        Inicializa el VectorStoreManager.

        Args:
            dim (int): Dimensionalidad de los vectores a indexar.

        Al inicializar, se crea una lista vacía `ids` para mapear posiciones de vectores a sus IDs.
        """
        self.dim = dim

        # Crear un índice FAISS para búsqueda de similitud L2
        self.index = faiss.IndexFlatL2(dim)

        # Lista para mapear posiciones de vectores en el índice a sus IDs
        self.ids: List[int] = []

    def add(self, vectors: np.ndarray, ids: List[int]) -> None:
        """
        Agrega vectores al índice FAISS y almacena sus IDs correspondientes.

        Args:
            vectors (np.ndarray): Arreglo de forma (N, dim) que contiene los vectores a agregar.
                Se espera dtype float32 o convertible a float32.
            ids (List[int]): Lista de IDs enteros correspondiente a cada vector.

        Precondiciones:
            - vectors debe tener forma (N, dim).
            - La longitud de ids debe coincidir con el número de vectores (N).
            - ids deben ser enteros; si no lo son, deben convertirse antes de pasarlos.

        Efecto:
            - Los vectores se agregan al índice FAISS.
            - Los IDs se agregan a la lista interna `ids` en el mismo orden.
        """
        # Convertir los vectores a float32 como requiere FAISS
        self.index.add(vectors.astype(np.float32))
        # Extender la lista interna de IDs con los nuevos IDs
        self.ids.extend(ids)

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Realiza una búsqueda de los k vecinos más cercanos en el índice para el/los vector(es) de consulta dado(s).

        Args:
            query_vec (np.ndarray): Vector(es) de consulta de forma (M, dim) o (dim,).
                                    Si se proporciona un solo vector, debe ser reformado en consecuencia.
            k (int): Número de vecinos más cercanos a recuperar.

        Returns:
            Tupla que contiene:
                - distances (np.ndarray): Arreglo de forma (M, k) con las distancias L2 a los vecinos.
                - result_ids (List[List[int]]): Lista de listas que contiene los IDs de los vecinos más cercanos
                  correspondientes a cada vector de consulta.

        Comportamiento:
            - El método convierte los vectores de consulta a float32.
            - Utiliza el índice FAISS para encontrar los k vecinos más cercanos.
            - Mapea los índices devueltos a los IDs almacenados.
        """
        # Realizar la búsqueda en el índice FAISS
        D, I = self.index.search(query_vec.astype(np.float32), k)
        # Mapear los índices devueltos por FAISS a los IDs almacenados
        result_ids = [[self.ids[j] for j in row] for row in I]
        return D, result_ids

    def save(self, path: str) -> None:
        """
        Guarda el índice FAISS en disco en la ruta especificada.

        Nota:
            Solo se guarda el índice FAISS. El mapeo de IDs (`self.ids`) debe guardarse por separado si se requiere persistencia.
        """
        faiss.write_index(self.index, path)
