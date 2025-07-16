from pathlib import Path
import json
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=)

class ClusterLoader:
    def __init__(self, path_json):
        self.path_json = path_json
        self.clusters = self._load()

    def _load(self):
        with open(self.path_json, encoding="utf-8") as f:
            return json.load(f)

    def obtener_top_clusters(self, n=2):
        return sorted(self.clusters, key=lambda c: len(c["articulos_incluidos"]), reverse=True)[:n]


class GPTArticleSeparator:
    def __init__(self, model="gpt-4"):
        self.model = model

    def refinar(self, texto: str, cluster_id: int, articulo_id: str) -> str:
        prompt = (
            f"Estás refinando un artículo del cluster {cluster_id}, ID {articulo_id}.\n"
            f"El siguiente texto contiene fragmentos posiblemente dispersos que pertenecen a un mismo artículo periodístico. "
            f"Unificá el contenido en un solo artículo coherente, redactado en español, claro, con estilo periodístico y sin repeticiones:\n\n{texto}"
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Sos un experto en redacción periodística."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR en cluster {cluster_id}, artículo {articulo_id}]: {e}"


class RAGPostProcessor:
    def __init__(self, loader: ClusterLoader, separator: GPTArticleSeparator):
        self.loader = loader
        self.separator = separator
        self.resultados = []

    def ejecutar(self):
        top_clusters = self.loader.obtener_top_clusters(2)

        for cluster in tqdm(top_clusters, desc="Procesando los 2 clusters más grandes"):
            chunks_por_articulo = defaultdict(list)
            for id_chunk in cluster["articulos_incluidos"]:
                articulo, _ = id_chunk.split("_p")
                chunks_por_articulo[articulo].append(id_chunk)

            for articulo_id, chunks in chunks_por_articulo.items():
                texto = self._reconstruir_texto(chunks, cluster["texto_reconstruido"])
                texto_refinado = self.separator.refinar(texto, cluster["cluster_id"], articulo_id)
                self.resultados.append({
                    "cluster_id": cluster["cluster_id"],
                    "articulo_id": articulo_id,
                    "texto_refinado": texto_refinado
                })

    def _reconstruir_texto(self, ids_chunks, texto_total):
        # En esta versión se usa todo el texto reconstruido por ahora (mejorable si tenés los chunks originales)
        joined_ids = ', '.join(ids_chunks)
        return f"(Fragmentos utilizados: {joined_ids})\n\n{texto_total}"

    def exportar(self, output_txt="noticias_reorganizadas.txt"):
        with open(output_txt, "w", encoding="utf-8") as f:
            for r in self.resultados:
                f.write(f"--- CLUSTER {r['cluster_id']} | ARTÍCULO {r['articulo_id']} ---\n")
                f.write(r["texto_refinado"] + "\n\n")
        print(f"✅ Archivo exportado: {output_txt}")


def run_rag_refinamiento(args=None):
    loader = ClusterLoader("output_files/noticias_reconstruidas_clasificadas.json")
    separator = GPTArticleSeparator()
    processor = RAGPostProcessor(loader, separator)
    processor.ejecutar()
    processor.exportar()