from pathlib import Path
import json
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=)

# === ETAPA 1: FILTRAR Y GUARDAR LOS TOP 2 CLUSTERS ===

def guardar_top_clusters(path_input_json, path_output_json="output_files/top_clusters.json", n=2):
    # Crear carpeta si no existe
    Path(path_output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(path_input_json, encoding="utf-8") as f:
        clusters = json.load(f)

    # Filtrar fuera el cluster -1
    clusters_filtrados = [c for c in clusters if c["cluster_id"] != -1]

    # Ordenar por cantidad de chunks y tomar top n
    top_clusters = sorted(clusters_filtrados, key=lambda c: len(c["articulos_incluidos"]), reverse=True)[:n]

    with open(path_output_json, "w", encoding="utf-8") as f_out:
        json.dump(top_clusters, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Se guardaron los {n} clusters más grandes (excluyendo cluster -1) en {path_output_json}")

# === ETAPA 2: PROCESAR CLUSTERS DESDE EL NUEVO JSON ===

class ClusterLoader:
    def __init__(self, path_json):
        self.path_json = path_json
        self.clusters = self._load()

    def _load(self):
        with open(self.path_json, encoding="utf-8") as f:
            return json.load(f)


class GPTArticleSeparator:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def refinar(self, texto: str, cluster_id: int, articulo_id: str) -> str:
        prompt = (
            f"Estás refinando un artículo del cluster {cluster_id}, ID {articulo_id}.\n"
            f"Dentro de este cluster se encuentran fragmentos de noticias. Dichos fragmentos pueden pertenecer a la misma noticia, aunque puede darse el caso de que haya fragmentos de noticias diferentes. "
            f"Quiero que unifiques los fragmentos correspondientes al mismo articulo dentro de los clusters. Esto significa que puede llegar a haber varios grupos de fragmentos dentro de un cluster, ya que puede haber varios articulos dentro de cada cluster"
            f"Utilizá los fragmentos de cada articulo y redactá un texto en español, claro, con estilo periodístico y sin repeticiones:\n\n{texto}"
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
        for cluster in tqdm(self.loader.clusters, desc="Procesando clusters seleccionados"):
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
        joined_ids = ', '.join(ids_chunks)
        return f"(Fragmentos utilizados: {joined_ids})\n\n{texto_total}"

    def exportar(self, output_txt="noticias_reorganizadas.txt"):
        with open(output_txt, "w", encoding="utf-8") as f:
            for r in self.resultados:
                f.write(f"--- CLUSTER {r['cluster_id']} | ARTÍCULO {r['articulo_id']} ---\n")
                f.write(r["texto_refinado"] + "\n\n")
        print(f"✅ Archivo exportado: {output_txt}")

def run_rag_refinamiento(args=None):
    ruta_entrada = "output_files/noticias_reconstruidas_clasificadas.json"
    ruta_filtrada = "output_files/top_clusters.json"
    guardar_top_clusters(ruta_entrada, ruta_filtrada, n=2)
    loader = ClusterLoader(ruta_filtrada)
    separator = GPTArticleSeparator()
    processor = RAGPostProcessor(loader, separator)
    processor.ejecutar()
    processor.exportar()

