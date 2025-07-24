import argparse
import config
import json
import numpy as np


from news_scraper import NewsScraper
from news_scraper import CompleteArticle
from ChunkAI import ChunkAI
from embedder import Embedder
from vector_store_manager import VectorStoreManager
from cluster_engine import ClusterEngine
from rag_reconstructor import RAGReconstructor, ImportanciaClassifier, RAGPipeline
from refinar_rag_gpt import run_rag_refinamiento
from sklearn.decomposition import PCA
import pickle
import pandas as pd
from visualizer import to_dataframe, interactive_scatter


from utils import save_to_json


# Se corre con "make news"
def run_news(args):
    """
    - Corre unicamente el scraper de noticias.
    - El output se guarda en output_files/articulos_completos.json
    """
    scraper = NewsScraper(config.RSS_FEEDS)
    results: list[CompleteArticle] = scraper.scrape()

    # Convierto los resultados a un formato JSON serializable
    serializable_results = [article.__dict__ for article in results]

    save_to_json(serializable_results, f"{config.OUTPUT_DIR}/articulos_completos.json")


# Se corre con "make summaries"
def run_summarization(args):
    """""
    Corre el chunkeo de noticias con IA.

    Lee los artículos desde output_files/articulos_completos.json,
    genera chunks para cada uno, y guarda el resultado en
    output_files/articulos_completos_chunked.json.
    """
    input_path = f"{config.OUTPUT_DIR}/articulos_completos.json"
    output_path = f"{config.OUTPUT_DIR}/articulos_completos_chunked.json"

    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunker = ChunkAI()

    for i, article in enumerate(articles):
        print(f"🔍 Chunkeando artículo {i+1}/{len(articles)}: {article['title'][:50]}...")
        try:
            doc_id = article.get("id", f"art_{i:03d}")
            chunks = chunker.chunk_article(doc_id, article["title"], article["text"])
            article["chunks"] = chunks
        except Exception as e:
            print(f"❌ Error al chunkearear '{article['title'][:30]}': {e}")
            article["chunks"] = []

    save_to_json(articles, output_path)
    print("✅ Chunks guardados correctamente.")


# Se corre con "make summary_vectorization"
def run_chunk_vectorization(args):
    """
    Vectoriza los chunks generados por ChunkAI y guarda los vectores y sus IDs.
    """
    embedder: Embedder = Embedder()
    dim: int = embedder.model.get_sentence_embedding_dimension()
    vector_store_manager: VectorStoreManager = VectorStoreManager(dim)

    with open(f"{config.OUTPUT_DIR}/articulos_completos_chunked.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_vectors = []
    all_ids = []

    for article in articles:
        for chunk in article.get("chunks", []):
            chunk_id = chunk["chunk_id"]
            texto = chunk["texto"]
            try:
                vector = embedder.get_embedding(texto)
                vector_store_manager.add(np.expand_dims(vector, axis=0), [chunk_id])
                all_vectors.append(vector)
                all_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ Error vectorizando chunk '{chunk_id}': {e}")

    # Guardar vectores, IDs, índice
    vector_store_manager.save(f"{config.OUTPUT_DIR}/vector_store_chunks.index")
    np.save(f"{config.OUTPUT_DIR}/embeddings_chunks.npy", np.stack(all_vectors))
    with open(f"{config.OUTPUT_DIR}/ids_chunks.json", "w") as f:
        json.dump(all_ids, f)
    print("✅ Vectorización de chunks completada.")

def run_clusterization(args):
    """
    Clusteriza los resumenes de las noticias usando HDBSCAN.
    """
    vectors = np.load(f"{config.OUTPUT_DIR}/embeddings.npy")
    with open(f"{config.OUTPUT_DIR}/ids.json") as f:
        ids = json.load(f)
    with open(f"{config.OUTPUT_DIR}/articulos_completos_con_resumenes.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    meta = {str(a["id"]): a for a in meta_raw}

    engine = ClusterEngine()
    labels = engine.fit(vectors)

    # Guardar los IDs como JSON (para referencia)
    with open(f"{config.OUTPUT_DIR}/ids.json", "w", encoding="utf-8") as f_ids:
        json.dump(ids, f_ids)

    # Guardar las etiquetas de cluster como .npy
    np.save(f"{config.OUTPUT_DIR}/labels.npy", np.array(labels))

    # Guardar los pares ID - CLUSTER como .pkl para el paso de RAG
    output_cluster_pkl = f"{config.OUTPUT_DIR}/articulos_clusterizados.pkl"
    with open(output_cluster_pkl, "wb") as f:
        pickle.dump(
            [{"id": doc_id, "cluster": int(label)} for doc_id, label in zip(ids, labels)],
            f
        )

    print("✅ Archivos de clusterización guardados:")
    print(f"   - {config.OUTPUT_DIR}/ids.json")
    print(f"   - {config.OUTPUT_DIR}/labels.npy")
    print(f"   - {output_cluster_pkl}")

    df = to_dataframe(vectors, labels, ids, meta)
    interactive_scatter(df)

def run_chunk_clusterization(args):
    vectors = np.load(f"{config.OUTPUT_DIR}/embeddings_chunks.npy")
    with open(f"{config.OUTPUT_DIR}/ids_chunks.json") as f:
        ids = json.load(f)
    with open(f"{config.OUTPUT_DIR}/articulos_completos_chunked.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    # Mapear cada chunk_id a su texto original (para hover)
    meta = {}
    for art in meta_raw:
        for chunk in art.get("chunks", []):
            meta[chunk["chunk_id"]] = {
                "title": art["title"],
                "link": art.get("link", "#")
            }

    engine = ClusterEngine()
    labels = engine.fit(vectors)

    output_cluster_pkl = f"{config.OUTPUT_DIR}/articulos_clusterizados.pkl"
    with open(output_cluster_pkl, "wb") as f:
        pickle.dump(
            [{"id": doc_id, "cluster": int(label)} for doc_id, label in zip(ids, labels)],
            f
        )

    print("✅ Archivos de clusterización guardados:")
    print(f"   - {config.OUTPUT_DIR}/ids.json")
    print(f"   - {config.OUTPUT_DIR}/labels.npy")
    print(f"   - {output_cluster_pkl}")

    df = to_dataframe(vectors, labels, ids, meta)
    interactive_scatter(df)

def run_rag_classification(args):
    reconstructor = RAGReconstructor(
        json_articulos=f"{config.OUTPUT_DIR}/articulos_completos_chunked.json",
        clustering_pkl=f"{config.OUTPUT_DIR}/articulos_clusterizados.pkl"
    )
    clasificador = ImportanciaClassifier()
    pipeline_rag = RAGPipeline(reconstructor, clasificador)
    pipeline_rag.ejecutar()
    pipeline_rag.exportar()

def run_rag_refinamiento(args):
    """
    Usa GPT para refinar el resultado de RAG, separando múltiples artículos que hayan quedado agrupados
    dentro de un mismo cluster reconstruido.
    """
    from refinar_rag_gpt import ClusterLoader, GPTArticleSeparator, RAGPostProcessor
    from refinar_rag_gpt import guardar_top_clusters

    path_json = f"{config.OUTPUT_DIR}/noticias_reconstruidas_clasificadas.json"

    # Verificamos que exista el archivo antes de ejecutar
    import os
    if not os.path.exists(path_json):
        print("❌ El archivo 'noticias_reconstruidas_clasificadas.json' no existe.")
        print("💡 Corré primero: python pipeline_chunk.py rag_classification")
        return

    ruta_filtrada = f"{config.OUTPUT_DIR}/top_clusters.json"
    guardar_top_clusters(path_json, ruta_filtrada, n=2)
    loader = ClusterLoader(ruta_filtrada)
    separator = GPTArticleSeparator()
    processor = RAGPostProcessor(loader, separator)
    processor.ejecutar()
    processor.exportar(output_txt=f"{config.OUTPUT_DIR}/noticias_reorganizadas.txt")


def run_all(args):
    """
    - Corre todo el pipeline de scraping y análisis.
    """
    run_news(args)
    run_summarization(args)
    run_chunk_vectorization(args)
    run_clusterization(args)
    run_rag_classification(args)
    run_rag_refinamiento(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline project")
    sub = parser.add_subparsers(dest="command")

    # Caso news: corremos solo el scraper de noticias
    sub_news = sub.add_parser("news", help="scrapea las noticias")
    sub_news.set_defaults(func=run_news)

    # Caso summaries: corremos solo el summarizer de noticias
    sub_news = sub.add_parser(
        "summaries", help="resume las noticias previamente scrapeadas"
    )
    sub_news.set_defaults(func=run_summarization)

    # Caso summary_vectorization: corremos solo la vectorización de resúmenes
    sub_vectorization = sub.add_parser(
        "summary_vectorization", help="vectoriza los resúmenes de noticias"
    )
    sub_vectorization.set_defaults(func=run_chunk_vectorization)

    # Caso Clusterization: corremos solo la clusterización de resúmenes
    sub_clusterization = sub.add_parser(
        "clusterization", help="clusteriza los resúmenes de noticias"
    )
    sub_clusterization.set_defaults(func=run_clusterization)

    # Caso RAG: reconstrucción y clasificación
    sub_rag = sub.add_parser("rag_classification", help="Reconstruye texto por cluster y clasifica su importancia")
    sub_rag.set_defaults(func=run_rag_classification)

    # Caso all: corremos todo el pipeline
    sub_all = sub.add_parser("all", help="Corre todo el pipeline en orden")
    sub_all.set_defaults(func=run_all)

    # Caso chunk_vectorization
    sub_chunk_vec = sub.add_parser("chunk_vectorization", help="vectoriza los chunks de noticias")
    sub_chunk_vec.set_defaults(func=run_chunk_vectorization)

    # Caso chunk_clusterization
    sub_chunk_cluster = sub.add_parser("chunk_clusterization", help="clusteriza los chunks de noticias")
    sub_chunk_cluster.set_defaults(func=run_chunk_clusterization)

    # Caso refinamiento posterior a RAG con GPT
    sub_refine = sub.add_parser("rag_refine", help="Separa artículos mal agrupados dentro de clusters y reestructura el texto final")
    sub_refine.set_defaults(func=run_rag_refinamiento)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)
