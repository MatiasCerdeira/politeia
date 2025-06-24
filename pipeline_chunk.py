import argparse
import config
import json
import numpy as np
import faiss

from news_scraper import NewsScraper
from news_scraper import CompleteArticle
from ChunkAI import ChunkAI
from embedder import Embedder
from vector_store_manager import VectorStoreManager
from cluster_engine import ClusterEngine
from sklearn.decomposition import PCA
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
            chunks = chunker.chunk_article(article["title"], article["text"])
            article["chunks"] = chunks
        except Exception as e:
            print(f"❌ Error al chunkearear '{article['title'][:30]}': {e}")
            article["chunks"] = []

    save_to_json(articles, output_path)
    print("✅ Chunks guardados correctamente.")


# Se corre con "make summary_vectorization"
def run_summary_vectorization(args):
    """
    Vectoriza y guarda en una base de datos vectorial los resumenes de los articulos.

    Guarda los vectores en output_files/vector_store.index.
    Guarda los embeddings en output_files/embeddings.npy
    Guarda los IDs de los articulos en output_files/ids.json.
    """
    embedder: Embedder = Embedder()

    # Me fijo las dimensiones de los embeddings
    dim: int = embedder.model.get_sentence_embedding_dimension()

    vector_store_manager: VectorStoreManager = VectorStoreManager(dim)

    with open(f"{config.OUTPUT_DIR}/articulos_completos_con_resumenes.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_vectors = []
    all_ids = []

    for i, article in enumerate(articles):
        print(
            f"🔍 Vectorizando resumen {i+1}/{len(articles)}: {article['title'][:50]}..."
        )
        try:
            vector: np.ndarray = embedder.get_embedding(article["summary"])
            vector_store_manager.add(np.expand_dims(vector, axis=0), [article["id"]])
            all_vectors.append(vector)
            all_ids.append(article["id"])
        except Exception as e:
            print(f"❌ Error al vectorizar el resumen '{article['title'][:30]}': {e}")

    # 1) Faiss Index
    vector_store_manager.save(f"{config.OUTPUT_DIR}/vector_store.index")
    print("✅ Vectores de resúmenes guardados correctamente.")

    # 2) Raw Embeddings
    vectors = np.stack(all_vectors)
    np.save(f"{config.OUTPUT_DIR}/embeddings.npy", vectors)

    # 3) IDs
    with open(f"{config.OUTPUT_DIR}/ids.json", "w") as f:
        json.dump(all_ids, f)
    print("✅ Embeddings e IDs guardados.")


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

    df = to_dataframe(vectors, labels, ids, meta)
    interactive_scatter(df)


def run_all(args):
    """
    - Corre todo el pipeline de scraping y análisis.
    """
    run_news(args)
    run_summarization(args)
    run_summary_vectorization(args)
    run_clusterization(args)


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
    sub_vectorization.set_defaults(func=run_summary_vectorization)

    # Caso Clusterization: corremos solo la clusterización de resúmenes
    sub_clusterization = sub.add_parser(
        "clusterization", help="clusteriza los resúmenes de noticias"
    )
    sub_clusterization.set_defaults(func=run_clusterization)

    # Caso all: corremos todo el pipeline
    sub_all = sub.add_parser("all", help="Corre todo el pipeline en orden")
    sub_all.set_defaults(func=run_all)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)
