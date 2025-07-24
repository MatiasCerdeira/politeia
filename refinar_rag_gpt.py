import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# Configurar API
client = OpenAI(api_key="
# Modelo para embeddings (rápido y económico)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === ETAPA 1: GUARDAR TOP CLUSTERS ===
def guardar_top_clusters(path_input_json, path_output_json="output_files/top_clusters.json", n=2):
    Path(path_output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(path_input_json, encoding="utf-8") as f:
        clusters = json.load(f)

    clusters_filtrados = [c for c in clusters if c["cluster_id"] != -1]
    top_clusters = sorted(clusters_filtrados, key=lambda c: len(c["articulos_incluidos"]), reverse=True)[:n]

    with open(path_output_json, "w", encoding="utf-8") as f_out:
        json.dump(top_clusters, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Se guardaron {n} clusters en {path_output_json}")
    return top_clusters

# === ETAPA 2: MAPA DE CHUNKS ===
def construir_mapa_chunks(path_chunks_json="output_files/articulos_completos_chunked.json"):
    with open(path_chunks_json, encoding="utf-8") as f:
        data = json.load(f)

    chunks_map = {}
    for art in data:
        for chunk in art.get("chunks", []):
            chunks_map[chunk["chunk_id"]] = chunk["texto"]
    return chunks_map

# === ETAPA 3: GPT PARA REFINAR ARTÍCULOS ===
def refinar_articulo(texto, cluster_id, articulo_id, model="gpt-4o-mini"):
    prompt = (
        f"Unificá los siguientes fragmentos en un artículo periodístico coherente y bien redactado, en español, sin repeticiones:\n\n{texto}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Sos un experto en redacción periodística."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR en cluster {cluster_id}, artículo {articulo_id}]: {e}"

# === ETAPA 4: REFINAMIENTO SEMÁNTICO (DETECCIÓN DE DUPLICADOS) ===
def agrupar_similares(articulos, threshold=0.85):
    textos = [a["texto_refinado"] for a in articulos]
    embeddings = embedder.encode(textos, convert_to_tensor=True)
    grupos = []
    usados = set()

    for i in range(len(textos)):
        if i in usados:
            continue
        grupo = [i]
        for j in range(i+1, len(textos)):
            if j not in usados and util.cos_sim(embeddings[i], embeddings[j]) >= threshold:
                grupo.append(j)
                usados.add(j)
        usados.add(i)
        grupos.append(grupo)
    return grupos

def fusionar_grupo(grupo, articulos):
    textos_unidos = "\n".join([articulos[i]["texto_refinado"] for i in grupo])
    return refinar_articulo(textos_unidos, "fusion", "grupo")

# === MAIN PIPELINE ===
def run_rag_refinamiento():
    ruta_entrada = "output_files/noticias_reconstruidas_clasificadas.json"
    ruta_filtrada = "output_files/top_clusters.json"
    ruta_chunks = "output_files/articulos_completos_chunked.json"
    ruta_salida = "output_files/noticias_reorganizadas_final.txt"

    # Paso 1: Extraer clusters
    top_clusters = guardar_top_clusters(ruta_entrada, ruta_filtrada, n=2)

    # Paso 2: Crear mapa chunks
    chunks_map = construir_mapa_chunks(ruta_chunks)

    # Paso 3: Generar artículos preliminares
    articulos_preliminares = []
    for cluster in tqdm(top_clusters, desc="Procesando clusters"):
        chunks_por_articulo = defaultdict(list)
        for id_chunk in cluster["articulos_incluidos"]:
            art_id, _ = id_chunk.split("_p")
            chunks_por_articulo[art_id].append(id_chunk)

        for art_id, ids_chunks in chunks_por_articulo.items():
            texto = "\n".join([chunks_map.get(cid, "") for cid in ids_chunks])
            texto_refinado = refinar_articulo(texto, cluster["cluster_id"], art_id)
            articulos_preliminares.append({
                "cluster_id": cluster["cluster_id"],
                "articulo_id": art_id,
                "texto_refinado": texto_refinado
            })

    # Paso 4: Detectar artículos muy similares y fusionarlos
    print("🔍 Detectando artículos duplicados...")
    grupos = agrupar_similares(articulos_preliminares)
    articulos_finales = []
    for grupo in grupos:
        if len(grupo) > 1:
            texto_fusionado = fusionar_grupo(grupo, articulos_preliminares)
            articulos_finales.append(texto_fusionado)
        else:
            articulos_finales.append(articulos_preliminares[grupo[0]]["texto_refinado"])

    # Paso 5: Guardar salida
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for i, art in enumerate(articulos_finales, 1):
            f.write(f"--- ARTÍCULO {i} ---\n{art}\n\n")
    print(f"✅ Archivo final guardado en {ruta_salida} con {len(articulos_finales)} artículos sin duplicados.")
