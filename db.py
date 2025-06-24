import sqlite3


def init_db(db_path: str = "pipeline.db"):
    """
    Inicializa la base de datos SQLite creando las tablas necesarias.
    Si el archivo no existe, lo crea automáticamente.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Tabla articles
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        text TEXT NOT NULL
    );
    """
    )

    # 2. Tabla summaries
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS summaries (
        article_id INTEGER PRIMARY KEY,
        summary TEXT NOT NULL,
        FOREIGN KEY(article_id) REFERENCES articles(id)
    );
    """
    )

    # 3. Tabla vectors_meta
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS vectors_meta (
        article_id INTEGER PRIMARY KEY,
        vector_index INTEGER NOT NULL,
        FOREIGN KEY(article_id) REFERENCES articles(id)
    );
    """
    )

    # 4. Tabla clusters
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS clusters (
        article_id INTEGER PRIMARY KEY,
        cluster_label INTEGER NOT NULL,
        FOREIGN KEY(article_id) REFERENCES articles(id)
    );
    """
    )

    # 5. Tabla settings para rutas o configs
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """
    )

    conn.commit()
    conn.close()
    print(f"✅ Base de datos inicializada en {db_path}")
