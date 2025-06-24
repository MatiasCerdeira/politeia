# config.py
import os
from dotenv import load_dotenv

load_dotenv()


DEFAULT_OUTPUT_DIR = "output_files"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

RSS_FEEDS = [
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.clarin.com/rss/politica/",
    "https://www.clarin.com/rss/economia/",
    "https://www.perfil.com/feed/politica",
    "https://www.pagina12.com.ar/rss/secciones/economia/notas",
    "https://www.pagina12.com.ar/rss/secciones/sociedad/notas",
    "https://www.pagina12.com.ar/rss/secciones/el-pais/notas",
]


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
