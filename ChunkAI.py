from google import genai
from google.genai import types

GEMINI_API_KEY = "AIzaSyDaQI34pII1zHQ5H7gwY1aVzsIeoZ09lh4"

class ChunkAI:
    """
    Clase que utiliza Gemini 2.0 Flash Lite para dividir artículos en chunks semánticamente coherentes.
    Cada chunk tiene entre 500 y 1000 caracteres, con solapamiento si es necesario.
    """

    def __init__(self):
        print(f"🔐 GEMINI KEY: {GEMINI_API_KEY}")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def chunk_article(self, title: str, text: str) -> list[str]:
        """
        Divide un artículo en fragmentos con sentido, de entre 500 y 1000 caracteres, usando IA.

        Args:
            title (str): El título de la noticia.
            text (str): El texto completo de la noticia.

        Returns:
            list[str]: Lista de chunks coherentes del artículo.
        """
        prompt = (
            f"Dividí la siguiente noticia en fragmentos (chunks) de entre 500 y 1000 caracteres, "
            f"con sentido completo (no cortes oraciones a la mitad). Si es posible, solapá 1 o 2 frases entre chunks. "
            f"Retorná los chunks como una lista en texto plano, separados por el token <CHUNK>.\n\n"
            f"Título: {title}\n\nTexto:\n{text}"
        )

        config = types.GenerateContentConfig(max_output_tokens=1024)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt],
            config=config,
        )

        raw_output = response.text

        # Dividir los chunks usando el token <CHUNK>
        chunks = [chunk.strip() for chunk in raw_output.split("<CHUNK>") if len(chunk.strip()) >= 400]

        return chunks
