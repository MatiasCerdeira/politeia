from google import genai
from google.genai import types
from config import GEMINI_API_KEY


class Summarizer:
    """
    Clase para abstraer la logica de resumir text usando Gemini_2.0_flash.
    Attributes:
        client (genai.Client): El cliente de Google GenAI inicializado con la clave de API.
    """

    def __init__(self):
        print(f"GEMINI KEY: {GEMINI_API_KEY}")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def summarize_text(self, title: str, text: str) -> str:
        """
        Genera un resumen de un texto dado (una noticia) utilizando el modelo Gemini 2.0 Flash Lite.

        Args:
            title (str): El título de la noticia.
            text (str): El texto completo de la noticia.
        Returns:
            str: El resumen generado por el modelo.

        """
        prompt: str = (
            f"Resumi la siguiente noticia en español en 150 a 200 palabras aproximadamente. Retorna solo el resumen en texto plano, no incluyas nada mas:\n + Titulo: {title}\n + Texto: {text}"
        )

        config = types.GenerateContentConfig(max_output_tokens=350)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt],
            config=config,
        )
        return response.text
