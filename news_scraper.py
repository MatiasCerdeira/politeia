import feedparser
from dataclasses import dataclass
from feedparser import FeedParserDict
from newspaper import Article
import time
import requests
import urllib.parse
import json


@dataclass
class FeedEntry:
    """
    Representa una entrada basica de un feed RSS (o de otro metodo), con titulo, link y fecha.

    Attributes:
        title (str): Título del artículo.
        link (str): URL original del artículo.
        date (str): Fecha de publicación del artículo.
        category (str): Categoría del artículo (opcional, puede ser "Sin categoría").
    """

    title: str
    link: str
    date: str
    category: str


@dataclass
class CompleteArticle:
    """
    Representa un articulo completo con metadata y texto extraido.

    Attributes:
        id (int): ID del artículo.
        title (str): Título del artículo.
        link (str): URL original del artículo.
        date (str): Fecha de publicación.
        text (str): Texto completo del artículo.
    """

    id: int
    title: str
    link: str
    date: str
    text: str
    category: str


class NewsScraper:
    """
    Scraper de noticias a partir de múltiples RSS feeds.

    Permite:
    - Obtener entradas básicas de cada feed (título, link, fecha).
    - Descargar el texto completo de los artículos usando newspaper3k.

    Uso:
        scraper = NewsScraper(feeds)
        articulos = scraper.scrape()
    """

    def __init__(self, RSS_feeds: list[str]):
        self.feeds = RSS_feeds

    # ------------------------------------- Public -------------------------------------
    def scrape(self) -> list[CompleteArticle]:
        """
        Scrapea Infobae y los RSS y devuelve una lista de artículos completos.

        Args:
            Ninguno.

        Returns:
            list[CompleteArticle]: Lista de artículos completos con ID, title, link, date y text.
        """
        print("🔍 Iniciando scraping de noticias...")

        # Paso 1: Obtener entradas de Infobae (feed no RSS)
        entradas: list[FeedEntry] = self._scrape_infobae()

        # Paso 2: Obtener todas las entradas de los feeds RSS
        entradas.extend(self._fetch_all_feeds())

        # Paso 3: Limpiar las entradas para quedarnos solo con categorias relevantes
        entradas = self._clean_entries(entradas)

        # Paso 4: Descargar el texto completo de cada artículo
        articulos_completos: list[CompleteArticle] = self._download_full_articles(
            entradas
        )

        return articulos_completos

    # ------------------------------------ Private ------------------------------------

    def _fetch_all_feeds(self) -> list[FeedEntry]:
        """
        Recorre cada RSS feed y genera una lista de entradas básicas (con las url pero sin el texto completo).

        Returns:
            list[FeedEntry]: Lista de entradas con título, link y fecha y categoria.
        """
        all_entries: list[FeedEntry] = []
        for url in self.feeds:
            feed: FeedParserDict = feedparser.parse(url)
            for entry in feed.entries:
                all_entries.append(
                    FeedEntry(
                        title=entry.title,
                        link=entry.link,
                        date=entry.get("published", entry.get("updated", "")),
                        category=entry.get(
                            "category", "Sin categoría"
                        ),  # Muchas entradas no tendran categoria
                    )
                )
        print(f"🔍 Encontré {len(all_entries)} entradas en total.\n")
        return all_entries

    def _clean_entries(self, entries: list[FeedEntry]) -> list[FeedEntry]:
        """
        Toma una lista de entradas RSS y filtra las que no queremos procesar.
        Se fija la categoria de cada entrada y descarta entradas en categorias que no nos interesan.

        Observación:
        Por ahora esto solo existe para limpiar a La Nacion. Los otros medios tienen feeds RSS dedicados a politica, por lo que no es necesario limpiarlos.

        Args:
            entries (list[FeedEntry]): Lista de entradas RSS con titulo, link, fecha y categoria

        Returns:
            list[FeedEntry]: Lista de entradas RSS de las categorias que queremos procesar.
        """

        entries = [
            entry
            for entry in entries
            if entry.category
            in [
                "Política",
                "Economía",
                "Sociedad",
                "Sin categoría",
            ]
        ]
        print(
            f"🔍 Quedaron {len(entries)} entradas después de filtrar por categorías.\n"
        )
        return entries

    def _scrape_infobae(self):
        """
        Scrapea Infobae utilizando su API oculta. Scrapea la sección de Política.

        Returns:
            list[FeedEntry]: Lista de entradas con título, link y fecha de la sección Política de Infobae.
        """
        payload = {
            "feedLimit": 50,
            "feedOffset": 0,
            "feedOrder": "display_date:desc",
            "feedQuery": "",
            "feedSections": "/politica",
            "feedSectionsToExclude": "",
            "feedSlugAuthor": "",
            "feedSlugAuthorToExclude": "",
            "feedTags": "",
            "feedTagsToExclude": "",
        }
        url = "https://www.infobae.com/pf/api/v3/content/fetch/content-feed"
        params = {
            "query": json.dumps(payload),
            "d": "3304",
            "mxId": "00000000",
            "_website": "infobae",
        }

        resp = requests.get(url, params=params)
        data = resp.json()
        feedList: list[FeedEntry] = []
        for art in data["content_elements"]:
            raw_link: str = art["website_url"].lstrip(
                "/"
            )  # Retorna el link sin el dominio
            base: str = "https://www.infobae.com/"
            full_link: str = base + raw_link

            item: FeedEntry = FeedEntry(
                title=art["headlines"]["basic"],
                link=full_link,
                date=art["last_updated_date"],
                category="Política",
            )
            feedList.append(item)
        return feedList

    def _download_full_articles(
        self, entradas: list[FeedEntry]
    ) -> list[CompleteArticle]:
        """
        Descarga el texto completo de una lista de entradas RSS usando newspaper3k.

        Args:
            entradas (list[FeedEntry]): Lista de entradas con título, link y fecha.

        Returns:
            list[CompleteArticle]: Lista de artículos completos con ID, título, link,
            fecha y texto extraído.
        """
        total: int = len(entradas)
        resultados: list[CompleteArticle] = (
            []
        )  # lista con texto y metadata de todos los articulos

        for idx, ent in enumerate(entradas, start=1):
            link = ent.link
            title = ent.title
            date = ent.date
            category = ent.category

            print(f"🔄 [{idx}/{total}] Bajando nota: {title[:50]}...")

            try:
                art: Article = Article(link, language="es")
                art.download()
                art.parse()
                text: str = art.text  # Texto completo del artículo

                resultados.append(
                    CompleteArticle(
                        id=idx,
                        title=title,
                        link=link,
                        date=date,
                        text=text,
                        category=category,
                    )
                )

            except Exception as e:
                print(f"   ❌ ERROR bajando {link}: {e}")

            # Pequeña pausa para no reventar el servidor de cada medio
            time.sleep(1)

        print(f"\n✅ Listo: bajé {len(resultados)} artículos completos.")

        return resultados
