"""HTML cleaning and structured table extraction."""
from bs4 import BeautifulSoup


def clean_html_content(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for elem in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        elem.decompose()
    return " ".join(soup.get_text().split())
