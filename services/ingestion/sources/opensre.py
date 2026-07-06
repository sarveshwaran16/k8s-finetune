import requests
from bs4 import BeautifulSoup
from typing import Generator
import time

BASE_URL = "https://opensre.dev"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; k8s-rag-bot/1.0)"
}


def get_urls() -> list[str]:
    """Crawl opensre.dev index to find all doc links."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/") and len(href) > 1:
            full_url = BASE_URL + href
            if full_url not in urls:
                urls.append(full_url)
        elif href.startswith(BASE_URL) and href != BASE_URL:
            if href not in urls:
                urls.append(href)

    return urls


def parse_page(url: str) -> dict | None:
    """Fetch and parse a single opensre.dev page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return None

    text = main.get_text(separator="\n", strip=True)

    if len(text) < 200:
        return None

    return {
        "text": text,
        "metadata": {
            "source_url": url,
            "title": title,
            "source": "opensre",
        }
    }


def fetch_all(delay: float = 0.5) -> Generator[dict, None, None]:
    """Yield parsed pages from opensre.dev."""
    urls = get_urls()
    print(f"[opensre] Found {len(urls)} URLs to fetch")

    for i, url in enumerate(urls):
        page = parse_page(url)
        if page:
            yield page
        if i % 10 == 0:
            print(f"[opensre] Progress: {i}/{len(urls)}")
        time.sleep(delay)