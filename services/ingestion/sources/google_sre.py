import requests
from bs4 import BeautifulSoup
from typing import Generator
import time

SOURCES = {
    "sre_book": "https://sre.google/sre-book/table-of-contents/",
    "sre_workbook": "https://sre.google/workbook/table-of-contents/",
}

BASE_URL = "https://sre.google"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; k8s-rag-bot/1.0)"
}


def get_urls(toc_url: str, source_key: str) -> list[dict]:
    """Crawl a SRE table of contents page and return all chapter links."""
    resp = requests.get(toc_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/") and len(href) > 1:
            full_url = BASE_URL + href
            if full_url not in [u["url"] for u in urls]:
                urls.append({"url": full_url, "source": source_key})
        elif href.startswith(BASE_URL) and href != toc_url:
            if href not in [u["url"] for u in urls]:
                urls.append({"url": href, "source": source_key})

    return urls


def parse_page(url: str, source_key: str) -> dict | None:
    """Fetch and parse a single SRE book/workbook chapter."""
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
            "source": source_key,
        }
    }


def fetch_all(delay: float = 0.5) -> Generator[dict, None, None]:
    """Yield parsed chapters from both SRE book and workbook."""
    for source_key, toc_url in SOURCES.items():
        urls = get_urls(toc_url, source_key)
        print(f"[google_sre] {source_key} — found {len(urls)} chapters")

        for i, entry in enumerate(urls):
            page = parse_page(entry["url"], entry["source"])
            if page:
                yield page
            if i % 10 == 0:
                print(f"[google_sre] {source_key} progress: {i}/{len(urls)}")
            time.sleep(delay)