import requests
from bs4 import BeautifulSoup
from typing import Generator
import time

BASE_URL = "https://runbooks.prometheus-operator.dev"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; k8s-rag-bot/1.0)"
}

RELEVANT_RUNBOOKS = [
    "/runbooks/",
]


def get_runbook_urls() -> list[str]:
    """Crawl the index page to find all runbook links."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/runbooks/" in href and href.startswith("http"):
            if href not in urls:
                urls.append(href)

    return urls


def parse_page(url: str) -> dict | None:
    """Fetch and parse a single runbook page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    title_tag = soup.find("h1")
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
            "source": "prometheus_runbooks",
        }
    }


def fetch_all(delay: float = 0.5) -> Generator[dict, None, None]:
    """Yield parsed pages from all runbook URLs."""
    urls = get_runbook_urls()
    print(f"[prometheus_runbooks] Found {len(urls)} URLs to fetch")

    for i, url in enumerate(urls):
        page = parse_page(url)
        if page:
            yield page
        if i % 10 == 0:
            print(f"[prometheus_runbooks] Progress: {i}/{len(urls)}")
        time.sleep(delay)