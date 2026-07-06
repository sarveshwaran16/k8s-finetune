import requests
from bs4 import BeautifulSoup
from typing import Generator
import time

BASE_URL = "https://k8s.af"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; k8s-rag-bot/1.0)"
}


def get_failure_urls() -> list[str]:
    """Crawl the k8s.af index to find all failure story links."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # k8s.af links out to external failure stories
        if href.startswith("http") and BASE_URL not in href:
            if href not in urls:
                urls.append(href)

    return urls


def parse_page(url: str) -> dict | None:
    """Fetch and parse a single failure story page."""
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
    title = title_tag.get_text(strip=True) if title_tag else url

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
            "source": "k8s_failures",
        }
    }


def fetch_all(delay: float = 1.0) -> Generator[dict, None, None]:
    """Yield parsed failure stories. Slower delay — external sites."""
    urls = get_failure_urls()
    print(f"[k8s_failures] Found {len(urls)} URLs to fetch")

    for i, url in enumerate(urls):
        page = parse_page(url)
        if page:
            yield page
        if i % 10 == 0:
            print(f"[k8s_failures] Progress: {i}/{len(urls)}")
        time.sleep(delay)