import requests
from bs4 import BeautifulSoup
from typing import Generator
import xml.etree.ElementTree as ET
import time

SITEMAP_URL = "https://kubernetes.io/en/sitemap.xml"
BASE_URL = "https://kubernetes.io"

RELEVANT_PATHS = [
    "/docs/concepts/",
    "/docs/tasks/",
    "/docs/tutorials/",
    "/docs/reference/",
    "/docs/setup/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; k8s-rag-bot/1.0)"
}


def get_doc_urls() -> list[str]:
    """Fetch all /docs/ URLs from the Kubernetes sitemap."""
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for loc in root.findall(".//sm:loc", ns):
        url = loc.text.strip()
        if any(url.startswith(BASE_URL + path) for path in RELEVANT_PATHS):
            urls.append(url)

    return urls


def parse_page(url: str) -> dict | None:
    """Fetch and parse a single K8s doc page. Returns clean text + metadata."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Get title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-2]

    # Get main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return None

    text = main.get_text(separator="\n", strip=True)

    # Skip very short pages
    if len(text) < 200:
        return None

    return {
        "text": text,
        "metadata": {
            "source_url": url,
            "title": title,
            "source": "kubernetes_docs",
        }
    }


def fetch_all(delay: float = 0.5) -> Generator[dict, None, None]:
    """Yield parsed pages from all relevant K8s doc URLs."""
    urls = get_doc_urls()
    print(f"[k8s_docs] Found {len(urls)} URLs to fetch")

    for i, url in enumerate(urls):
        page = parse_page(url)
        if page:
            yield page
        if i % 10 == 0:
            print(f"[k8s_docs] Progress: {i}/{len(urls)}")
        time.sleep(delay)