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

# Minimum length for a section to stand alone as a chunk. Sections shorter
# than this get merged into the next one instead of being dropped, so we
# don't lose short-but-relevant content (e.g. a one-paragraph "Overview").
MIN_SECTION_CHARS = 300

# Heuristic keyword -> failure-class tags, so later generation steps (Task 2)
# can target specific troubleshooting scenarios instead of generic trivia.
FAILURE_CLASS_KEYWORDS = {
    "OOMKilled": ["oomkilled", "out of memory", "memory limit", "137"],
    "CrashLoopBackOff": ["crashloopbackoff", "crash loop", "restart", "liveness probe"],
    "ImagePullBackOff": ["imagepullbackoff", "errimagepull", "image pull", "pull image"],
    "NodeNotReady": ["nodenotready", "node not ready", "node pressure", "kubelet"],
    "RBAC": ["rbac", "role binding", "rolebinding", "clusterrole", "serviceaccount", "forbidden"],
    "Ingress": ["ingress", "load balancer", "503", "service selector"],
    "PVC": ["persistentvolumeclaim", "pvc", "storage class", "volume binding"],
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


def _tag_failure_classes(text: str) -> list[str]:
    """Heuristically tag a chunk with the failure classes it seems to cover."""
    lowered = text.lower()
    tags = []
    for label, keywords in FAILURE_CLASS_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            tags.append(label)
    return tags


def _split_into_sections(main, page_title: str) -> list[dict]:
    """
    Split a page's main content into self-contained sections based on
    heading structure (h2/h3), instead of returning the whole page as one
    blob. Each section keeps everything between one heading and the next
    heading of the same or higher level — a complete logical unit, not an
    arbitrary character-count slice.
    """
    # Collect top-level content nodes in document order.
    nodes = list(main.find_all(["h1", "h2", "h3", "p", "ul", "ol", "pre", "table"], recursive=True))

    sections = []
    current_heading = page_title
    current_anchor = None
    current_parts = []

    def flush():
        if not current_parts:
            return
        text = "\n".join(current_parts).strip()
        if text:
            sections.append({"heading": current_heading, "anchor": current_anchor, "text": text})

    for node in nodes:
        if node.name in ("h1", "h2", "h3"):
            # Starting a new section — flush whatever we've accumulated so far.
            flush()
            current_parts = []
            current_heading = node.get_text(strip=True) or page_title
            current_anchor = node.get("id")
        else:
            text = node.get_text(separator="\n", strip=True)
            if text:
                current_parts.append(text)

    flush()

    # Merge short sections forward into the next one, so a one-line
    # "Overview" heading doesn't become its own tiny, low-value chunk.
    merged = []
    carry = None
    for sec in sections:
        combined_text = (carry["text"] + "\n\n" + sec["text"]) if carry else sec["text"]
        combined = {
            "heading": carry["heading"] if carry else sec["heading"],
            "anchor": carry["anchor"] if carry else sec["anchor"],
            "text": combined_text,
        }
        if len(combined_text) < MIN_SECTION_CHARS:
            carry = combined
        else:
            merged.append(combined)
            carry = None
    if carry and len(carry["text"]) >= 50:  # keep a trailing small leftover rather than losing it
        merged.append(carry)

    return merged


def parse_page(url: str) -> list[dict]:
    """
    Fetch a single K8s doc page and split it into self-contained section
    chunks (not the whole page, not a character-count slice).
    Returns a list of chunk dicts (may be empty if the page can't be parsed).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Get page-level title
    title_tag = soup.find("h1")
    page_title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-2]

    # Get main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return []

    sections = _split_into_sections(main, page_title)

    chunks = []
    for sec in sections:
        text = sec["text"]
        if len(text) < 200:
            continue  # skip near-empty leftovers

        source_url = url if not sec["anchor"] else f"{url}#{sec['anchor']}"
        chunk_title = page_title if sec["heading"] == page_title else f"{page_title} — {sec['heading']}"

        chunks.append({
            "text": text,
            "metadata": {
                "source_url": source_url,
                "title": chunk_title,
                "source": "kubernetes_docs",
                "failure_classes": _tag_failure_classes(text),
            }
        })

    return chunks


def fetch_all(delay: float = 0.5) -> Generator[dict, None, None]:
    """Yield parsed section-chunks from all relevant K8s doc URLs."""
    urls = get_doc_urls()
    print(f"[k8s_docs] Found {len(urls)} URLs to fetch")

    total_chunks = 0
    for i, url in enumerate(urls):
        chunks = parse_page(url)
        for chunk in chunks:
            yield chunk
            total_chunks += 1
        if i % 10 == 0:
            print(f"[k8s_docs] Progress: {i}/{len(urls)} pages, {total_chunks} chunks so far")
        time.sleep(delay)

    print(f"[k8s_docs] Done — {total_chunks} chunks from {len(urls)} pages")