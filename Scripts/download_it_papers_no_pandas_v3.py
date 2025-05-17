"""
download_it_papers_no_pandas.py

Fetches titles & full abstracts for 42 core‑IT research papers from SpringerOpen
(Journal of Cloud Computing) and writes them to `it_core_additional_42.csv`.

Dependencies:
    pip install requests beautifulsoup4

Usage:
    python download_it_papers_no_pandas.py
"""

import csv
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = [
    "https://doi.org/10.1186/s13677-025-00748-7",
    "https://doi.org/10.1186/s13677-025-00741-0",
    "https://doi.org/10.1186/s13677-025-00740-1",
    "https://doi.org/10.1186/s13677-025-00739-8",
    "https://doi.org/10.1186/s13677-025-00743-y",
    "https://doi.org/10.1186/s13677-025-00733-0",
    "https://doi.org/10.1186/s13677-024-00724-7",
    "https://doi.org/10.1186/s13677-025-00729-w",
    "https://doi.org/10.1186/s13677-025-00734-z",
    "https://doi.org/10.1186/s13677-025-00735-y",
    "https://doi.org/10.1186/s13677-025-00749-6",
    "https://doi.org/10.1186/s13677-025-00742-x",
    "https://doi.org/10.1186/s13677-025-00738-9",
    "https://doi.org/10.1186/s13677-025-00730-9",
    "https://doi.org/10.1186/s13677-025-00744-x",
    "https://doi.org/10.1186/s13677-025-00728-x",
    "https://doi.org/10.1186/s13677-024-00710-1",
    "https://doi.org/10.1186/s13677-024-00705-y",
    "https://doi.org/10.1186/s13677-024-00701-2",
    "https://doi.org/10.1186/s13677-024-00700-3",
    "https://doi.org/10.1186/s13677-024-00691-9",
    "https://doi.org/10.1186/s13677-024-00698-x",
    "https://doi.org/10.1186/s13677-024-00690-0",
    "https://doi.org/10.1186/s13677-024-00692-8",
    "https://doi.org/10.1186/s13677-024-00695-0",
    "https://doi.org/10.1186/s13677-024-00694-1",
    "https://doi.org/10.1186/s13677-024-00693-2",
    "https://doi.org/10.1186/s13677-024-00688-6",
    "https://doi.org/10.1186/s13677-024-00687-7",
    "https://doi.org/10.1186/s13677-024-00686-8",
    "https://doi.org/10.1186/s13677-024-00685-9",
    "https://doi.org/10.1186/s13677-024-00684-0",
    "https://doi.org/10.1186/s13677-024-00683-1",
    "https://doi.org/10.1186/s13677-024-00682-2",
    "https://doi.org/10.1186/s13677-024-00681-3",
    "https://doi.org/10.1186/s13677-024-00679-7",
    "https://doi.org/10.1186/s13677-024-00678-8",
    "https://doi.org/10.1186/s13677-024-00677-9",
    "https://doi.org/10.1186/s13677-024-00597-w",
    "https://doi.org/10.1186/s13677-024-00596-x",
    "https://doi.org/10.1186/s13677-024-00595-y",
    "https://doi.org/10.1186/s13677-024-00594-z",
    "https://doi.org/10.1186/s13677-024-00593-0",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (AbstractFetcher/1.0)"}


def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"[!] Failed {url}: {exc}", file=sys.stderr)
        return None


def parse_title_abstract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else "N/A"

    abs_tag = soup.find(id=re.compile(r"^Abs")) or soup.find("section", {"data-title": "Abstract"})
    if abs_tag:
        abstract = abs_tag.get_text(" ", strip=True)
    else:
        meta = soup.find("meta", {"name": "dc.Description"})
        abstract = meta["content"].strip() if meta and meta.get("content") else "N/A"

    return title, abstract


def main() -> None:
    csv_path = Path("it_core_additional_42.csv").resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Title", "Abstract", "Link"])

        for url in URLS:
            print(f"→ {url}")
            html = fetch_html(url)
            if not html:
                continue
            title, abstract = parse_title_abstract(html)
            writer.writerow([title, abstract, url])

    print(f"\n✅ Saved {csv_path}")


if __name__ == "__main__":
    main()
