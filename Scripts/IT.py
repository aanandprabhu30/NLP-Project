# -*- coding: utf-8 -*-
"""download_it_24_v2.py

Fetch 24 verified core‑IT research papers (title + full abstract) and save
them to 'it_core_additional_24.csv' in the current directory.

Dependencies:
    pip install requests beautifulsoup4

Run:
    python download_it_24_v2.py
"""

import csv
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = [
    "https://doi.org/10.1186/s13677-023-00471-1",
    "https://doi.org/10.1186/s13677-023-00445-3",
    "https://doi.org/10.1186/s13677-023-00453-3",
    "https://doi.org/10.1186/s13677-023-00428-4",
    "https://doi.org/10.1186/s13677-023-00501-y",
    "https://doi.org/10.1186/s13677-023-00452-4",
    "https://doi.org/10.1186/s13677-025-00737-w",
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

    # Title
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "N/A"

    # Abstract
    abs_tag = (
        soup.find(id=re.compile(r"^Abs", re.I))
        or soup.find("section", {"data-title": re.compile("Abstract", re.I)})
        or soup.find("div", class_=re.compile("abstract", re.I))
    )
    if abs_tag:
        abstract = abs_tag.get_text(" ", strip=True)
    else:
        meta = soup.find("meta", {"name": "dc.Description"})
        abstract = meta["content"].strip() if meta and meta.get("content") else "N/A"

    return title, abstract


def main() -> None:
    rows = []
    for url in URLS:
        print(f"→ {url}")
        html = fetch_html(url)
        if not html:
            continue
        title, abstract = parse_title_abstract(html)
        rows.append({"Title": title, "Abstract": abstract, "Link": url})

    if not rows:
        sys.exit("❌ No papers fetched successfully.")

    csv_path = Path("it_core_additional_24.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Title", "Abstract", "Link"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {csv_path.resolve()} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
