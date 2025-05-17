#!/usr/bin/env python3
"""
Script: harvest_it_links.py

This script automates harvesting of candidate IT research paper URLs by:
  1. Querying Semantic Scholar API for IT-related keywords
  2. Fetching recent entries from specified arXiv categories
  3. Pulling paper links from RSS/Atom feeds of top IT venues

All harvested URLs are de-duplicated and appended to `new_it_links.txt`.

Requirements:
  - requests
  - feedparser

Usage:
  python harvest_it_links.py
"""
import time
import requests
import feedparser
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

# Configuration
KEYWORDS = [
    "cloud infrastructure",
    "edge computing",
    "network virtualization",
    "AIOps",
    "infrastructure as code",
]
SEMANTIC_SCHOLAR_SEARCH =     "https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=url"

ARXIV_CATEGORIES = ["cs.NI", "cs.DC", "cs.SD"]
ARXIV_MAX = 100  # number of results per category
ARXIV_API = "http://export.arxiv.org/api/query?search_query=cat:{cat}&max_results={max_results}"

# RSS/Atom feeds for key IT venues
RSS_FEEDS = [
    # Example: IEEE INFOCOM
    "https://ieeexplore.ieee.org/rss/TOC66.XML",
    # USENIX ATC
    "https://www.usenix.org/conference/atc23/atc23-archive.xml",
    # ACM SOSP proceedings
    "https://dl.acm.org/action/showFeed?type=etoc&feed=rss&jc=sosp",
]

OUTPUT_FILE = "hl.txt"

def harvest_semantic_scholar():
    urls = set()
    for kw in KEYWORDS:
        q = quote_plus(kw)
        url = SEMANTIC_SCHOLAR_SEARCH.format(query=q, limit=100)
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            for paper in data:
                u = paper.get("url")
                if u:
                    urls.add(u)
        except Exception as e:
            print(f"Semantic Scholar error for '{kw}': {e}")
        time.sleep(1)
    return urls

def harvest_arxiv():
    urls = set()
    for cat in ARXIV_CATEGORIES:
        api = ARXIV_API.format(cat=cat, max_results=ARXIV_MAX)
        try:
            r = requests.get(api, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                link = entry.find("atom:id", ns).text
                if link:
                    urls.add(link)
        except Exception as e:
            print(f"arXiv error for {cat}: {e}")
        time.sleep(1)
    return urls

def harvest_rss_feeds():
    urls = set()
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for entry in d.entries:
                link = entry.get("link")
                if link:
                    urls.add(link)
        except Exception as e:
            print(f"RSS parse error for {feed}: {e}")
        time.sleep(1)
    return urls

def load_existing():
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def save_urls(urls):
    existing = load_existing()
    all_urls = existing.union(urls)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for u in sorted(all_urls):
            f.write(u + "\n")
    print(f"Total candidate links saved: {len(all_urls)}")

def main():
    print("Harvesting Semantic Scholar...")
    ss_urls = harvest_semantic_scholar()

    print("Harvesting arXiv categories...")
    ax_urls = harvest_arxiv()

    print("Harvesting RSS feeds...")
    rss_urls = harvest_rss_feeds()

    new_urls = ss_urls.union(ax_urls).union(rss_urls)
    print(f"Found {len(new_urls)} unique candidate URLs")

    save_urls(new_urls)

if __name__ == '__main__':
    main()
