import requests
import time
import xml.etree.ElementTree as ET
import csv
import random
from collections import defaultdict

# Subfield to arXiv categories (primary and optional backups)
CS_SUBFIELDS = {
    "AI": ["cs.AI"],
    "ML": ["cs.LG", "stat.ML"],
    "CV": ["cs.CV"],
    "CYB": ["cs.CR"],
    "PAST": ["cs.PL", "cs.SE"]
}

def fetch_arxiv_papers(arxiv_category, max_results=400, batch_size=100, delay=3):
    base_url = "http://export.arxiv.org/api/query?"
    results = []

    for start in range(0, max_results, batch_size):
        query = {
            "search_query": f"cat:{arxiv_category}",
            "start": start,
            "max_results": min(batch_size, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        print(f"Fetching {arxiv_category}: {start + 1} to {start + batch_size}")
        response = requests.get(base_url, params=query)
        if response.status_code != 200:
            print(f"❌ Error {response.status_code} while fetching {arxiv_category}")
            break

        root = ET.fromstring(response.content)

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip().replace("\n", " ")
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip().replace("\n", " ")
            link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()

            results.append((title, abstract, link))

        time.sleep(delay)

    return results

def fetch_and_balance_all(output_file="cs_subfields_balanced_1500.csv", target_per_class=300):
    seen_links = set()
    subfield_data = defaultdict(list)

    for subfield, categories in CS_SUBFIELDS.items():
        collected = []

        for cat in categories:
            papers = fetch_arxiv_papers(cat)
            for title, abstract, link in papers:
                if link not in seen_links:
                    seen_links.add(link)
                    collected.append((title, abstract, subfield, link))
            if len(collected) >= target_per_class:
                break

        # Shuffle and trim to target_per_class
        random.shuffle(collected)
        trimmed = collected[:target_per_class]
        subfield_data[subfield] = trimmed
        print(f"✅ {subfield}: Collected {len(trimmed)} papers")

    # Merge all into final list
    all_rows = [row for rows in subfield_data.values() for row in rows]
    print(f"\n📦 Final total: {len(all_rows)} papers across all subfields")

    with open(output_file, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Abstract", "Subfield", "Link"])
        writer.writerows(all_rows)

    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    fetch_and_balance_all()
