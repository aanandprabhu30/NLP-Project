
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

# CONFIG
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}
QUERY = "Information Systems e-government site:aisel.aisnet.org"
LIMIT = 100
PER_PAGE = 20

# Step 1: Semantic Scholar API
def fetch_papers_via_semantic_scholar(query, max_results):
    print("Fetching via Semantic Scholar API...")
    papers = []
    offset = 0

    while len(papers) < max_results:
        params = {
            "query": query,
            "limit": PER_PAGE,
            "offset": offset,
            "fields": "title,abstract,url"
        }
        res = requests.get(API_URL, params=params)
        if res.status_code != 200:
            print("API failed or quota exhausted.")
            break

        data = res.json()
        for paper in data.get("data", []):
            if paper.get("abstract") and paper.get("url"):
                papers.append({
                    "Title": paper["title"],
                    "Abstract": paper["abstract"],
                    "Link": paper["url"],
                    "Discipline": "IS"
                })
        offset += PER_PAGE
        time.sleep(1)

    return papers

# Step 2: Fallback Scraper
def scrape_aisel_search_results(base_url, max_pages=5):
    print("Fallback: Scraping AISel search results...")
    all_papers = []
    for page in range(max_pages):
        url = f"{base_url}&start={page * 10}"
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, 'html.parser')

        for result in soup.select('.search-result'):
            title_tag = result.find('h4', class_='title')
            abs_tag = result.find('p', class_='description')
            link_tag = title_tag.find('a') if title_tag else None

            if title_tag and abs_tag and link_tag:
                all_papers.append({
                    "Title": title_tag.get_text(strip=True),
                    "Abstract": abs_tag.get_text(strip=True),
                    "Link": "https://aisel.aisnet.org" + link_tag.get("href"),
                    "Discipline": "IS"
                })
        time.sleep(1)
    return all_papers

# Run
ss_papers = fetch_papers_via_semantic_scholar(QUERY, LIMIT)
if len(ss_papers) < LIMIT:
    fallback = scrape_aisel_search_results("https://aisel.aisnet.org/do/search/?q=e-government&context=2037993", max_pages=5)
    ss_papers.extend(fallback[:LIMIT - len(ss_papers)])

df = pd.DataFrame(ss_papers)
df.to_csv("is_papers_scraped.csv", index=False)
print(f"Saved {len(df)} papers to is_papers_scraped.csv")
