import requests
import csv
import pandas as pd
import time
import random
from typing import List, Dict

# API configuration
API_KEY = ""  # Enter your Semantic Scholar API key here for higher rate limits
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def read_existing_papers() -> set:
    """Read existing papers from it_arxiv.csv and return set of lowercase titles."""
    existing_titles = set()
    try:
        with open('it_arxiv.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Title' in row and row['Title']:
                    existing_titles.add(row['Title'].lower().strip())
        print(f"Read {len(existing_titles)} existing papers from it_arxiv.csv")
    except FileNotFoundError:
        print("File it_arxiv.csv not found, creating a new file")
        with open('it_arxiv.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Title', 'Abstract', 'Link', 'Subfield'])
            writer.writeheader()
    except Exception as e:
        print(f"Error reading existing papers: {str(e)}")
    
    return existing_titles

def fetch_papers(keyword: str, subfield: str, existing_titles: set, max_results: int = 75) -> List[Dict]:
    """Fetch papers from Semantic Scholar API for a given keyword and subfield."""
    papers = []
    offset = 0
    limit = 100  # Maximum allowed by API
    retries = 0
    max_retries = 3
    
    while len(papers) < max_results and retries < max_retries:
        params = {
            'query': keyword,
            'limit': limit,
            'offset': offset,
            'fields': 'title,abstract,url'
        }
        
        if API_KEY:
            headers = {'x-api-key': API_KEY}
        else:
            headers = {}
            
        try:
            response = requests.get(BASE_URL, params=params, headers=headers)
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = random.uniform(2, 5) * (retries + 1)  # Exponential backoff
                print(f"Rate limited. Waiting for {wait_time:.1f} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get('data'):
                break
                
            for paper in data['data']:
                title = paper.get('title', '').strip()
                if not title or title.lower() in existing_titles:
                    continue
                
                # Handle None values properly
                abstract = paper.get('abstract', '')
                if abstract is not None:
                    abstract = abstract.replace('\n', ' ').strip()
                else:
                    abstract = ''
                    
                papers.append({
                    'Title': title,
                    'Abstract': abstract,
                    'Link': paper.get('url', ''),
                    'Subfield': subfield
                })
                existing_titles.add(title.lower())
                
                if len(papers) >= max_results:
                    break
                    
            # Reset retries on success
            retries = 0
            offset += limit
            time.sleep(2)  # Increased rate limiting delay
            
        except Exception as e:
            print(f"Error fetching papers for {keyword}: {str(e)}")
            retries += 1
            if retries < max_retries:
                wait_time = random.uniform(2, 5) * retries
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                break
            
    return papers

# Define subfields and their keywords
subfields = {
    'CLD': [
        "cloud computing", "virtualization", "orchestration", "SaaS",
        "cloud infrastructure", "cloud monitoring", "cloud platform",
        "cloud service delivery"
    ],
    'IOTNET': [
        "Internet of Things", "IoT", "edge computing", "fog computing",
        "smart city", "networked systems", "connected devices", "IoT architecture"
    ],
    'OPS': [
        "workflow scheduling", "resource optimization", "predictive maintenance",
        "IT operations", "system performance", "process automation",
        "capacity planning", "resource allocation"
    ],
    'SEC': [
        "cybersecurity", "access control", "encryption", "blockchain security",
        "identity management", "security in cloud", "privacy in IoT",
        "network security"
    ]
}

# Define target quotas for each subfield
target_quotas = {
    'CLD': 75,
    'IOTNET': 60,
    'OPS': 100,
    'SEC': 75
}

# Read existing papers
existing_titles = read_existing_papers()
print(f"Found {len(existing_titles)} existing papers")

# Collect new papers for each subfield
all_new_papers = []
for subfield, keywords in subfields.items():
    target = target_quotas[subfield]
    print(f"\nFetching papers for {subfield} (target: {target})...")
    subfield_papers = []
    
    for keyword in keywords:
        if len(subfield_papers) >= target:
            break
        papers = fetch_papers(keyword, subfield, existing_titles, target - len(subfield_papers))
        subfield_papers.extend(papers)
        print(f"Found {len(papers)} new papers for keyword '{keyword}'")
    
    all_new_papers.extend(subfield_papers)
    print(f"Total new papers for {subfield}: {len(subfield_papers)}/{target}")

# Write to CSV
if all_new_papers:
    with open('it_semanticscholar_new.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Title', 'Abstract', 'Link', 'Subfield'])
        writer.writeheader()
        writer.writerows(all_new_papers)

# Print summary statistics
print("\nSummary Statistics:")
for subfield in subfields.keys():
    count = sum(1 for paper in all_new_papers if paper['Subfield'] == subfield)
    target = target_quotas[subfield]
    print(f"{subfield}: {count}/{target} new papers")
print(f"\nTotal new papers collected: {len(all_new_papers)}")
