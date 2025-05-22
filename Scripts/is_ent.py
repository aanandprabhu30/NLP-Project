import requests
import time
import csv
import json
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def fetch_papers(target_count=80):
    # Base URL for Semantic Scholar API
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Add variations of the query to get more diverse results
    queries = [
        "enterprise systems information systems",
        "enterprise resource planning information systems",
        "ERP systems information systems",
        "CRM systems information systems",
        "customer relationship management information systems",
        "supply chain management systems information systems",
        "organizational integration platforms information systems"
    ]
    
    # Headers to identify the application
    headers = {
        "User-Agent": "Research Paper Collector/1.0"
        # If you have an API key, add it here:
        # "x-api-key": "YOUR_API_KEY_HERE"
    }
    
    # Configure retry strategy with more patience
    retry_strategy = Retry(
        total=5,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    papers = []
    seen_titles = set()  # To avoid duplicates
    
    for query_idx, query in enumerate(queries):
        if len(papers) >= target_count:
            break
            
        # Parameters for the API request
        params = {
            "query": query,
            "limit": 100,
            "fields": "title,abstract,url",
            "offset": 0
        }
        
        try:
            print(f"Making request for query: '{query}'")
            
            # Wait longer before each query variation
            if query_idx > 0:
                wait_time = random.uniform(10, 15)
                print(f"Waiting {wait_time:.1f} seconds before next query...")
                time.sleep(wait_time)
                
            # Make the API request
            response = session.get(base_url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            new_papers = data.get("data", [])
            
            if not new_papers:
                print("No papers found for this query.")
                continue
                
            # Process each paper in the response
            for paper in new_papers:
                # Check if paper has both title and abstract and not seen before
                title = paper.get("title")
                if title and paper.get("abstract") and title not in seen_titles:
                    paper_data = {
                        "Title": title,
                        "Abstract": paper["abstract"],
                        "Link": paper.get("url", "N/A"),
                        "Subfield": "ENT"
                    }
                    papers.append(paper_data)
                    seen_titles.add(title)
                    
                    if len(papers) >= target_count:
                        break
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers: {e}")
            # Wait longer after an error before continuing
            wait_time = random.uniform(20, 30)
            print(f"Waiting {wait_time:.1f} seconds after error...")
            time.sleep(wait_time)
    
    print(f"Collected {len(papers)} papers")
    return papers

def save_to_csv(papers, filename):
    if not papers:
        print("No papers to save.")
        return
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Title", "Abstract", "Link", "Subfield"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for paper in papers:
                writer.writerow(paper)
                
        print(f"Successfully saved {len(papers)} papers to {filename}")
        
    except IOError as e:
        print(f"Error saving to CSV: {e}")

def main():
    print("Fetching papers from Semantic Scholar...")
    papers = fetch_papers()
    
    if papers:
        save_to_csv(papers, "is_ent.csv")
    else:
        print("No papers were collected.")

if __name__ == "__main__":
    main()
