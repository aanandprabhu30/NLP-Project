import requests
import time
import csv
import json
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def fetch_papers():
    # Base URL for Semantic Scholar API
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Parameters for the API request
    params = {
        "query": "blockchain security privacy information systems",
        "limit": 80,
        "fields": "title,abstract,url"
    }
    
    # Headers to identify the application
    headers = {
        "User-Agent": "Research Paper Collector/1.0"
        # Add your API key here if you have one
        # "x-api-key": "YOUR_API_KEY"
    }
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    papers = []
    
    try:
        # Make the API request
        response = session.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        # Process each paper in the response
        for paper in data.get("data", []):
            # Check if paper has both title and abstract
            if paper.get("title") and paper.get("abstract"):
                paper_data = {
                    "Title": paper["title"],
                    "Abstract": paper["abstract"],
                    "Link": paper.get("url", "N/A"),
                    "Subfield": "BSP"
                }
                papers.append(paper_data)
            
            # Random sleep between 2-5 seconds to avoid rate limiting
            time.sleep(random.uniform(2, 5))
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching papers: {e}")
        return []
    
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
        save_to_csv(papers, "is_blockchain_security_privacy_papers.csv")
    else:
        print("No papers were collected.")

if __name__ == "__main__":
    main()
