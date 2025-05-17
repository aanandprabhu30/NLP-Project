import requests
import pandas as pd
import time
from typing import List, Dict, Any
import csv

def fetch_papers(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch papers from Semantic Scholar API with pagination and retry logic.
    
    Args:
        query: Search query string
        limit: Maximum number of papers to fetch
    
    Returns:
        List of paper dictionaries
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {
        "User-Agent": "PaperFetcher/1.0 (Academic Research)",
        "Accept": "application/json"
    }
    
    all_papers = []
    page_size = 20  # API pagination size as per requirements
    total_fetched = 0
    offset = 0
    
    print(f"Starting to fetch up to {limit} papers about '{query}'...")
    
    fields = "title,abstract,url"  # Only request fields we need
    
    while total_fetched < limit:
        params = {
            "query": query,
            "fields": fields,
            "limit": min(page_size, limit - total_fetched),
            "offset": offset
        }
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                print(f"Fetching papers {total_fetched+1}-{total_fetched+min(page_size, limit-total_fetched)}...")
                response = requests.get(base_url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if not data.get("data"):
                        print("No more papers available.")
                        return all_papers
                    
                    papers = data.get("data", [])
                    all_papers.extend(papers)
                    total_fetched += len(papers)
                    offset += len(papers)
                    
                    # Sleep between requests to respect rate limits
                    if total_fetched < limit:
                        print(f"Sleeping for 5 seconds before next request...")
                        time.sleep(5)
                    
                    break  # Success, exit retry loop
                    
                elif response.status_code == 429:
                    retry_count += 1
                    wait_time = 10
                    print(f"Rate limit exceeded (429). Waiting {wait_time} seconds. Retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    
                else:
                    print(f"Error: API returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                    return all_papers
                    
            except Exception as e:
                print(f"Exception occurred: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying ({retry_count}/{max_retries})...")
                    time.sleep(5)
                else:
                    print("Maximum retries reached. Moving on.")
                    return all_papers
    
    return all_papers

def process_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Process papers and convert to a list of dictionaries with string values.
    
    Args:
        papers: List of paper dictionaries from API
    
    Returns:
        List of processed paper dictionaries with string values
    """
    processed_data = []
    
    for paper in papers:
        # Initialize with default values
        processed_paper = {
            "Title": "",
            "Abstract": "",
            "URL": ""
        }
        
        # Extract and clean title
        if "title" in paper and paper["title"]:
            processed_paper["Title"] = str(paper["title"])
            
        # Extract and clean abstract
        if "abstract" in paper and paper["abstract"]:
            processed_paper["Abstract"] = str(paper["abstract"])
        
        # Extract and clean URL
        if "url" in paper and paper["url"]:
            processed_paper["URL"] = str(paper["url"])
        
        processed_data.append(processed_paper)
    
    return processed_data

def clean_and_deduplicate(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deep-clean all fields and remove duplicates.
    
    Args:
        data: List of paper dictionaries
    
    Returns:
        Cleaned and deduplicated list of dictionaries
    """
    # Force all values to be strings
    for item in data:
        for key in item:
            item[key] = str(item[key]) if item[key] is not None else ""
    
    # Remove duplicates based on URL
    unique_urls = set()
    unique_data = []
    
    for item in data:
        if item["URL"] not in unique_urls and item["URL"]:
            unique_urls.add(item["URL"])
            unique_data.append(item)
    
    duplicate_count = len(data) - len(unique_data)
    print(f"Removed {duplicate_count} duplicate entries.")
    
    return unique_data

def main():
    query = "Information Systems"
    output_file = "is_papers_strict_filtered.csv"
    limit = 100
    
    # Fetch papers
    papers = fetch_papers(query, limit)
    if not papers:
        print("No papers found or error occurred.")
        return
    
    # Process papers
    processed_data = process_papers(papers)
    
    # Clean and deduplicate
    cleaned_data = clean_and_deduplicate(processed_data)
    
    # Save to CSV without using pandas DataFrame
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not cleaned_data:
                print("No data to write to CSV.")
                return
                
            # Get field names from the first item
            fieldnames = list(cleaned_data[0].keys())
            
            # Create CSV writer
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            
            # Write header
            writer.writeheader()
            
            # Write data
            for item in cleaned_data:
                writer.writerow(item)
        
        print(f"Successfully saved {len(cleaned_data)} unique papers to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    main()