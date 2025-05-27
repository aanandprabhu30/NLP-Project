"""
Semantic Scholar API Paper Collector for IT Operations and IoT/Networking

"""

import requests
import csv
import pandas as pd
import time
from typing import List, Dict

# API configuration
API_KEY = ""  # Get your key from https://www.semanticscholar.org/product/api
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def read_existing_papers() -> set:
    """Read existing papers from IT_Subfield.csv and return set of lowercase titles."""
    existing_titles = set()
    try:
        with open('IT_Subfield.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Title' in row and row['Title']:
                    existing_titles.add(row['Title'].lower().strip())
        print(f"Read {len(existing_titles)} existing papers from IT_Subfield.csv")
    except FileNotFoundError:
        print("File IT_Subfield.csv not found, creating a new file")
        with open('IT_Subfield.csv', 'w', newline='', encoding='utf-8') as f:
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
    
    # Adjust limits based on API key availability
    if not API_KEY:
        limit = 10  # Be more conservative without API key
        print(f"    📝 No API key - using conservative limits (slower collection)")
    
    while len(papers) < max_results and retries < max_retries:
        params = {
            'query': keyword,
            'limit': limit,
            'offset': offset,
            'fields': 'title,abstract,url'
        }
        
        if API_KEY:
            headers = {'x-api-key': API_KEY}
            print(f"    🔑 Using API key for faster collection")
        else:
            headers = {}
            
        try:
            response = requests.get(BASE_URL, params=params, headers=headers)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"    ⏳ Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                retries += 1
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                break
                
            for paper in data['data']:
                if len(papers) >= max_results:
                    break
                    
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '')
                if abstract is None:
                    abstract = ''
                abstract = abstract.strip()
                # Only include papers with both full title and abstract
                if not title or not abstract:
                    continue
                papers.append({
                    'Title': title,
                    'Abstract': abstract.replace('\n', ' '),
                    'Link': paper.get('url', ''),
                    'Subfield': subfield
                })
                existing_titles.add(title.lower())
            
            if not data.get('next'):
                break
                
            offset += limit
            
            # Longer delays without API key to avoid rate limiting
            sleep_time = 3 if not API_KEY else 1
            time.sleep(sleep_time)
            
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Error fetching papers: {str(e)}")
            retries += 1
            sleep_time = 10 if not API_KEY else 5
            time.sleep(sleep_time)
            
    return papers

def main():
    # Check API key status and provide guidance
    if not API_KEY:
        print("🚨 NO API KEY DETECTED")
        print("=" * 50)
        print("✅ This script will still work, but with limitations:")
        print("   • Slower collection (3-second delays between requests)")
        print("   • Smaller batch sizes (10 papers per request)")
        print("   • Risk of hitting rate limits (~100 requests per 5 minutes)")
        print()
        print("💡 For better performance, get a FREE API key:")
        print("   1. Visit: https://www.semanticscholar.org/product/api")
        print("   2. Sign up (it's free!)")
        print("   3. Add your key to the API_KEY variable in this script")
        print("=" * 50)
        print()
        # No prompt, always proceed
    else:
        print("✅ API key detected - using optimized collection settings")
    
    # Define subfields and their keywords
    subfield_keywords = {
        'OPS': [
            "IT operations", "workflow scheduling", "resource optimization",
            "predictive maintenance", "system performance", "process automation",
            "capacity planning", "resource allocation", "DevOps", "incident management"
        ],
        'IOTNET': [
            "Internet of Things", "IoT", "edge computing", "fog computing",
            "smart city", "networked systems", "connected devices",
            "IoT architecture", "sensor networks", "industrial IoT"
        ]
    }
    
    # Define target quotas
    target_quotas = {
        'OPS': 120,
        'IOTNET': 60
    }
    
    # Read existing papers
    existing_titles = read_existing_papers()
    
    # Collect papers for each subfield
    all_papers = []
    
    for subfield, keywords in subfield_keywords.items():
        print(f"\nFetching papers for {subfield}...")
        subfield_papers = []
        target = target_quotas[subfield]
        
        for keyword in keywords:
            if len(subfield_papers) >= target:
                break
                
            print(f"  Searching for: {keyword}")
            new_papers = fetch_papers(keyword, subfield, existing_titles, target - len(subfield_papers))
            subfield_papers.extend(new_papers)
            print(f"  Found {len(new_papers)} new papers")
            
            # Rate limiting - be respectful to the API
            time.sleep(1)
        
        all_papers.extend(subfield_papers)
        
        # Report results for this subfield
        collected = len(subfield_papers)
        if collected < target:
            shortfall = target - collected
            print(f"⚠️  {subfield}: Collected {collected}/{target} papers (shortfall: {shortfall})")
        else:
            print(f"✅ {subfield}: Successfully collected {collected}/{target} papers")
    
    # Save results to CSV
    output_file = 'it_semanticscholar_ops_iotnet.csv'
    
    if all_papers:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Title', 'Abstract', 'Link', 'Subfield']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_papers)
        
        print(f"\n📁 Saved {len(all_papers)} papers to '{output_file}'")
    else:
        print("\n⚠️  No new papers found to save")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY:")
    print("=" * 60)
    
    for subfield in subfield_keywords.keys():
        count = sum(1 for paper in all_papers if paper['Subfield'] == subfield)
        target = target_quotas[subfield]
        print(f"{subfield:8}: {count:3d}/{target:3d} papers collected")
    
    print(f"{'TOTAL':8}: {len(all_papers):3d} papers collected")
    
    if all_papers:
        print(f"\n📄 Output saved to: {output_file}")
        print("\n💡 To merge with existing data:")
        print("   1. Load both IT_Subfield.csv and it_semanticscholar_ops_iotnet.csv")
        print("   2. Concatenate and deduplicate by Title")
        print("   3. Save as updated master dataset")

if __name__ == "__main__":
    main()
