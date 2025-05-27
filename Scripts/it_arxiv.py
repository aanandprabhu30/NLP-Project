import requests
import xml.etree.ElementTree as ET
import csv
from urllib.parse import quote
import time

def fetch_papers(keywords, subfield, max_results=50):
    papers = []
    base_url = "http://export.arxiv.org/api/query"
    
    for keyword in keywords:
        if len(papers) >= max_results:
            break
            
        query = f'search_query=all:{quote(keyword)}&start=0&max_results=100'
        response = requests.get(f"{base_url}?{query}")
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('.//atom:entry', namespace):
                title = entry.find('atom:title', namespace).text.strip()
                
                # Skip if we already have this paper
                if any(p['title'] == title for p in papers):
                    continue
                
                abstract = entry.find('atom:summary', namespace).text.strip()
                link = entry.find('atom:id', namespace).text
                
                papers.append({
                    'title': title,
                    'abstract': abstract,
                    'subfield': subfield,
                    'link': link
                })
                
                if len(papers) >= max_results:
                    break
        
        # Be nice to arXiv's servers
        time.sleep(3)
    
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

# Collect papers for each subfield
all_papers = []
for subfield, keywords in subfields.items():
    print(f"Fetching papers for {subfield}...")
    papers = fetch_papers(keywords, subfield)
    all_papers.extend(papers)
    print(f"Found {len(papers)} papers for {subfield}")

# Write to CSV
with open('it_arxiv.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['title', 'abstract', 'subfield', 'link'])
    writer.writeheader()
    writer.writerows(all_papers)

# Print summary statistics
print("\nSummary Statistics:")
for subfield in subfields.keys():
    count = sum(1 for paper in all_papers if paper['subfield'] == subfield)
    print(f"{subfield}: {count} papers")
