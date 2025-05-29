#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Set
import requests
from urllib.parse import quote

def parse_args():
    parser = argparse.ArgumentParser(description='Harvest qualitative papers from Semantic Scholar')
    parser.add_argument('--existing', default='methodology_trimmed (1440).csv',
                      help='Existing CSV file to check for duplicates')
    parser.add_argument('--out', default='qualitative_new.csv',
                      help='Output CSV file')
    parser.add_argument('--year_from', type=int, default=2018,
                      help='Minimum publication year')
    parser.add_argument('--max_api_pages', type=int, default=50,
                      help='Maximum number of API pages to query')
    return parser.parse_args()

def load_existing_titles(filename: str) -> Set[str]:
    titles = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            titles.update(row['Title'].lower() for row in reader)
    return titles

def is_qualitative(abstract: str) -> bool:
    if not abstract:
        return False
    
    abstract = abstract.lower()
    qualitative_indicators = {
        'qualitative', 'interview', 'interviews', 'case study', 'case studies',
        'ethnography', 'ethnographic', 'grounded theory', 'phenomenology',
        'narrative', 'focus group', 'focus groups', 'observation', 'observations',
        'content analysis', 'discourse analysis', 'thematic analysis',
        'semi-structured', 'unstructured', 'in-depth', 'participant',
        'qualitative research', 'qualitative study', 'qualitative analysis'
    }
    
    return any(indicator in abstract for indicator in qualitative_indicators)

def search_papers(discipline: str, year_from: int, page: int) -> Dict:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query = f"{discipline} qualitative research"
    params = {
        'query': query,
        'year': f"{year_from}-",
        'fields': 'title,abstract,url,year',
        'limit': 100,
        'offset': page * 100
    }
    
    for attempt in range(3):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            time.sleep(0.2)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    
    return {}

def main():
    args = parse_args()
    existing_titles = load_existing_titles(args.existing)
    
    disciplines = ['Computer Science', 'Information Systems', 'Information Technology']
    results = {d: [] for d in disciplines}
    api_calls = 0
    
    for discipline in disciplines:
        page = 0
        while len(results[discipline]) < 250 and page < args.max_api_pages:
            data = search_papers(discipline, args.year_from, page)
            api_calls += 1
            
            if not data.get('data'):
                break
                
            for paper in data['data']:
                if len(results[discipline]) >= 250:
                    break
                    
                title = paper.get('title', '').strip()
                if not title or title.lower() in existing_titles:
                    continue
                    
                if is_qualitative(paper.get('abstract', '')):
                    results[discipline].append({
                        'Title': title,
                        'Abstract': paper.get('abstract', ''),
                        'Link': paper.get('url', ''),
                        'Discipline': discipline,
                        'Methodology': 'Qualitative'
                    })
                    existing_titles.add(title.lower())
            
            page += 1
    
    # Write results
    fieldnames = ['Title', 'Abstract', 'Link', 'Discipline', 'Methodology']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for papers in results.values():
            writer.writerows(papers)
    
    # Print summary
    print(f"\nResults Summary:")
    for discipline in disciplines:
        print(f"{discipline}: {len(results[discipline])} papers")
    print(f"Total API calls: {api_calls}")

if __name__ == '__main__':
    main()
