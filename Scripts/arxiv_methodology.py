import arxiv
import time
from typing import List, Dict
import re
import csv

def search_papers(query: str, max_results: int = 100) -> List[Dict]:
    """Search arXiv papers with given query and return results."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for paper in search.results():
        try:
            results.append({
                'title': str(paper.title),
                'abstract': str(paper.summary),
                'url': str(paper.entry_id),
                'field': None,  # Will be set later
                'methodology': None  # Will be set later
            })
        except Exception as e:
            print(f"Skipping paper due to error: {e}")
    return results

def get_methodology_keywords():
    """Return methodology-specific search keywords."""
    return {
        'Qualitative': [
            'qualitative', 'case study', 'interview', 'focus group', 
            'thematic analysis', 'grounded theory', 'ethnography'
        ],
        'Quantitative': [
            'quantitative', 'statistical analysis', 'survey', 'experiment',
            'regression', 'correlation', 'statistical', 'metrics'
        ],
        'Mixed': [
            'mixed methods', 'multi-method', 'combined qualitative and quantitative',
            'mixed methodology', 'mixed-method'
        ]
    }

def get_field_categories():
    """Return field-specific search categories."""
    return {
        'CS': ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.SE'],
        'IS': ['cs.IR', 'cs.CY', 'cs.DB', 'cs.HC'],
        'IT': ['cs.DC', 'cs.NI', 'cs.PF', 'cs.SE']
    }

def main():
    all_papers = []
    methodology_keywords = get_methodology_keywords()
    field_categories = get_field_categories()
    
    # Search for papers in each field and methodology combination
    for field, categories in field_categories.items():
        for methodology, keywords in methodology_keywords.items():
            for keyword in keywords:
                query = f"cat:({' OR '.join(categories)}) AND ({keyword})"
                papers = search_papers(query)
                
                # Add field and methodology information
                for paper in papers:
                    paper['field'] = field
                    paper['methodology'] = methodology
                
                all_papers.extend(papers)
                print(f"Found {len(papers)} papers for {field} - {methodology} - {keyword}")

    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        if paper['title'] not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(paper['title'])

    # Save to CSV using csv module
    if unique_papers:
        fieldnames = list(unique_papers[0].keys())
        try:
            with open('methodology_papers.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in unique_papers:
                    writer.writerow(paper)
            print(f"Saved {len(unique_papers)} unique papers to methodology_papers.csv")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    else:
        print("No papers found to save.")

if __name__ == "__main__":
    main()
