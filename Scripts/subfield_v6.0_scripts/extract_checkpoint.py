#!/usr/bin/env python3
"""Extract all results from checkpoint to CSV"""

import json
import csv

# Load checkpoint
with open('multi_llm_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)

# Extract results
results = checkpoint['results']
print(f"Found {len(results)} classified papers in checkpoint")

# Sort by index
results.sort(key=lambda x: x[0])

# Write to CSV
with open('classified_from_checkpoint.csv', 'w', newline='', encoding='utf-8') as csvfile:
    if results:
        # Get data from results (index 1 is the dict)
        data = [r[1] for r in results]
        
        # Define fieldnames (excluding Timestamp)
        fieldnames = ['Title', 'Abstract', 'Discipline', 'Subfield', 
                     'Discipline_Confidence', 'Subfield_Confidence', 'Classifier']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
        
print(f"Saved {len(results)} papers to classified_from_checkpoint.csv")

# Show provider summary
provider_counts = {}
for _, result in results:
    provider = result.get('Classifier', 'unknown')
    provider_counts[provider] = provider_counts.get(provider, 0) + 1

print("\nPapers by provider:")
for provider, count in provider_counts.items():
    print(f"  {provider}: {count}")