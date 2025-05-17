
import arxiv
import csv
import time

# Trusted CS subcategories for safe coverage
categories = ["cs.AI", "cs.CV", "cs.CL", "cs.CR", "cs.SE"]
results_per_category = 200
batch_size = 100
delay_seconds = 3.0

# Final dataset
all_results = []

# Loop through each category and fetch results in batches
for cat in categories:
    print(f"🔍 Scraping {results_per_category} papers from: {cat}")
    for start in range(0, results_per_category, batch_size):
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=batch_size,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        client = arxiv.Client(page_size=batch_size, delay_seconds=delay_seconds)

        try:
            for result in client.results(search):
                all_results.append({
                    "Title": result.title.strip().replace("\n", " "),
                    "Abstract": result.summary.strip().replace("\n", " "),
                    "Link": result.entry_id.strip()
                })
        except Exception as e:
            print(f"⚠️ Error scraping {cat} batch starting at {start}: {e}")
        time.sleep(delay_seconds)

# Save to CSV
output_file = "arxiv_cs.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Title", "Abstract", "Link"])
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print(f"✅ Done! Saved {len(all_results)} CS papers to {output_file}")
