
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

# Setup
base_url = "https://arxiv.org/list/cs.NI/recent"
abs_base = "https://arxiv.org"

# Request recent page
res = requests.get(base_url)
soup = BeautifulSoup(res.text, "html.parser")

# Lists to store data
results = []

# Find all abstract links
for dt in soup.find_all("dt"):
    link_tag = dt.find("a", title="Abstract")
    if not link_tag:
        continue

    abs_url = abs_base + link_tag["href"]
    abs_res = requests.get(abs_url)
    abs_soup = BeautifulSoup(abs_res.text, "html.parser")

    try:
        title = abs_soup.find("h1", class_="title").text.replace("Title:", "").strip()
        abstract = abs_soup.find("blockquote", class_="abstract").text.replace("Abstract:", "").strip()
        results.append([title, abstract, abs_url])
    except Exception as e:
        print(f"❌ Failed to extract from {abs_url}: {e}")
        continue

# Write to Excel using openpyxl
wb = Workbook()
ws = wb.active
ws.title = "arXiv cs.NI"
ws.append(["Title", "Abstract", "Link"])  # Header row

for row in results:
    ws.append(row)

output_file = "arxiv_csni_recent.xlsx"
wb.save(output_file)
print(f"✅ Done! Saved {len(results)} entries to {output_file}")
