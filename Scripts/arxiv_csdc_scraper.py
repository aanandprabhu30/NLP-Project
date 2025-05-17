
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

# Target arXiv category for Distributed Computing
base_url = "https://arxiv.org/list/cs.DC/recent"
abs_base = "https://arxiv.org"

# Request the main page
res = requests.get(base_url)
soup = BeautifulSoup(res.text, "html.parser")

# Lists to store extracted data
results = []

# Extract abstract links from the listing
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

# Save to Excel using openpyxl
wb = Workbook()
ws = wb.active
ws.title = "arXiv cs.DC"
ws.append(["Title", "Abstract", "Link"])  # Header

for row in results:
    ws.append(row)

output_file = "arxiv_csdc_recent.xlsx"
wb.save(output_file)
print(f"✅ Scraping complete. Saved {len(results)} entries to {output_file}")
