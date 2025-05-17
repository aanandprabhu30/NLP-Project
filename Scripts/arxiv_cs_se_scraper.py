
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

URL = "https://arxiv.org/list/cs.SE/recent?show=2000"
BASE_ABS = "https://arxiv.org"

# Get HTML content
res = requests.get(URL)
soup = BeautifulSoup(res.text, "html.parser")

# Parse all abstracts
results = []
entries = soup.find_all("dt")

for dt in entries:
    link_tag = dt.find("a", title="Abstract")
    if not link_tag:
        continue
    abs_url = BASE_ABS + link_tag["href"]
    abs_page = requests.get(abs_url)
    abs_soup = BeautifulSoup(abs_page.text, "html.parser")
    try:
        title = abs_soup.find("h1", class_="title").text.replace("Title:", "").strip()
        abstract = abs_soup.find("blockquote", class_="abstract").text.replace("Abstract:", "").strip()
        results.append([title, abstract, abs_url])
    except Exception as e:
        print(f"❌ Skipped {abs_url}: {e}")
        continue

# Save to Excel
wb = Workbook()
ws = wb.active
ws.title = "cs.SE recent"
ws.append(["Title", "Abstract", "Link"])

for row in results:
    ws.append(row)

output_path = "cs_se_recent.xlsx"
wb.save(output_path)
print(f"✅ Done. Saved {len(results)} papers to {output_path}")
