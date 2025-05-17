
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from openpyxl import Workbook
import time

# Config
SEARCH_URL = "https://www.semanticscholar.org/search?q=information%20technology%20deployment"
NUM_PAGES = 5

# Chrome in non-headless mode for visibility
options = Options()
driver = webdriver.Chrome(options=options)

# Setup Excel
wb = Workbook()
ws = wb.active
ws.append(["Title", "Abstract", "Link"])

# Start scrape
driver.get(SEARCH_URL)
time.sleep(5)

for page in range(NUM_PAGES):
    print(f"📄 Processing page {page + 1}")
    try:
        links = driver.find_elements(By.TAG_NAME, "a")
        paper_links = [link for link in links if "/paper/" in link.get_attribute("href") and link.text.strip()]

        for link_elem in paper_links:
            try:
                title = link_elem.text.strip()
                url = link_elem.get_attribute("href")
                driver.execute_script("window.open(arguments[0]);", url)
                driver.switch_to.window(driver.window_handles[1])
                time.sleep(2)
                try:
                    abstract = driver.find_element(By.CSS_SELECTOR, "meta[name='description']").get_attribute("content").strip()
                except:
                    abstract = "N/A"
                ws.append([title, abstract, url])
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except Exception as e:
                print(f"⚠️ Skipping one due to error: {e}")
                driver.switch_to.window(driver.window_handles[0])
                continue

        # Try to go to next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next Page']")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(4)
        except:
            print("⛔ No more pages or next button not found.")
            break

    except Exception as e:
        print(f"❌ Failed to process page {page + 1}: {e}")
        break

# Save file
output_file = "semantic_scholar_web_scraped_loose.xlsx"
wb.save(output_file)
driver.quit()
print(f"✅ Done. Saved to {output_file}")
