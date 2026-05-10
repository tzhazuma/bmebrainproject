#!/usr/bin/env python3
"""
ADNI IDA automated search using Selenium + Chromium.
Navigates to the Advanced Search page, logs in, and selects
the appropriate modalities for our project.

Usage:
    python scripts/adni_search.py
"""

import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


# ============================================================
# CONFIGURATION
# ============================================================
IDA_URL = "https://ida.loni.usc.edu/pages/access/search.jsp"
ADNI_SEARCH_URL = (
    "https://ida.loni.usc.edu/pages/access/search.jsp"
    "?tab=advSearch&project=ADNI&page=DOWNLOADS&subPage=IMAGE_COLLECTIONS"
)

CREDENTIALS = {
    "email": "tangzhh2022@shanghaitech.edu.cn",
    "password": "Azuma1145141919810",
}

# What we're looking for:
# - All three modalities: MRI (T1w), FDG-PET, Tau-PET
# - Research groups: CN (31), MCI (32/33/34), AD (35)
# - Visits: prefer Screening/Baseline for each phase
# - Image type: Original (1)


class ADNISearcher:
    def __init__(self, headless=False):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-data-dir=/tmp/chrome_adni_{os.getpid()}")

        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)

    def login(self):
        """Log into IDA."""
        print("[1/5] Logging into IDA...")
        self.driver.get(IDA_URL)

        # Accept cookie policy if present
        try:
            accept_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
            )
            accept_btn.click()
            time.sleep(1)
        except:
            pass

        # Fill login form
        try:
            email_input = self.wait.until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            password_input = self.driver.find_element(By.NAME, "password")
            email_input.clear()
            email_input.send_keys(CREDENTIALS["email"])
            password_input.clear()
            password_input.send_keys(CREDENTIALS["password"])

            # Try various submit buttons
            submit_selectors = [
                "//button[contains(text(), 'Log In')]",
                "//input[@type='submit' and contains(@value, 'Log')]",
                "//button[@type='submit']",
                "//form//button",
            ]
            for sel in submit_selectors:
                try:
                    btn = self.driver.find_element(By.XPATH, sel)
                    btn.click()
                    break
                except:
                    continue

            time.sleep(5)
            print(f"  Current URL: {self.driver.current_url}")
            print(f"  Page title: {self.driver.title}")
        except Exception as e:
            print(f"  Login form error: {e}")
            # Save screenshot for debugging
            self.driver.save_screenshot("/tmp/adni_login_debug.png")
            print(f"  Screenshot saved: /tmp/adni_login_debug.png")

    def navigate_to_advanced_search(self):
        """Go to the ADNI Advanced Image Search page."""
        print("[2/5] Navigating to Advanced Search...")
        self.driver.get(ADNI_SEARCH_URL)
        time.sleep(3)

    def find_result_count(self):
        """Find the current search result count."""
        soup_text = self.driver.page_source
        import re
        counts = re.findall(r'(\d+[\d,]*)\s+(?:images|results|subjects)', soup_text, re.IGNORECASE)
        return counts

    def print_page_summary(self):
        """Print key elements on the current page."""
        try:
            title = self.driver.title
            url = self.driver.current_url
            print(f"  Page: {title}")
            print(f"  URL: {url[:120]}")

            # Find submit/action buttons
            buttons = self.driver.find_elements(By.XPATH, "//input[@type='submit'] | //button")
            for btn in buttons[:10]:
                val = btn.get_attribute('value') or btn.text
                if val.strip():
                    print(f"    Button: {val.strip()[:80]}")
        except Exception as e:
            print(f"  Summary error: {e}")

    def click_advanced_tab(self):
        """Ensure we're on the Advanced Search tab."""
        try:
            # Look for Advanced Search tab/link
            adv_tabs = self.driver.find_elements(
                By.XPATH, "//*[contains(text(), 'Advanced Search')]"
            )
            for tab in adv_tabs:
                if tab.tag_name in ('a', 'button', 'span', 'li'):
                    try:
                        tab.click()
                        time.sleep(2)
                        print(f"  Clicked Advanced Search")
                        return True
                    except:
                        pass
        except:
            pass
        return False

    def check_login_status(self):
        """Check if we're logged in."""
        page_source = self.driver.page_source.lower()
        if 'sign in' in page_source or 'please log' in page_source:
            return False
        if 'advanced search' in page_source or 'search criteria' in page_source:
            return True
        return 'logout' in page_source or 'sign out' in page_source

    def run(self):
        """Main search workflow."""
        try:
            self.login()
            time.sleep(3)

            if not self.check_login_status():
                print("  ⚠️  Login may have failed. Check /tmp/adni_login_debug.png")
                print("  Continuing anyway to see what's available...")

            self.navigate_to_advanced_search()
            self.click_advanced_tab()
            self.print_page_summary()

            # Print the page HTML structure for analysis
            page_text = self.driver.page_source
            print(f"\n  Page size: {len(page_text):,} chars")

            # Look for ADNI project checkbox
            adni_checkboxes = self.driver.find_elements(
                By.XPATH, "//input[@type='checkbox' and contains(@value, 'ADNI')]"
            )
            if adni_checkboxes:
                print(f"\n  Found {len(adni_checkboxes)} ADNI checkboxes:")
                for cb in adni_checkboxes:
                    if not cb.is_selected():
                        cb.click()
                        time.sleep(0.5)
                        print(f"    Checked: {cb.get_attribute('value')}")

        finally:
            print("\n[✓] Done. Keeping browser open for manual interaction.")
            print("Press Enter to close browser and exit...")
            # Keep browser open for inspection
            time.sleep(3)
            self.driver.quit()


if __name__ == "__main__":
    searcher = ADNISearcher(headless=True)
    try:
        searcher.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
