"""Open the deployed dashboard in a headless browser and wake it if it is asleep.

A plain HTTP request is not enough: a sleeping Streamlit app shows a page with a
"Yes, get this app back up!" button that has to be clicked, and a running app only
counts a visit once the browser opens its live connection. So this uses a real browser.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://lazy-prices.streamlit.app")
EXPECTED_TEXT = "Lazy Prices replication"  # part of the dashboard title
WAKE_BUTTON = "get this app back up"


def app_is_showing(page) -> bool:
    """The dashboard may render inside an iframe, so check every frame."""
    for frame in page.frames:
        try:
            if frame.get_by_text(EXPECTED_TEXT).count() > 0:
                return True
        except Exception:  # a frame can disappear while the page is loading
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(8_000)

        button = page.get_by_role("button", name=WAKE_BUTTON)
        if button.count() > 0:
            print("App was asleep. Clicking the wake button.")
            button.first.click()

        # A cold start can take a couple of minutes: poll for up to 5 minutes.
        for _ in range(60):
            if app_is_showing(page):
                print("Dashboard is up.")
                page.wait_for_timeout(5_000)  # keep the connection open briefly
                browser.close()
                return 0
            try:
                page.wait_for_timeout(5_000)
            except PlaywrightTimeout:
                pass
        print("Dashboard did not load within 5 minutes.", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
