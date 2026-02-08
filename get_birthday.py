import sys
import argparse
import time
import re
import random
import math
import pandas as pd
import numpy as np
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuration & Patterns ---
pat_age_compiled = re.compile(r'Age:\s*(\d+)', flags=re.DOTALL | re.IGNORECASE)


def extract_first(pat, text, cast=None):
    m = pat.search(text)
    if not m: return np.nan
    val = m.group(1)
    if cast:
        try:
            return cast(val)
        except:
            return np.nan
    return val


def has_error_toast(driver, timeout=1):
    """Detects the 'Something wrong happened' popup for deleted profiles."""
    locator = (
        By.XPATH,
        "//div[contains(@class,'rs-alertbox-danger') "
        "and .//div[contains(@class,'text-toast') and contains(., 'Something wrong happened')]]"
    )
    try:
        WebDriverWait(driver, timeout, poll_frequency=0.1).until(
            EC.presence_of_element_located(locator)
        )
        return True
    except TimeoutException:
        return False


def get_age_with_retry(driver, url, max_tries=3):
    """Attempts to scrape age; handles deleted profiles and standard retries."""
    for attempt in range(1, max_tries + 1):
        try:
            driver.get(url)

            # Check for deleted profile popup immediately
            if has_error_toast(driver):
                print(f"Detected deleted profile: {url}. Waiting 5s penalty...")
                time.sleep(5)
                return "DELETED"

            time.sleep(1.5 + random.uniform(0.2, 0.5))
            html = driver.page_source
            age = extract_first(pat_age_compiled, html, cast=int)

            if not pd.isna(age) and 3 <= age <= 200:
                return age
        except Exception:
            pass
    return np.nan


def _process_chunk_age(ids_chunk, federation, ath_ids_series):
    """Worker process: Handles a chunk of athlete IDs."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.page_load_strategy = "eager"

    # FIX: On GitHub Ubuntu runners, Chrome is already installed.
    # We don't need webdriver-manager inside the workers.
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception:
        # Fallback for local testing if standard call fails
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

    results = []
    try:
        for ridx in ids_chunk:
            ath_id_val = ath_ids_series.iloc[ridx]
            if pd.isna(ath_id_val):
                results.append((ridx, np.nan))
                continue

            url = f"https://{federation}.results.info/athlete/{int(ath_id_val)}"
            age = get_age_with_retry(driver, url)
            results.append((ridx, age))
    finally:
        driver.quit()
    return results


def run_scraping_cycle(federation, n_workers=10):
    """Main logic: Manages the 5-cycle retry and permanent profile removal."""
    csv_path = f"{federation}_results/athletes_age.csv"

    # Load data
    try:
        ath_age = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    today_str = date.today().strftime("%m_%d")
    if today_str not in ath_age.columns:
        ath_age[today_str] = np.nan

    # RE-DO LOGIC: Loop up to 5 times
    for cycle in range(1, 6):
        # Filter: (Age doesn't match Max Age) AND (Today's entry is still NaN)
        val = ath_age.iloc[:, 2]
        rowmax = ath_age.iloc[:, 3:].max(axis=1, skipna=True)
        idx_list = ath_age.index[~val.eq(rowmax) & ath_age[today_str].isna()].tolist()

        if not idx_list:
            print(f"All target data captured for {federation} by cycle {cycle}.")
            break

        print(f"Cycle {cycle}: Scraping {len(idx_list)} entries for {federation}...")

        chunk_size = math.ceil(len(idx_list) / max(1, n_workers))
        chunks = [idx_list[k:k + chunk_size] for k in range(0, len(idx_list), chunk_size)]

        cycle_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_chunk_age, ch, federation, ath_age["ath_id"]) for ch in chunks]
            for fut in as_completed(futures):
                cycle_results.extend(fut.result())

        # Process results
        ids_to_remove = []
        for ridx, age in cycle_results:
            if age == "DELETED":
                ids_to_remove.append(ridx)
            else:
                ath_age.loc[ridx, today_str] = age

        # Permanent removal of deleted profiles
        if ids_to_remove:
            print(f"Permanently removing {len(ids_to_remove)} deleted records.")
            ath_age = ath_age.drop(index=ids_to_remove).reset_index(drop=True)

        # Immediate save after each cycle
        ath_age.to_csv(csv_path, index=False)
        print(f"Cycle {cycle} complete. Data saved.")


if __name__ == "__main__":
    Service(ChromeDriverManager().install())
    parser = argparse.ArgumentParser()
    parser.add_argument("--fed", choices=['ifsc', 'dav'], required=True)
    args = parser.parse_args()

    run_scraping_cycle(args.fed)