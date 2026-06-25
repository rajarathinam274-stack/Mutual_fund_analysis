import argparse
import json
import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# CONFIG
BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL    = "https://api.mfapi.in/mf"
REQUEST_DELAY_SEC = 0.5   

KEY_SCHEMES = {
    125497: "HDFC Top 100 Direct",
    119551: "SBI Bluechip Direct",
    120503: "ICICI Prudential Bluechip Direct",
    118632: "Nippon India Large Cap Direct",
    119092: "Axis Bluechip Direct",
    120841: "Kotak Bluechip Direct",
}

# FETCH SINGLE SCHEME
def fetch_scheme_nav(scheme_code: int, timeout: int = 15) -> dict | None:
    """
    GETs https://api.mfapi.in/mf/{scheme_code}
    Returns parsed JSON dict, or None on failure.

    Response structure:
        {
          "meta": {
            "fund_house": "...",
            "scheme_type": "...",
            "scheme_category": "...",
            "scheme_code": 125497,
            "scheme_name": "..."
          },
          "data": [
            {"date": "25-06-2026", "nav": "142.3456"},
            ...  (newest → oldest)
          ],
          "status": "SUCCESS"
        }
    """
    url = f"{BASE_URL}/{scheme_code}"
    print(f"  → GET {url}")
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "MutualFundAnalytics/1.0"})
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "SUCCESS":
            print(f"    ✗  API status not SUCCESS: {data.get('status')}")
            return None
        return data
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗  Connection error: {e}")
    except requests.exceptions.Timeout:
        print(f"    ✗  Request timed out after {timeout}s")
    except requests.exceptions.HTTPError as e:
        print(f"    ✗  HTTP {resp.status_code}: {e}")
    except json.JSONDecodeError as e:
        print(f"    ✗  JSON parse error: {e}")
    return None


# PARSE → DATAFRAME
def parse_nav_response(raw: dict, scheme_code: int) -> pd.DataFrame | None:
    """Converts raw mfapi.in JSON into a clean, typed DataFrame."""
    if not raw or "data" not in raw:
        return None

    meta = raw.get("meta", {})
    nav_records = raw["data"]   # list of {"date": "DD-MM-YYYY", "nav": "string"}

    if not nav_records:
        print(f"    ⚠  No NAV records returned for code {scheme_code}")
        return None

    df = pd.DataFrame(nav_records)
  
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")

    bad_rows = df[df["date"].isna() | df["nav"].isna()]
    if len(bad_rows) > 0:
        print(f"    ⚠  Dropped {len(bad_rows)} unparseable rows")
        df = df.dropna(subset=["date", "nav"])

    df["amfi_code"]        = scheme_code
    df["scheme_name"]      = meta.get("scheme_name", "")
    df["fund_house"]       = meta.get("fund_house", "")
    df["scheme_type"]      = meta.get("scheme_type", "")
    df["scheme_category"]  = meta.get("scheme_category", "")
    df["fetched_at"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = df.sort_values("date").reset_index(drop=True)

    return df


# SAVE TO CSV
def save_nav_csv(df: pd.DataFrame, scheme_code: int, scheme_name: str) -> Path:
    """Saves NAV DataFrame to data/raw/nav_live_{code}.csv"""
    safe_name = scheme_name.replace(" ", "_").replace("/", "-")[:40]
    filename  = RAW_DIR / f"nav_live_{scheme_code}_{safe_name}.csv"
    df.to_csv(filename, index=False, date_format="%Y-%m-%d")
    return filename


# FETCH ALL KEY SCHEMES

def fetch_all_schemes(schemes: dict = None, latest_only: bool = False) -> pd.DataFrame:
    """
    Fetches all schemes in the dict {code: name}.
    Returns a combined DataFrame of all NAV records.
    """
    if schemes is None:
        schemes = KEY_SCHEMES

    all_dfs = []
    summary = []

    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  Fetching live NAV for {len(schemes)} schemes …")
    print(f"{bar}")

    for code, friendly_name in schemes.items():
        print(f"\n  [{code}] {friendly_name}")

        raw = fetch_scheme_nav(code)
        if raw is None:
            summary.append({"code": code, "name": friendly_name, "status": "FAIL", "records": 0})
            time.sleep(REQUEST_DELAY_SEC)
            continue

        df = parse_nav_response(raw, code)
        if df is None or df.empty:
            summary.append({"code": code, "name": friendly_name, "status": "EMPTY", "records": 0})
            time.sleep(REQUEST_DELAY_SEC)
            continue

        if latest_only:
            df = df.tail(1)

        latest_row = df.iloc[-1]
        oldest_row = df.iloc[0]
        print(f"    ✓  {len(df):,} records  |  "
              f"Oldest: {oldest_row['date'].strftime('%d-%b-%Y')}  →  "
              f"Latest: {latest_row['date'].strftime('%d-%b-%Y')}")
        print(f"    ₹  Latest NAV = {latest_row['nav']:.4f}  |  "
              f"Min = {df['nav'].min():.4f}  |  Max = {df['nav'].max():.4f}")

        saved_path = save_nav_csv(df, code, friendly_name)
        print(f"    💾 Saved → {saved_path.name}")

        all_dfs.append(df)
        summary.append({"code": code, "name": friendly_name, "status": "OK",
                         "records": len(df), "latest_nav": latest_row["nav"],
                         "latest_date": latest_row["date"].strftime("%Y-%m-%d")})

        time.sleep(REQUEST_DELAY_SEC) 

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = RAW_DIR / "nav_live_all_schemes.csv"
        combined.to_csv(combined_path, index=False, date_format="%Y-%m-%d")
        print(f"\n  Combined CSV → {combined_path.name}  ({len(combined):,} total rows)")
    else:
        combined = pd.DataFrame() 

    print(f"\n{'═'*60} ") 
    print(f"  FETCH SUMMARY")
    print(f"{'═'*60}")
    for s in summary:
        icon = "✓" if s["status"] == "OK" else "✗"
        extra = f"  NAV: ₹{s.get('latest_nav', 'N/A')}" if s["status"] == "OK" else ""
        print(f"  {icon}  [{s['code']}]  {s['name']:<35}  {s['records']:>5} records{extra}")

    ok_count   = sum(1 for s in summary if s["status"] == "OK")
    fail_count = len(summary) - ok_count
    print(f"\n  Result: {ok_count}/{len(summary)} schemes fetched successfully  |  {fail_count} failed")

    return combined


# QUICK ANALYSIS ON FETCHED DATA
def quick_analysis(df: pd.DataFrame):
    """Prints a brief analysis of fetched NAV data."""
    if df.empty:
        print("  No data to analyse.")
        return

    print(f"\n{'═'*60}")
    print(f"  QUICK NAV ANALYSIS")
    print(f"{'═'*60}")

    latest = (df.sort_values("date")
                .groupby("amfi_code")
                .last()
                .reset_index()[["amfi_code", "scheme_name", "date", "nav"]])

    print(f"\n  Latest NAV per scheme:")
    for _, row in latest.iterrows():
        print(f"    [{row['amfi_code']}] {str(row['scheme_name'])[:40]:<40}  "
              f"₹{row['nav']:>10.4f}  ({row['date'].strftime('%d-%b-%Y')})")
      
    print(f"\n  1-Year Approx Return (last 252 trading days):")
    for code in df["amfi_code"].unique():
        sub = df[df["amfi_code"] == code].sort_values("date")
        if len(sub) >= 252:
            nav_now  = sub.iloc[-1]["nav"]
            nav_1yr  = sub.iloc[-252]["nav"]
            ret_1yr  = ((nav_now - nav_1yr) / nav_1yr) * 100
            name     = str(sub.iloc[-1]["scheme_name"])[:35]
            icon     = "▲" if ret_1yr > 0 else "▼"
            print(f"    {icon}  [{code}] {name:<35}  {ret_1yr:>+7.2f}%")


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Fetch live NAV from mfapi.in")
    parser.add_argument("--code", type=int, default=None,
                        help="Fetch a single scheme by AMFI code")
    parser.add_argument("--latest-only", action="store_true",
                        help="Fetch only the most recent NAV (faster)")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   LIVE NAV FETCHER  —  mfapi.in                          ║")
    print("╚" + "═"*58 + "╝")
    print(f"  Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API Base      : {BASE_URL}")
    print(f"  Output dir    : {RAW_DIR}")

    if args.code:
        schemes = {args.code: f"Custom Code {args.code}"}
    else:
        schemes = KEY_SCHEMES

    combined = fetch_all_schemes(schemes, latest_only=args.latest_only)

    if not combined.empty:
        quick_analysis(combined)

    print(f"\n  Done. Files saved to: {RAW_DIR}\n")


if __name__ == "__main__":
    main()
