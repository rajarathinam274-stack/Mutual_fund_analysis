import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# PATHS

BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports"

for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# DATASET REGISTRY

  DATASETS = {
    "fund_master":            "fund_master.csv",
    "nav_history":            "nav_history.csv",
    "sip_transactions":       "sip_transactions.csv",
    "lumpsum_transactions":   "lumpsum_transactions.csv",
    "investor_profiles":      "investor_profiles.csv",
    "portfolio_holdings":     "portfolio_holdings.csv",
    "benchmark_index":        "benchmark_index.csv",
    "dividends":              "dividends.csv",
    "amc_details":            "amc_details.csv",
    "expense_ratio_history":  "expense_ratio_history.csv",
}

# HELPER: PRETTY SECTION HEADER

def section(title: str):
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def subsection(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# LOAD ALL 10 DATASETS

def load_all_datasets() -> dict[str, pd.DataFrame]:
    section("STEP 1 ▸ Loading All 10 CSV Datasets")
    dfs: dict[str, pd.DataFrame] = {}
    anomalies: dict[str, list[str]] = {}

    for name, filename in DATASETS.items():
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f"  ✗  {name}: FILE NOT FOUND at {filepath}")
            continue

        df = pd.read_csv(filepath, low_memory=False)
        dfs[name] = df

        subsection(f"{name.upper()}  ←  {filename}")

        print(f"\n  .shape   → {df.shape[0]:,} rows × {df.shape[1]} cols")
      
        print(f"\n  .dtypes:")
        for col, dtype in df.dtypes.items():
            null_count = df[col].isna().sum()
            null_pct   = null_count / len(df) * 100
            flag = f"  ⚠ {null_count} nulls ({null_pct:.1f}%)" if null_count > 0 else ""
            print(f"    {col:<35} {str(dtype):<12}{flag}")

        print(f"\n  .head(3):")
        print(df.head(3).to_string(max_cols=8, max_colwidth=25))
      
        noted = []

        null_cols = df.columns[df.isna().any()].tolist()
        if null_cols:
            for col in null_cols:
                n = df[col].isna().sum()
                noted.append(f"NULL  : {col} has {n} missing values ({n/len(df)*100:.1f}%)")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            noted.append(f"DUP   : {dup_count} fully duplicate rows detected")
        for col in df.select_dtypes(include=[np.number]).columns:
            if col.lower() in ('unrealised_pnl', 'xirr_pct'):
                continue 
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                noted.append(f"NEG   : {col} has {neg_count} negative values")
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    bad    = parsed.isna().sum()
                    if bad > 0:
                        noted.append(f"DATE  : {col} has {bad} un-parseable date strings")
                    future_count = (parsed > pd.Timestamp.now()).sum()
                    if future_count > 0:
                        noted.append(f"DATE  : {col} has {future_count} future-dated records")
                except Exception:
                    pass
      
        for col in df.select_dtypes(include='object').columns:
            if df[col].nunique() < 10:
                vals = df[col].dropna().unique().tolist()
                noted.append(f"CATS  : {col} unique values = {vals}")

        anomalies[name] = noted

        if noted:
            print(f"\n  ⚠  ANOMALIES / OBSERVATIONS ({len(noted)}):")
            for a in noted:
                print(f"     • {a}")
        else:
            print(f"\n  ✓  No anomalies detected.")

    return dfs, anomalies


# STEP 2 – FUND MASTER TAXONOMY EXPLORATION

def explore_fund_master(df: pd.DataFrame):
    section("STEP 2 ▸ Fund Master — Taxonomy Exploration")

    print(f"\n  Total schemes in fund_master : {len(df):,}")

    subsection("Unique Fund Houses")
    fh_counts = df['fund_house'].value_counts()
    for fh, cnt in fh_counts.items():
        print(f"    {fh:<40} {cnt:>4} schemes")

    subsection("Unique Categories")
    cat_counts = df['category'].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"    {cat:<30} {cnt:>4} schemes")

    subsection("Unique Sub-Categories")
    sub_counts = df['sub_category'].value_counts()
    for sub, cnt in sub_counts.items():
        print(f"    {sub:<30} {cnt:>4} schemes")

    subsection("Risk Grade Distribution")
    risk_counts = df['risk_grade'].value_counts()
    for risk, cnt in risk_counts.items():
        bar = '█' * (cnt // 5)
        print(f"    {risk:<25} {cnt:>4}  {bar}")

    subsection("Plan Type  ×  Option Cross-Tab")
    cross = pd.crosstab(df['plan_type'], df['option'])
    print(cross.to_string())

    subsection("AMFI Scheme Code Structure Analysis")
    codes = df['amfi_code'].dropna().astype(int)
    print(f"    Min code   : {codes.min():,}")
    print(f"    Max code   : {codes.max():,}")
    print(f"    Range span : {codes.max()-codes.min():,}")
    print(f"    Unique     : {codes.nunique():,}")
    print(f"    Duplicates : {codes.duplicated().sum()}")
    print(f"\n    NOTE: AMFI codes are sequential integers assigned by AMFI.")
    print(f"    Codes <110000 are typically older schemes (pre-2010).")
    print(f"    Direct Plan codes are generally higher than Regular Plan codes.")
    print(f"    Growth option codes precede IDCW option codes for same scheme.")

    subsection("AUM Distribution (top 5 schemes by AUM)")
    top_aum = df.nlargest(5, 'aum_crores')[['scheme_name','aum_crores','fund_house']]
    for _, row in top_aum.iterrows():
        print(f"    ₹{row['aum_crores']:>12,.2f} Cr  |  {str(row['scheme_name'])[:50]}")

# STEP 3 – DATA QUALITY VALIDATION (AMFI CODE COVERAGE ) 

def validate_amfi_codes(fund_master: pd.DataFrame, nav_history: pd.DataFrame) -> dict:
    section("STEP 3 ▸ AMFI Code Validation — fund_master ↔ nav_history")

    master_codes = set(fund_master['amfi_code'].dropna().astype(int))
    nav_codes    = set(nav_history['amfi_code'].dropna().astype(int))

    in_both       = master_codes & nav_codes
    only_master   = master_codes - nav_codes
    only_nav      = nav_codes - master_codes

    coverage_pct  = len(in_both) / len(master_codes) * 100 if master_codes else 0

    print(f"\n  Codes in fund_master                : {len(master_codes):>6,}")
    print(f"  Codes in nav_history                : {len(nav_codes):>6,}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  ✓ Present in BOTH                   : {len(in_both):>6,}")
    print(f"  ✗ In fund_master but NOT nav_history : {len(only_master):>6,}  ({100-coverage_pct:.1f}% missing)")
    print(f"  ? In nav_history but NOT fund_master : {len(only_nav):>6,}  (orphan NAV records)")
    print(f"\n  NAV Coverage                        : {coverage_pct:.2f}%")
  
    key_codes = {125497: 'HDFC Top 100 Direct', 119551: 'SBI Bluechip', 120503: 'ICICI Bluechip',
                 118632: 'Nippon LargeCap',      119092: 'Axis Bluechip', 120841: 'Kotak Bluechip'}

    subsection("Key Scheme Code Validation")
    for code, name in key_codes.items():
        in_m = code in master_codes
        in_n = code in nav_codes
        nav_recs = len(nav_history[nav_history['amfi_code'] == code])
        status = "✓ PASS" if (in_m and in_n) else "✗ FAIL"
        print(f"  {status}  {code}  {name:<30}  master:{in_m}  nav:{in_n}  records:{nav_recs:,}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "fund_master_total_codes": len(master_codes),
        "nav_history_total_codes": len(nav_codes),
        "codes_in_both": len(in_both),
        "codes_only_in_master": len(only_master),
        "codes_only_in_nav": len(only_nav),
        "nav_coverage_pct": round(coverage_pct, 4),
        "orphan_nav_codes": sorted(list(only_nav))[:10],
        "missing_nav_codes_sample": sorted(list(only_master))[:10],
    }

    return summary


# STEP 4 – WRITE DATA QUALITY REPORT

def write_quality_report(anomalies: dict, validation: dict):
    section("STEP 4 ▸ Writing Data Quality Report")
    report_path = REPORTS_DIR / "data_quality_report.md"

    lines = [
        "# Data Quality Report — Day 1",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Dataset Anomalies",
        "",
    ]

    for ds, issues in anomalies.items():
        lines.append(f"### {ds}")
        if issues:
            for iss in issues:
                lines.append(f"- {iss}")
        else:
            lines.append("- ✓ No anomalies detected")
        lines.append("")

    lines += [
        "## 2. AMFI Code Coverage (fund_master ↔ nav_history)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| fund_master unique codes | {validation['fund_master_total_codes']:,} |",
        f"| nav_history unique codes | {validation['nav_history_total_codes']:,} |",
        f"| Codes present in both | {validation['codes_in_both']:,} |",
        f"| Codes missing from nav_history | {validation['codes_only_in_master']:,} |",
        f"| Orphan NAV codes (no master entry) | {validation['codes_only_in_nav']:,} |",
        f"| **NAV Coverage %** | **{validation['nav_coverage_pct']:.2f}%** |",
        "",
        "## 3. Recommendations",
        "",
        "- Codes missing from `nav_history` should be investigated — likely newer/inactive schemes.",
        "- Orphan codes in `nav_history` suggest `fund_master` is stale and needs refresh from AMFI portal.",
        "- `investor_profiles.age` nulls (5 records) → impute with median or flag for data team.",
        "- Parse all `*_date` columns as `datetime64` in processing step.",
        "- Normalize `expense_ratio_history` to deduplicate before joining.",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Report written → {report_path}")


# MAIN

def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║   MUTUAL FUND ANALYTICS — DAY 1 DATA INGESTION                    ║")
    print("╚" + "═"*68 + "╝")
    print(f"  Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  Pandas        : {pd.__version__}")

    dfs, anomalies = load_all_datasets()

    if not dfs:
        print("\nERROR: No datasets loaded. Check data/raw/ directory.")
        sys.exit(1)
      
    if "fund_master" in dfs:
        explore_fund_master(dfs["fund_master"])
      
    if "fund_master" in dfs and "nav_history" in dfs:
        validation = validate_amfi_codes(dfs["fund_master"], dfs["nav_history"])
    else:
        validation = {}
        print("Skipping AMFI validation — fund_master or nav_history missing.")

    if validation:
        write_quality_report(anomalies, validation)

    section("DONE ▸ Day 1 Ingestion Complete")
    print(f"\n  Datasets loaded       : {len(dfs)}/10")
    print(f"  Total rows ingested   : {sum(len(df) for df in dfs.values()):,}")
    print(f"  Quality report        : reports/data_quality_report.md")
    print(f"\n  Next → run live_nav_fetch.py to pull live NAV data.")
    print()


if __name__ == "__main__":
    main()
