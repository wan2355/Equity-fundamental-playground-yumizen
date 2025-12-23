#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yy02_merge_manual_into_core_FULL_v4.py

Assumption (FINAL)
------------------
- ALL numeric values in tv_manual.csv are already in USD.
- NO unit conversion is performed.

Behavior
--------
- core is NEVER overwritten
- manual USD values fill ONLY missing core values
- Revenue / EBIT / FCF / CapEx / SBC / MktCap_USD supported
- SRC_* columns track data origin (core / manual_USD)

Human-edited files
------------------
- draft_input_core_usd.csv  (reference only)
- tv_manual.csv             (USD values only)

Output
------
- draft_input_core_merged.csv
- tv_manual.csv (regenerated: remaining blanks only)
"""

import argparse
import pandas as pd
import re
from pathlib import Path

KEYS = ["Ticker", "Year"]

FILL_COLS = [
    "Revenue",
    "EBIT",
    "FCF",
    "CapEx",
    "SBC",
    "MktCap_USD",
    "Price",
]

SRC_COLS = {
    "Revenue": "SRC_Revenue",
    "EBIT": "SRC_EBIT",
    "FCF": "SRC_FCF",
    "CapEx": "SRC_CapEx",
    "SBC": "SRC_SBC",
    "MktCap_USD": "SRC_MktCap",
}

def norm_col(c):
    if c is None:
        return c
    c0 = str(c).strip()
    c1 = re.sub(r"\s+", "", c0.lower())
    mapping = {
        "ticker": "Ticker",
        "symbol": "Ticker",
        "year": "Year",
        "fy": "Year",
        "revenue": "Revenue",
        "sales": "Revenue",
        "ebit": "EBIT",
        "fcf": "FCF",
        "capex": "CapEx",
        "capex設備投資": "CapEx",
        "sbc": "SBC",
        "stockbasedcomp": "SBC",
        "stockbasedcompensation": "SBC",
        "marketcap": "MktCap_USD",
        "mktcap": "MktCap_USD",
        "mktcap_usd": "MktCap_USD",
        "price": "Price",
        "last": "Price",
    }
    return mapping.get(c1, c0)

def load_csv(p):
    df = pd.read_csv(p)
    df = df.rename(columns={c: norm_col(c) for c in df.columns})
    return df

def ensure_keys(df, name):
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suggest", required=True)
    args = ap.parse_args()

    core = ensure_keys(load_csv(args.core), "core")
    manual = ensure_keys(load_csv(args.manual), "manual")

    for c in FILL_COLS:
        if c not in core.columns:
            core[c] = pd.NA
        if c not in manual.columns:
            manual[c] = pd.NA

    manual_use = manual[KEYS + FILL_COLS].copy()

    merged = core.merge(
        manual_use,
        on=KEYS,
        how="left",
        suffixes=("", "__manual"),
        validate="m:1",
    )

    core_present = {c: ~merged[c].isna() for c in SRC_COLS}

    # Fill only missing core values (NO conversion)
    for c in FILL_COLS:
        mc = f"{c}__manual"
        if mc in merged.columns:
            merged[c] = merged[c].where(~merged[c].isna(), merged[mc])

    # Source flags
    for c, src in SRC_COLS.items():
        merged[src] = "core"
        filled = (~core_present[c]) & (~merged[c].isna())
        merged.loc[filled, src] = "manual_USD"

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__manual")])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    # Build suggest file (remaining blanks only)
    suggest = core[KEYS + FILL_COLS].copy()
    mask = False
    for c in FILL_COLS:
        mask = mask | suggest[c].isna()
    suggest = suggest[mask].copy()

    Path(args.suggest).parent.mkdir(parents=True, exist_ok=True)
    suggest.to_csv(args.suggest, index=False)

    print("OK")
    print(f"merged  : {args.out}")
    print(f"suggest : {args.suggest}")

if __name__ == "__main__":
    main()
