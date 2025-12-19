#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## uasge
#{{{
"""

python3 sec_fetch_draft_csv.py \
  --tickers LITE MU PLTR NVDA \
  --years 8 \
  --outdir sec_out \
  --user-agent "YourProjectName/0.1 (your_email@example.com)"


sec_fetch_draft_csv.py

Semi-automatic fundamentals collector (public source, no paid API).
- Input: tickers (US listed), one per line, or via --tickers AAPL MSFT ...
- Fetches SEC EDGAR XBRL "companyfacts" JSON per company (requires internet).
- Extracts annual (FY) series for:
    Revenue  -> us-gaap:Revenues
    EBIT     -> us-gaap:OperatingIncomeLoss (proxy for EBIT)
    CFO      -> us-gaap:NetCashProvidedByUsedInOperatingActivities
    CapEx    -> us-gaap:PaymentsToAcquirePropertyPlantAndEquipment (positive outflow)
    SBC      -> us-gaap:ShareBasedCompensation
    FCF      -> CFO - CapEx
- Output: draft_input.csv compatible with financial_eval_04k.py.
Notes:
- Many companies have missing tags / restatements; this script aims for "good enough draft".
- You will still want to spot-check and manually fix the CSV for edge cases.

Usage examples:
  python3 sec_fetch_draft_csv.py --tickers NVDA MU PLTR LITE --years 8 --outdir sec_out
  python3 sec_fetch_draft_csv.py --ticker-file tickers.txt --years 10 --outdir sec_out

==============================================================
PROGRAM 1 : yy01_sec_fetch_draft_csv_v03.py
================================================================================
目的
  SEC（companyfacts）から財務データを取得し、下流分析用の下書きCSVを作成する。
  ※ 金額単位はすべて USD（1ドル単位）

引数一覧
  -t  --tickers        ティッカーを直接指定（空白区切り）
  -f  --ticker-file    ティッカー一覧ファイル（1行1銘柄）
  -y  --years          取得する直近年数（default: 8）
  -o  --outdir         出力ディレクトリ（default: sec_out）
  -u  --user-agent     SEC用User-Agent（推奨：連絡先入り）

使用例（全引数）
  python3 yy01_sec_fetch_draft_csv_v03.py \
    -t LITE MU PLTR NVDA \
    -y 8 \
    -o sec_out \
    -u "YourProject/0.1 (your_email@example.com)"

補足
  ・--tickers と --ticker-file は併用可能
  ・この段階では年数を「揃えない」（素材取得が目的）

"""

from __future__ import annotations
# }}}

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

SEC_HEADERS = {
    # IMPORTANT: SEC requests a descriptive User-Agent with contact info.
    # NOTE: Do NOT set the "Host" header manually.
    # If you set Host=data.sec.gov while requesting https://www.sec.gov/..., some environments will
    # receive a 404 (because the Host header and URL host disagree).
    "User-Agent": "kana-fundamental-playground/0.1 (contact: you@example.com)",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Optional: let the user override the User-Agent from CLI without editing code.
# Example:
#   python3 sec_fetch_draft_csv.py --user-agent "YourProject/0.1 (email@example.com)" ...

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
TICKER_CIK_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# XBRL tags we try (first match wins)
TAG_REVENUE = [("us-gaap", "Revenues"), ("ifrs-full", "Revenue")]
TAG_EBIT = [("us-gaap", "OperatingIncomeLoss")]
TAG_CFO = [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")]
TAG_CAPEX = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
]
TAG_SBC = [("us-gaap", "ShareBasedCompensation")]


def _sleep_polite():
    # SEC rate limits; keep it gentle.
    time.sleep(0.25)


def _download_json(url: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    _sleep_polite()
    return data


def _load_ticker_cik_map(cache_dir: Path) -> Dict[str, str]:
    """Load mapping TICKER -> CIK (no zero padding).

    Primary source: company_tickers.json
    Fallback: company_tickers_exchange.json

    Why fallback?
      - Some environments get a misleading 404 when request headers are malformed.
      - SEC occasionally changes or redirects endpoints.
    """

    def _parse_mapping(data: object) -> Dict[str, str]:
        m: Dict[str, str] = {}
        # company_tickers.json: dict keyed by integer strings
        if isinstance(data, dict):
            items = data.values()
        # company_tickers_exchange.json: sometimes list, sometimes dict with 'data'
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for v in items:
            if not isinstance(v, dict):
                continue
            t = str(v.get("ticker", v.get("Ticker", ""))).upper().strip()
            cik = v.get("cik_str", v.get("cik", v.get("CIK", "")))
            cik = str(cik).strip()
            if t and cik:
                m[t] = cik
        return m

    # Try primary
    try:
        cache_path = cache_dir / "company_tickers.json"
        data = _download_json(TICKER_CIK_URL, cache_path)
        m = _parse_mapping(data)
        if m:
            return m
    except requests.HTTPError as e:
        print(f"[WARN] Failed to fetch company_tickers.json ({e}). Trying exchange file...")

    # Fallback
    cache_path2 = cache_dir / "company_tickers_exchange.json"
    data2 = _download_json(TICKER_CIK_EXCHANGE_URL, cache_path2)
    m2 = _parse_mapping(data2)
    if not m2:
        raise SystemExit(
            "Could not build ticker->CIK mapping from SEC. "
            "Tip: set --user-agent with a real contact, and check outbound access to sec.gov."
        )
    return m2


def _pick_fact_series(facts: dict, namespace: str, tag: str) -> Optional[List[dict]]:
    try:
        node = facts["facts"][namespace][tag]["units"]
    except Exception:
        return None
    # prefer USD
    for unit in ["USD", "usd"]:
        if unit in node:
            return node[unit]
    # fallback: first unit
    for _, arr in node.items():
        return arr
    return None


def _extract_annual(series: List[dict]) -> pd.DataFrame:
    """
    Convert companyfacts array into annual rows.
    We prefer:
      - fp == "FY"
      - form in 10-K / 20-F / 40-F
    Then group by fiscal year end year (fy) if present, else by end date year.
    """
    if not series:
        return pd.DataFrame(columns=["Year", "value"])

    df = pd.DataFrame(series)
    # normalize columns we need
    for c in ["val", "fy", "fp", "form", "end", "start"]:
        if c not in df.columns:
            df[c] = None

    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])

    # filter FY rows when possible
    df_fy = df[df["fp"].astype(str).str.upper() == "FY"].copy()
    if not df_fy.empty:
        df = df_fy

    # prefer annual forms
    annual_forms = {"10-K", "20-F", "40-F"}
    df_form = df[df["form"].astype(str).isin(annual_forms)].copy()
    if not df_form.empty:
        df = df_form

    # derive year
    if df["fy"].notna().any():
        df["Year"] = pd.to_numeric(df["fy"], errors="coerce")
    else:
        df["Year"] = pd.to_datetime(df["end"], errors="coerce").dt.year

    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # in case of duplicates per year, keep the latest 'end'
    df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.sort_values(["Year", "end_dt"])
    df = df.groupby("Year", as_index=False).tail(1)

    return df[["Year", "val"]].rename(columns={"val": "value"})


def _get_tag_value(facts: dict, tag_list: List[Tuple[str, str]]) -> pd.DataFrame:
    for ns, tag in tag_list:
        series = _pick_fact_series(facts, ns, tag)
        if series:
            return _extract_annual(series)
    return pd.DataFrame(columns=["Year", "value"])


def build_draft_for_ticker(ticker: str, cik: str, cache_dir: Path) -> pd.DataFrame:
    cik10 = str(cik).zfill(10)
    cache_path = cache_dir / f"companyfacts_{ticker}_{cik10}.json"
    facts = _download_json(COMPANYFACTS_URL.format(cik10=cik10), cache_path)

    company = facts.get("entityName", ticker)

    rev = _get_tag_value(facts, TAG_REVENUE).rename(columns={"value": "Revenue"})
    ebit = _get_tag_value(facts, TAG_EBIT).rename(columns={"value": "EBIT"})
    cfo = _get_tag_value(facts, TAG_CFO).rename(columns={"value": "CFO"})
    capex = _get_tag_value(facts, TAG_CAPEX).rename(columns={"value": "CapEx"})
    sbc = _get_tag_value(facts, TAG_SBC).rename(columns={"value": "SBC"})

    # merge on Year
    out = rev.merge(ebit, on="Year", how="outer")
    out = out.merge(cfo, on="Year", how="outer")
    out = out.merge(capex, on="Year", how="outer")
    out = out.merge(sbc, on="Year", how="outer")

    # CapEx in SEC is usually a positive cash outflow number. Keep positive here.
    # Compute FCF = CFO - CapEx
    out["FCF"] = pd.to_numeric(out["CFO"], errors="coerce") - pd.to_numeric(out["CapEx"], errors="coerce")

    out.insert(0, "Company", company)
    out.insert(0, "Ticker", ticker)

    # Optional columns that financial_eval can accept (left blank for manual fill)
    for c in ["Price", "MktCap_B", "SharesOut_B"]:
        out[c] = pd.NA

    # order columns
    cols = ["Ticker", "Company", "Year", "Revenue", "EBIT", "FCF", "CapEx", "SBC", "Price", "MktCap_B", "SharesOut_B"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].sort_values("Year")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('-t',"--tickers", nargs="*", default=[])
    #ap.add_argument('-t',"--ticker-file", default="yy00_my_tickers.txt", help="One ticker per line")
    ap.add_argument('-f',"--ticker-file", default=None, help="One ticker per line")
    ap.add_argument('-y',"--years", type=int, default=8, help="Keep last N years")
    ap.add_argument('-o',"--outdir", default="sec_out")
    ap.add_argument('-u',"--user-agent", default=None, help="Override SEC User-Agent header")
    args = ap.parse_args()

    if args.user_agent:
        SEC_HEADERS["User-Agent"] = args.user_agent

    tickers: List[str] = [t.upper() for t in args.tickers if t.strip()]
    if args.ticker_file:
        lines = Path(args.ticker_file).read_text(encoding="utf-8").splitlines()
        tickers += [ln.strip().upper() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers or --ticker-file.")

    outdir = Path(args.outdir)
    cache_dir = outdir / "sec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t2c = _load_ticker_cik_map(cache_dir)

    frames = []
    missing = []
    for t in tickers:
        cik = t2c.get(t)
        if not cik:
            missing.append(t)
            continue
        try:
            df = build_draft_for_ticker(t, cik, cache_dir)
            if args.years and df["Year"].notna().any():
                maxy = int(df["Year"].max())
                df = df[df["Year"] >= maxy - (args.years - 1)]
            frames.append(df)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
            continue

    if missing:
        print("[WARN] No CIK found for:", ", ".join(missing))

    if not frames:
        raise SystemExit("No data fetched.")

    #out = pd.concat(frames, ignore_index=True)

    ##251219
    # Drop empty frames and all-NA columns to avoid FutureWarning on concat dtype inference
    clean = []
    for df in frames:
        if df is None or df.empty:
            continue
        df2 = df.dropna(axis=1, how="all")
        ## if it became empty (all columns were NA), skip it
        if df2.empty:
            continue
        clean.append(df2)

    if clean:
        out = pd.concat(clean, ignore_index=True)
    else:
        out = pd.DataFrame()
    #---------------------------------------


    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "draft_input.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"OK: wrote {outdir/'draft_input.csv'}")


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
