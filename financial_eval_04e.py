#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
financial_eval.py (COMPLETE v3)

目的:
- 入力CSV（年次データ）から、複数年（WindowYears=サイクル長）で
  平均/最悪/成長/バリュエーションを計算し、3軸(CYCLE/SW/HW)でスコア化します。

出力（ユーザー要望に合わせ「prettyのみ」）:
- metrics_pretty.csv
- scorecards_pretty.csv（Scoreは整数）
- summary_A4.csv / summary_A4.md（重要指標だけ）
- charts/*.png（重要指標の比較グラフ）
- glossary_full.md（出力列の完全用語集：日本語/英語）
- README.md

入力CSV（UTF-8/LF推奨）
必須列:
- Ticker, Company, Year, Revenue, EBIT, FCF

任意列:
- CapEx, SBC, Price, MktCap_B, SharesOut_B

単位:
- Revenue/EBIT/FCF/CapEx/SBC/MktCap_B: Billion USD（B）
- Price: USD/株
- SharesOut_B: Billion shares（B株）※任意

重要: Pとは何か
- 本ツールの基準は **時価総額（Market Cap / MktCap）**
  MktCap = Price * SharesOut
  （TradingViewのMarket Capを MktCap_B として直接入力するのが簡単）

定義（Latest=最新年）
- PS_Latest        = MktCap / Rev.（売上）
- PFCF_Latest      = MktCap / FCF（FCF<=0 は意味が壊れるので空欄）
- POwnerFCF_Latest = MktCap / (FCF - SBC)（<=0は空欄）

実行:
python3 financial_eval.py --input input.csv --window-years 5 --outdir out
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm_col(c: str) -> str:
    c = c.replace("\r\n", "\n").replace("\r", "\n")
    c = c.replace("\n", " ").strip()
    c = re.sub(r"\s+", " ", c)
    return c


def _canonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    new = {}
    for c in cols:
        cjp = c
        cl = c.lower()

        if cjp == "Ticker":
            new[c] = "Ticker"; continue
        if cjp == "Company":
            new[c] = "Company"; continue
        if cjp == "Year":
            new[c] = "Year"; continue
        if cjp == "Revenue":
            new[c] = "Revenue"; continue
        if cjp == "EBIT":
            new[c] = "EBIT"; continue
        if cjp == "FCF":
            new[c] = "FCF"; continue

        if ("capex" in cl) or ("設備投資" in cjp):
            new[c] = "CapEx"; continue
        if ("sbc" in cl) or ("stock-based" in cl) or ("株式報酬" in cjp):
            new[c] = "SBC"; continue
        if ("price" in cl) or ("株価" in cjp):
            new[c] = "Price"; continue
        if ("mktcap" in cl) or ("market cap" in cl) or ("時価総額" in cjp):
            new[c] = "MktCap_B"; continue
        if ("shares" in cl) or ("発行" in cjp) or ("shares out" in cl):
            new[c] = "SharesOut_B"; continue

    if new:
        df = df.rename(columns=new)

    required = ["Ticker", "Company", "Year", "Revenue", "EBIT", "FCF"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["CapEx", "SBC", "Price", "MktCap_B", "SharesOut_B"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


def read_input_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=",", engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=",", engine="python", encoding="cp932")

    df.columns = [_norm_col(c) for c in df.columns]
    df = _canonize_columns(df)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    num_cols = ["Revenue", "EBIT", "FCF", "CapEx", "SBC", "Price", "MktCap_B", "SharesOut_B"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Ticker", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    miss = df["MktCap_B"].isna() & df["Price"].notna() & df["SharesOut_B"].notna()
    df.loc[miss, "MktCap_B"] = df.loc[miss, "Price"] * df.loc[miss, "SharesOut_B"]

    return df


def cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    n = len(s) - 1
    if n <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1.0 / n) - 1.0


def compute_metrics(df: pd.DataFrame, window_years: int) -> pd.DataFrame:
    out = []
    for tkr, g in df.groupby("Ticker"):
        g = g.sort_values("Year")
        max_year = int(g["Year"].max())
        win_start = max_year - (window_years - 1)
        gw = g[g["Year"] >= win_start].copy()

        def safe_mean(x: pd.Series) -> float:
            return float(x.mean()) if x.notna().any() else np.nan

        rev_avg = safe_mean(gw["Revenue"])
        ebit_avg = safe_mean(gw["EBIT"])
        fcf_avg = safe_mean(gw["FCF"])

        ebit_margin = (gw["EBIT"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        fcf_margin = (gw["FCF"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        capex_ratio = (gw["CapEx"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        sbc_ratio = (gw["SBC"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)

        ebit_margin_avg = safe_mean(ebit_margin)
        fcf_margin_avg = safe_mean(fcf_margin)
        capex_ratio_avg = safe_mean(capex_ratio)
        sbc_ratio_avg = safe_mean(sbc_ratio)

        ebit_margin_min = float(ebit_margin.min()) if ebit_margin.notna().any() else np.nan
        fcf_margin_min = float(fcf_margin.min()) if fcf_margin.notna().any() else np.nan

        rev_cagr = cagr(gw.set_index("Year")["Revenue"])
        fcf_cagr = cagr(gw.set_index("Year")["FCF"])
        owner_fcf_series = (gw.set_index("Year")["FCF"] - gw.set_index("Year")["SBC"]) if ("FCF" in gw and "SBC" in gw) else None
        owner_fcf_cagr = cagr(owner_fcf_series) if owner_fcf_series is not None else np.nan


        gl = g[g["Year"] == max_year].iloc[-1]
        mktcap = gl["MktCap_B"]
        price = gl["Price"]

        owner_fcf = (gl["FCF"] - gl["SBC"]) if pd.notna(gl["FCF"]) and pd.notna(gl["SBC"]) else np.nan

        ps = (mktcap / gl["Revenue"]) if pd.notna(mktcap) and pd.notna(gl["Revenue"]) and gl["Revenue"] != 0 else np.nan
        pfcf = (mktcap / gl["FCF"]) if pd.notna(mktcap) and pd.notna(gl["FCF"]) and gl["FCF"] > 0 else np.nan
        p_owner_fcf = (mktcap / owner_fcf) if pd.notna(mktcap) and pd.notna(owner_fcf) and owner_fcf > 0 else np.nan
        peg_ps_latest = (ps / (rev_cagr*100.0)) if pd.notna(ps) and pd.notna(rev_cagr) and rev_cagr > 0 else np.nan
        peg_ownerfcf_latest = (p_owner_fcf / (owner_fcf_cagr*100.0)) if pd.notna(p_owner_fcf) and pd.notna(owner_fcf_cagr) and owner_fcf_cagr > 0 else np.nan

        out.append({
            "Ticker": tkr,
            "Company": str(g["Company"].iloc[0]),
            "MaxYear": max_year,
            "WindowYears": window_years,
            "WindowStartYear": win_start,

            "Revenue_Avg": rev_avg,
            "EBIT_Avg": ebit_avg,
            "FCF_Avg": fcf_avg,

            "EBITMargin_Avg": ebit_margin_avg,
            "EBITMargin_Min": ebit_margin_min,
            "FCFMargin_Avg": fcf_margin_avg,
            "FCFMargin_Min": fcf_margin_min,

            "CapExOverRev_Avg": capex_ratio_avg,
            "SBCOverRev_Avg": sbc_ratio_avg,

            "Rev_CAGR": rev_cagr,
            "FCF_CAGR": fcf_cagr,
            "OwnerFCF_CAGR": owner_fcf_cagr,
            "PEG_PS_Latest": peg_ps_latest,
            "PEG_OwnerFCF_Latest": peg_ownerfcf_latest,
            "OwnerFCF_CAGR": owner_fcf_cagr,
            "PEG_PS_Latest": peg_ps_latest,
            "PEG_OwnerFCF_Latest": peg_ownerfcf_latest,

            "Price_Latest": price,
            "MktCap_B_Latest": mktcap,

            "PS_Latest": ps,
            "PFCF_Latest": pfcf,
            "POwnerFCF_Latest": p_owner_fcf,
        })

    return pd.DataFrame(out).sort_values("Ticker").reset_index(drop=True)


def score_1to5(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.Series(values, dtype="float64")
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(np.where(mask, 3.0, np.nan), index=s.index, dtype="float64")
    if higher_is_better:
        r = s[mask].rank(ascending=False, method="min")
    else:
        r = s[mask].rank(ascending=True, method="min")
    n = int(mask.sum())
    score = 1 + 4 * (1 - (r - 1) / (n - 1))
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[mask] = score
    return out


def build_scorecard(metrics: pd.DataFrame, profile: str) -> pd.DataFrame:
    profile = profile.lower()

    if profile == "cycle":
        defs: List[Tuple[str, bool, float]] = [
            ("FCFMargin_Avg", True, 0.25),
            ("FCFMargin_Min", True, 0.20),
            ("EBITMargin_Avg", True, 0.15),
            ("CapExOverRev_Avg", False, 0.10),
            ("PS_Latest", False, 0.10),
            ("PFCF_Latest", False, 0.10),
            ("Rev_CAGR", True, 0.10),
        ]
    elif profile == "sw":
        defs = [
            ("FCFMargin_Avg", True, 0.25),
            ("EBITMargin_Avg", True, 0.10),
            ("Rev_CAGR", True, 0.15),
            ("FCF_CAGR", True, 0.10),
            ("SBCOverRev_Avg", False, 0.10),
            ("PS_Latest", False, 0.15),
            ("PFCF_Latest", False, 0.15),
        ]
    elif profile == "hw":
        defs = [
            ("EBITMargin_Avg", True, 0.25),
            ("EBITMargin_Min", True, 0.15),
            ("CapExOverRev_Avg", False, 0.20),
            ("FCFMargin_Avg", True, 0.15),
            ("PS_Latest", False, 0.10),
            ("PFCF_Latest", False, 0.15),
        ]
    else:
        raise ValueError("profile must be one of: cycle, sw, hw")

    m = metrics.copy().reset_index(drop=True)
    for col, hib, w in defs:
        m[f"S_{col}"] = score_1to5(m[col], higher_is_better=hib)
        m[f"W_{col}"] = w

    s_cols = [f"S_{col}" for col, _, _ in defs]
    w_cols = [f"W_{col}" for col, _, _ in defs]

    S = m[s_cols].to_numpy(dtype=float)
    W = m[w_cols].to_numpy(dtype=float)
    mask = ~np.isnan(S)

    num = np.nansum(S * W, axis=1)
    denom = np.nansum(W * mask, axis=1)

    m["Score_1to5"] = np.where(denom > 0, num / denom, np.nan)
    m["Score_0to100"] = m["Score_1to5"] * 20

    def rating(x: float) -> str:
        if pd.isna(x):
            return ""
        if x >= 85: return "A"
        if x >= 70: return "B"
        if x >= 55: return "C"
        if x >= 40: return "D"
        return "E"

    m["Rating"] = m["Score_0to100"].apply(rating)
    m["Profile"] = profile.upper()

    core = ["Profile", "Ticker", "Company", "MaxYear", "WindowYears", "WindowStartYear", "Score_0to100", "Rating"]
    metrics_cols = [col for col, _, _ in defs]
    score_cols = [f"S_{col}" for col, _, _ in defs]
    return m[core + metrics_cols + score_cols]


def _fmt_num_general(v):
    if pd.isna(v):
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.3f}"


def _fmt_score_int(v):
    if pd.isna(v):
        return ""
    try:
        return str(int(round(float(v))))
    except Exception:
        return ""


def df_pretty_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_num_general)
    return out


def df_pretty_scorecards(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "Score_0to100":
            out[c] = out[c].map(_fmt_score_int)
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_num_general)
    return out



# ----------------------------
# Naming simplification layer
# ----------------------------
# 目的:
# - pandas内部の列名（例: FCFMargin_Avg）を壊さず、
#   出力/表示では数式表記（例: FCF/Rev_Avg）に統一する。
# - 「margin」という単語は出力側で使わない。

# pandas列名(旧/alias) -> 出力列名(数式/canonical)
OUTPUT_COL_RENAME = {
    # valuation
    "PFCF_Latest": "MC/FCF",
    "POwnerFCF_Latest": "MC/OwnerFCF",

    # ratios (avoid 'margin' wording in outputs)
    "EBITMargin_Avg": "EBIT/Rev_Avg",
    "EBITMargin_Min": "EBIT/Rev_Min",
    "FCFMargin_Avg": "FCF/Rev_Avg",
    "FCFMargin_Min": "FCF/Rev_Min",
    "CapExOverRev_Avg": "CapEx/Rev_Avg",
    "SBCOverRev_Avg": "SBC/Rev_Avg",

    # keep these readable too
    "PS_Latest": "MC/Rev",
}

# 軸ラベル/凡例用（必要ならさらに短縮）
OUTPUT_LABEL_MAP = {
    "MC/FCF": "MC/FCF",
    "MC/OwnerFCF": "MC/OwnerFCF",
    "MC/Rev": "MC/Rev",
    "EBITMargin_Avg": "EBIT/Rev (Avg)",
    "EBITMargin_Min": "EBIT/Rev (Min)",
    "FCFMargin_Avg": "FCF/Rev (Avg)",
    "FCFMargin_Min": "FCF/Rev (Min)",
    "CapExOverRev_Avg": "CapEx/Rev (Avg)",
    "SBCOverRev_Avg": "SBC/Rev (Avg)",
    "PS_Latest": "MC/Rev",
    "PFCF_Latest": "MC/FCF",
    "POwnerFCF_Latest": "MC/OwnerFCF",
}

def rename_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """出力ファイル用に列名を数式表記へ統一（内部計算の列名は維持）。"""
    return df.rename(columns=OUTPUT_COL_RENAME)


def write_utf8_lf(path: Path, text: str) -> None:
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


A4_COLUMNS = [
    "Ticker","Company","MaxYear","WindowYears",
    "Score_CYCLE","Rating_CYCLE","Score_SW","Rating_SW","Score_HW","Rating_HW",
    "EBITMargin_Avg","EBITMargin_Min","FCFMargin_Avg","FCFMargin_Min",
    "CapExOverRev_Avg","SBCOverRev_Avg","Rev_CAGR",
    #"MktCap_B_Latest","PS_Latest","PFCF_Latest","POwnerFCF_Latest",
    "MktCap_B_Latest","PS_Latest","MC/FCF","MC/OwnerFCF",
    "PEG_PS_Latest","PEG_OwnerFCF_Latest",
]

SUMMARY2_COLUMNS = [
    "Ticker",
    "Rating_CYCLE","Score_CYCLE","Score_SW","Score_HW",
    "PEG_PS_Latest","PEG_OwnerFCF_Latest",
    #"PFCF_Latest","POwnerFCF_Latest",
    "MC/FCF","MC/OwnerFCF",
    "EBITMargin_Avg","FCFMargin_Avg",
    "CapExOverRev_Avg","SBCOverRev_Avg",
    "Rev_CAGR",
    "MaxYear","Company",
]


def build_summary_A4(metrics: pd.DataFrame, scorecards: pd.DataFrame) -> pd.DataFrame:
    sc = scorecards[["Profile","Ticker","Score_0to100","Rating"]].copy()
    sc_piv = sc.pivot(index="Ticker", columns="Profile", values=["Score_0to100","Rating"])
    sc_piv.columns = [f"{a}_{b}" for a,b in sc_piv.columns]
    sc_piv = sc_piv.reset_index()

    merged = metrics.merge(sc_piv, on="Ticker", how="left")
    merged = merged.rename(columns={
        "Score_0to100_CYCLE":"Score_CYCLE","Rating_CYCLE":"Rating_CYCLE",
        "Score_0to100_SW":"Score_SW","Rating_SW":"Rating_SW",
        "Score_0to100_HW":"Score_HW","Rating_HW":"Rating_HW",
    })

    OUT_COL_RENAME = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    merged = merged.rename(columns=OUT_COL_RENAME)
    cols = [c for c in A4_COLUMNS if c in merged.columns]
    return merged[cols].sort_values(["Score_CYCLE","Score_SW","Score_HW"], ascending=False)

def build_summary2(metrics: pd.DataFrame, scorecards: pd.DataFrame) -> pd.DataFrame:
    """
    summary2: 比較・投資判断向け（重要指標を厳選）
    - 並びはユーザー指定順
    - Score列は整数表示（pretty側で整形）
    """
    sc = scorecards[["Profile","Ticker","Score_0to100","Rating"]].copy()
    sc_piv = sc.pivot(index="Ticker", columns="Profile", values=["Score_0to100","Rating"])
    sc_piv.columns = [f"{a}_{b}" for a,b in sc_piv.columns]
    sc_piv = sc_piv.reset_index()

    merged = metrics.merge(sc_piv, on="Ticker", how="left")
    merged = merged.rename(columns={
        "Score_0to100_CYCLE":"Score_CYCLE","Rating_CYCLE":"Rating_CYCLE",
        "Score_0to100_SW":"Score_SW","Rating_SW":"Rating_SW",
        "Score_0to100_HW":"Score_HW","Rating_HW":"Rating_HW",
    })

    OUT_COL_RENAME = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    merged = merged.rename(columns=OUT_COL_RENAME)
    cols = [c for c in SUMMARY2_COLUMNS if c in merged.columns]
    # Score優先で並べる（空欄でも落ちないように）
    sort_cols = [c for c in ["Score_CYCLE","Score_SW","Score_HW"] if c in cols]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=False)
    return merged[cols]



def save_barh(df: pd.DataFrame, value_col: str, title: str, unit: str, outpath: Path):
    d = df[["Ticker", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna().sort_values(value_col, ascending=True)
    if d.empty:
        return
    plt.figure(figsize=(8.27, 11.69/3.2))
    plt.barh(d["Ticker"], d[value_col])
    plt.title(title, fontsize=10)
    #label_map = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    xlab = OUTPUT_LABEL_MAP.get(value_col, value_col)
    if unit:
        plt.xlabel(f"{xlab} ({unit})")
    else:
        plt.xlabel(xlab)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def write_glossary_full(outdir: Path):
    rows = []
    def add(jp, en, desc): rows.append((jp,en,desc))

    add("時価総額（MC）","Market Cap (MC)","企業全体の値段。MC=株価×発行株式数（本ツールではMktCap_Bを使用）")
    add("MC/Rev","MC/Rev (Market Cap to Revenue)","MC/Rev=MC/Revenue。売上1ドルをいくらで買っているか（alias: PS, PS_Latest）")
    add("PFCF","MC/FCF (Price to Free Cash Flow)","PFCF=MC/FCF。FCFが0以下なら倍率の意味が崩れるので空欄")
    add("POwnerFCF","MC/OwnerFCF (Price to Owner FCF)","POwnerFCF=MC/(FCF-SBC)。株主に残る現金で評価。<=0は空欄")

    add("Ticker","Ticker","銘柄コード")
    add("Company","Company","会社名")
    add("Year","Year","年度（FY）")
    add("MaxYear","MaxYear","入力データの最新年")
    add("WindowYears","WindowYears","1サイクル=この年数。平均/最悪/成長をこの窓で計算")
    add("WindowStartYear","WindowStartYear","窓の開始年（MaxYear-(WindowYears-1)）")

    add("Revenue","Revenue","売上（Billion USD）")
    add("EBIT","EBIT","利息・税引前利益（本業利益に近い）")
    add("FCF","FCF (Free Cash Flow)","FCF=営業CF−設備投資。自由に使える現金の増減")
    add("CapEx","CapEx (Capital Expenditures)","設備投資（現金支出）")
    add("SBC","SBC (Stock-Based Compensation)","株式報酬（株主の取り分が薄まるコスト）")
    add("OwnerFCF","Owner FCF","OwnerFCF=FCF−SBC（株主に実質残る現金）")

    add("Revenue_Avg","Revenue_Avg","WindowYears内の売上平均")
    add("EBIT_Avg","EBIT_Avg","WindowYears内のEBIT平均")
    add("FCF_Avg","FCF_Avg","WindowYears内のFCF平均")

    add("EBIT/Rev_Avg","EBIT/Rev_Avg","EBIT/Rev（平均）。本ツール内部列名 alias: EBITMargin_Avg")
    add("EBIT/Rev_Min","EBIT/Rev_Min","EBIT/Rev（最悪）。不況耐性の目安。alias: EBITMargin_Min")
    add("FCF/Rev_Avg","FCF/Rev_Avg","FCF/Rev（平均）。alias: FCFMargin_Avg")
    add("FCF/Rev_Min","FCF/Rev_Min","FCF/Rev（最悪）。不況耐性の目安。alias: FCFMargin_Min")

    add("CapEx/Rev_Avg","CapEx/Rev_Avg","CapEx/Rev（平均）。装備コストの重さの目安。alias: CapExOverRev_Avg")
    add("SBC/Rev_Avg","SBC/Rev_Avg","SBC/Rev（平均）。株式報酬の重さの目安。alias: SBCOverRev_Avg")

    add("Rev_CAGR","Rev_CAGR","売上CAGR（start/end>0のみ）")
    add("FCF_CAGR","FCF_CAGR","FCF CAGR（start/end>0のみ）")

    add("Price_Latest","Price_Latest","最新年の株価（任意入力）")
    add("MktCap_B_Latest","MktCap_B_Latest","最新年の時価総額（Billion）。TradingViewのMarket Capを推奨")
    add("PS_Latest","PS_Latest","最新年: MC/Rev（出力表記はMC/Rev。alias吸収）")
    add("PFCF_Latest","PFCF_Latest","最新年: MC/FCF（出力表記はMC/FCF。FCF<=0は空欄）")
    add("POwnerFCF_Latest","POwnerFCF_Latest","最新年: MC/OwnerFCF=MC/(FCF-SBC)（出力表記はMC/OwnerFCF。<=0は空欄）")

    add("Profile","Profile","評価軸（CYCLE / SW / HW）")
    add("CYCLE","CYCLE profile","景気循環（不況）でも崩れにくい軸。FCF/Revや最悪年の耐性、投資負担（CapEx/Rev）、割高さ（MC/Rev, MC/FCF）をバランス良く見る")
    add("SW","SW profile","ソフトウェア/軽装備型を想定した軸。成長（Rev_CAGR, FCF_CAGR）と株主取り分の薄まり（SBC/Rev）を重視し、バリュエーションもチェック")
    add("HW","HW profile","ハードウェア/重装備型を想定した軸。EBIT/Revと最悪年の耐性に加え、CapEx/Rev（装備負担）を強めに評価し、割高さも確認")
    add("Score_0to100","Score_0to100","総合点（0-100）。出力では整数")
    add("Rating","Rating","A〜E評価（Scoreを丸め）")
    add("S_xxx","S_xxx","各指標の順位スコア（1-5）")

    md = ["# Glossary (Full)\n",
          "| 日本語/Key | English | 説明 |",
          "|---|---|---|"]
    for jp,en,desc in rows:
        md.append(f"| {jp} | {en} | {desc} |")
    write_utf8_lf(outdir/"a2_glossary.md", "\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-w","--window-years", type=int, default=5)
    ap.add_argument("-o","--outdir", default="res_stock_funda")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"charts").mkdir(parents=True, exist_ok=True)

    df = read_input_csv(Path(args.input))
    metrics = compute_metrics(df, window_years=args.window_years)

    sc_cycle = build_scorecard(metrics, "cycle")
    sc_sw = build_scorecard(metrics, "sw")
    sc_hw = build_scorecard(metrics, "hw")
    scorecards = pd.concat([sc_cycle, sc_sw, sc_hw], ignore_index=True)

    # pretty only
    metrics_pretty = df_pretty_metrics(metrics)
    scorecards_pretty = df_pretty_scorecards(scorecards)
    metrics_pretty = rename_output_columns(metrics_pretty)
    scorecards_pretty = rename_output_columns(scorecards_pretty)
    metrics_pretty.to_csv(outdir/"metrics_pretty.csv", index=False, encoding="utf-8", lineterminator="\n")
    scorecards_pretty.to_csv(outdir/"scorecards_pretty.csv", index=False, encoding="utf-8", lineterminator="\n")

    # summary A4
    summary = build_summary_A4(metrics, scorecards)
    summary_pretty = df_pretty_metrics(summary.copy())
    for c in ["Score_CYCLE","Score_SW","Score_HW"]:
        if c in summary_pretty.columns:
            summary_pretty[c] = summary_pretty[c].map(lambda x: _fmt_score_int(x) if x!="" else "")
    summary_pretty = rename_output_columns(summary_pretty)

    summary_pretty.to_csv(outdir/"summary_A4.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_utf8_lf(outdir/"summary_A4.md", summary_pretty.to_markdown(index=False))

    ###--- debug ---
    #print('\n debug \n')
    #print("summary cols:", list(summary.columns))

    # summary2 (重要指標のみ・比較向け)
    summary2 = build_summary2(metrics, scorecards)
    summary2_pretty = df_pretty_metrics(summary2.copy())
    for c in ["Score_CYCLE","Score_SW","Score_HW"]:
        if c in summary2_pretty.columns:
            summary2_pretty[c] = summary2_pretty[c].map(lambda x: _fmt_score_int(x) if x!="" else "")
    summary2_pretty = rename_output_columns(summary2_pretty)

    summary2_pretty.to_csv(outdir/"00_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_utf8_lf(outdir/"00_summary.md", summary2_pretty.to_markdown(index=False))

    ## charts
    charts_base = summary.copy()

    save_barh(charts_base, "Score_CYCLE", "Score (CYCLE)", "points", outdir/"charts"/"00_score_cycle.png")
    save_barh(charts_base, "MktCap_B_Latest", "Market_Cap", "B USD", outdir/"charts"/"mc_marcket_cap.png")
    save_barh(charts_base, "FCFMargin_Avg", "FCF/Rev (Avg)", "ratio", outdir/"charts"/"fcf_rev_ratio.png")
    save_barh(charts_base, "CapExOverRev_Avg", "CapEx/Rev (Avg)", "ratio", outdir/"charts"/"capex_over_rev.png")
    save_barh(charts_base, "PS_Latest", "(MC / Rev.)","ratio" , outdir/"charts"/"ps_MC_Rev_ratio.png")
    save_barh(charts_base, "PEG_PS_Latest", "PEG-like: (MC / Rev.) / RevGrowth%", "ratio", outdir/"charts"/"01_peg_ps.png")
    ##save_barh(charts_base, "PFCF_Latest", "MC/FCF Latest (MC / FCF)", "ratio", outdir/"charts"/"02_pfcf_latest.png")
    ##save_barh(charts_base, "POwnerFCF_Latest", "MC/OwnerFCF Latest (P / (FCF-SBC))", "ratio", outdir/"charts"/"03_pownerfcf_latest.png")
    ##save_barh(charts_base, "PFCF_Latest", "MC/FCF (x)", "ratio", outdir/"charts"/"02_pfcf_latest.png")
    ##save_barh(charts_base, "POwnerFCF_Latest", "MC/OwnerFCF (x)", "ratio", outdir/"charts"/"03_pownerfcf_latest.png")
    save_barh(charts_base, "MC/FCF", "MC/FCF (x)", "ratio", outdir/"charts"/"02_pfcf_latest.png")
    save_barh(charts_base, "MC/OwnerFCF", "MC/OwnerFCF (x)", "ratio", outdir/"charts"/"03_MC_fcf_ratio.png")

    #save_barh(charts_base, "MktCap_B_Latest", "Market_Cap", "Billion $", outdir/"charts"/"marcket_cap.png")


    ## glossary + README
    write_glossary_full(outdir)

    ## 以下でglosarry.md, readme.md を作成している。
    ## このままで修正しないこと。
    readme = f"""# financial_eval.py (COMPLETE v3)

    ## 何がわかるか（重要）
    - **summary_A4.csv** を見れば、主要指標だけで比較できます（A4相当）
    - **scorecards_pretty.csv** で、CYCLE/SW/HWのスコア（0-100）を比較できます
    - **charts/** で、重要指標の横棒グラフ比較ができます

    ## 入力CSV
    必須: Ticker, Company, Year, Revenue, EBIT, FCF
    任意: CapEx, SBC, Price, MktCap_B, SharesOut_B

    単位:
    - Revenue/EBIT/FCF/CapEx/SBC/MktCap_B: Billion USD（B）
    - Price: USD/株
    - SharesOut_B: Billion shares（B株）

    ## P（Price）とは何か
    本ツールの **P は株価ではなく時価総額（Market Cap）** です。
    - MC = Market Cap = Price * SharesOut
    （TradingViewのMarket Capを MktCap_B に入れる運用が最も簡単です）

    ## 定義（Latest=最新年）
    - PS_Latest        = MC / Rev.（売上）
    - PFCF_Latest      = MC / FCF（FCF<=0は空欄）
    - POwnerFCF_Latest = MC / (FCF - SBC)（<=0は空欄）

    ## 実行
    ```bash
    python3 financial_eval.py --input input.csv --window-years {args.window_years} --outdir out
    ```

    出力:
    - out/metrics_pretty.csv
    - out/scorecards_pretty.csv
    - out/summary_A4.csv
    - out/charts/*.png
    - out/glossary_full.md
    """
    write_utf8_lf(outdir/"00_README.md", readme)

    print(f"OK: outputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
