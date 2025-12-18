# financial_eval.py (COMPLETE v3)

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
    python3 financial_eval.py --input input.csv --window-years 5 --outdir out
    ```

    出力:
    - out/metrics_pretty.csv
    - out/scorecards_pretty.csv
    - out/summary_A4.csv
    - out/charts/*.png
    - out/glossary_full.md
    