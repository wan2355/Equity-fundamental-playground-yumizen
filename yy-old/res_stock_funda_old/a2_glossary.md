# Glossary (Full)

| 日本語/Key | English | 説明 |
|---|---|---|
| 時価総額（MC） | Market Cap (MC) | 企業全体の値段。MC=株価×発行株式数（本ツールではMktCap_Bを使用） |
| MC/Rev | MC/Rev (Market Cap to Revenue) | MC/Rev=MC/Revenue。売上1ドルをいくらで買っているか（alias: PS, PS_Latest） |
| PFCF | MC/FCF (Price to Free Cash Flow) | PFCF=MC/FCF。FCFが0以下なら倍率の意味が崩れるので空欄 |
| POwnerFCF | MC/OwnerFCF (Price to Owner FCF) | POwnerFCF=MC/(FCF-SBC)。株主に残る現金で評価。<=0は空欄 |
| Ticker | Ticker | 銘柄コード |
| Company | Company | 会社名 |
| Year | Year | 年度（FY） |
| MaxYear | MaxYear | 入力データの最新年 |
| WindowYears | WindowYears | 1サイクル=この年数。平均/最悪/成長をこの窓で計算 |
| WindowStartYear | WindowStartYear | 窓の開始年（MaxYear-(WindowYears-1)） |
| Revenue | Revenue | 売上（Billion USD） |
| EBIT | EBIT | 利息・税引前利益（本業利益に近い） |
| FCF | FCF (Free Cash Flow) | FCF=営業CF−設備投資。自由に使える現金の増減 |
| CapEx | CapEx (Capital Expenditures) | 設備投資（現金支出） |
| SBC | SBC (Stock-Based Compensation) | 株式報酬（株主の取り分が薄まるコスト） |
| OwnerFCF | Owner FCF | OwnerFCF=FCF−SBC（株主に実質残る現金） |
| Revenue_Avg | Revenue_Avg | WindowYears内の売上平均 |
| EBIT_Avg | EBIT_Avg | WindowYears内のEBIT平均 |
| FCF_Avg | FCF_Avg | WindowYears内のFCF平均 |
| EBIT/Rev_Avg | EBIT/Rev_Avg | EBIT/Rev（平均）。本ツール内部列名 alias: EBITMargin_Avg |
| EBIT/Rev_Min | EBIT/Rev_Min | EBIT/Rev（最悪）。不況耐性の目安。alias: EBITMargin_Min |
| FCF/Rev_Avg | FCF/Rev_Avg | FCF/Rev（平均）。alias: FCFMargin_Avg |
| FCF/Rev_Min | FCF/Rev_Min | FCF/Rev（最悪）。不況耐性の目安。alias: FCFMargin_Min |
| CapEx/Rev_Avg | CapEx/Rev_Avg | CapEx/Rev（平均）。装備コストの重さの目安。alias: CapExOverRev_Avg |
| SBC/Rev_Avg | SBC/Rev_Avg | SBC/Rev（平均）。株式報酬の重さの目安。alias: SBCOverRev_Avg |
| Rev_CAGR | Rev_CAGR | 売上CAGR（start/end>0のみ） |
| FCF_CAGR | FCF_CAGR | FCF CAGR（start/end>0のみ） |
| Price_Latest | Price_Latest | 最新年の株価（任意入力） |
| MktCap_B_Latest | MktCap_B_Latest | 最新年の時価総額（Billion）。TradingViewのMarket Capを推奨 |
| PS_Latest | PS_Latest | 最新年: MC/Rev（出力表記はMC/Rev。alias吸収） |
| PFCF_Latest | PFCF_Latest | 最新年: MC/FCF（出力表記はMC/FCF。FCF<=0は空欄） |
| POwnerFCF_Latest | POwnerFCF_Latest | 最新年: MC/OwnerFCF=MC/(FCF-SBC)（出力表記はMC/OwnerFCF。<=0は空欄） |
| Profile | Profile | 評価軸（CYCLE / SW / HW） |
| CYCLE | CYCLE profile | 景気循環（不況）でも崩れにくい軸。FCF/Revや最悪年の耐性、投資負担（CapEx/Rev）、割高さ（MC/Rev, MC/FCF）をバランス良く見る |
| SW | SW profile | ソフトウェア/軽装備型を想定した軸。成長（Rev_CAGR, FCF_CAGR）と株主取り分の薄まり（SBC/Rev）を重視し、バリュエーションもチェック |
| HW | HW profile | ハードウェア/重装備型を想定した軸。EBIT/Revと最悪年の耐性に加え、CapEx/Rev（装備負担）を強めに評価し、割高さも確認 |
| Score_0to100 | Score_0to100 | 総合点（0-100）。出力では整数 |
| Rating | Rating | A〜E評価（Scoreを丸め） |
| S_xxx | S_xxx | 各指標の順位スコア（1-5） |