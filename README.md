# CRD Data Analysis

このリポジトリは、Collective Risk Dilemma（CRD）実験のデータ分析および可視化を行うためのPythonプログラムです。主にoTreeで収集した `all_apps_wide-<date>.csv` 形式のデータをもとに、グループごとの貢献行動や成功率を整理し、成果報酬の集計やグラフ描画、PayPayホワイトリストの作成を支援します。

---

## 📦 主なコンポーネント

* **`GetData` クラス**
  * `all_apps_wide` 形式のCSVを読み込み、グループ構成やアンケート、寄付額などを整理
  * 成功者ラベルの抽出、寄付額の整形、ターゲット達成状況の判定、離脱ラウンドの検出など豊富な集計メソッドを提供
* **`YahooTask` クラス**
  * Yahoo!クラウドソーシングの成果報酬TSVと実験データを突合
  * PayPayポイント換算、切り上げ処理、ホワイトリストTXTの自動出力、ポイント分布ヒストグラムの描画に対応
* **`GraphPlot` クラス**
  * `GetData` の結果を使い、個人・グループ・ラウンド別の貢献推移やターゲット達成状況など多数のグラフを生成
  * 生成したグラフは日付付きディレクトリ（例：`20250724_Figures/`）に自動保存（`savefig=False`なら画面表示）

---

## 📁 前提ディレクトリ構成

```
プロジェクト/
│
├─ all_apps_wide-YYYY-MM-DD.csv   ← oTreeのエクスポートCSV
├─ getdata.py                     ← 分析・可視化ロジック
├─ 3589457371.tsv                 ← Yahoo!クラウドソーシングの成果報酬TSV
└─ 20250724_Figures/ など          ← グラフ出力ディレクトリ（自動生成）
```

---

## 🔧 必要なライブラリ

* Python 3.9 以上推奨
* 必須ライブラリ：
  * pandas
  * numpy
  * matplotlib

```bash
pip install pandas numpy matplotlib
```

---

## 🧐 クラス構成と使い方

### ✅ `GetData` クラス

```python
from getdata import GetData

g = GetData(
    dirpath="./",                    # CSV や TSV が置かれているディレクトリ
    filename="all_apps_wide-2025-07-24.csv",
    debug=True                       # 任意：途中経過を表示
)

print(g)  # グループ構成、失敗理由、離脱状況などをまとめて表示

success_contrib = g.get_success_contribution()    # 成功者10ラウンド分の寄付額
group_targets = g.get_group_target()              # グループごとの目標額
dropouts = g.get_dropout_timing()                 # 離脱者とラウンド
surveys = g.get_surveys()                         # アンケート回答
```

主なメソッドの例：

* `pick_passed_comprehension_test()`：理解テスト合格者／未マッチ者のラベルを取得
* `set_contribution_by_group()`：グループ別の寄付推移DataFrameを返す
* `get_group_contribution(success_contrib=False)`：公共基金の最終額（成功者のみ/全員）
* `get_paidcost_label()` / `get_paidcost_num()`：情報購入状況と人数を集計

### ✅ `GraphPlot` クラス

```python
from getdata import GraphPlot

gp = GraphPlot(g, savefig=True, figtype="png")  # figtype は "png" または "eps"

gp.plot_total_contribution()                  # 累積貢献額（提案額込みは pledges=True）
gp.plot_contribution_per_round()              # ラウンドごとの平均寄付額
gp.plot_group_contrib()                       # グループ別の最終公共基金と目標額
gp.plot_individual_contrib()                  # 個人別の累積寄付額
gp.plot_pa_frequency()                        # 公共基金の分布
gp.plot_pa_pledge()                           # 公共基金と提案額の散布図
gp.plot_target_pledge()                       # 目標額と提案額の散布図
```

2条件を比較するグラフでは、`other_datacls` に別の `GetData` インスタンスを渡します。

```python
other_g = GetData(dirpath="./", filename="all_apps_wide-2025-08-07.csv")

gp_comp = GraphPlot(g, savefig=True, color="white")
gp_comp.plot_success_percentage_ver2(other_datacls=other_g)  # 成功率の比較棒グラフ
gp_comp.plot_box(other_datacls=other_g)                      # 個人合計貢献額の箱ひげ図
gp_comp.plot_box_round(other_datacls=other_g)                # ラウンド別箱ひげ図
gp_comp.plot_diff_targets_group(other_datacls=other_g)       # ターゲットとの差分
```

### ✅ `YahooTask` クラス

```python
from getdata import YahooTask

ytask = YahooTask(
    task_filename="3589457371.tsv",  # Yahoo!クラウドソーシングの成果報酬ファイル
    experiment_cls=g,                 # 連携させたい GetData インスタンス
)

ytask.create_whitelist()             # PayPayポイント別ホワイトリストを whitelists/ に出力
ytask.show_point_histgram()          # ポイント分布のヒストグラムを表示

whitelist_df = ytask.merge_result()  # Yahoo! ID と実験データを突合した DataFrame
```

キーワード（成功／未マッチなど）が異なる場合は、`keywords={"XXXXX": "Success", ...}` を渡すことでマッピングを上書きできます。

---

## 📌 典型的なワークフロー

1. `GetData` に oTree の CSV を読み込ませ、グループ構成・寄付行動・アンケートなどを確認する。
2. 必要に応じて `YahooTask` で成果報酬 TSV と突合し、PayPay ホワイトリストを出力する。
3. `GraphPlot` でグラフを作成し、`savefig=True` の場合は `YYYYMMDD_Figures/` 配下に保存する。

スクリプトとして直接実行するエントリーポイントは用意していないため、上記の手順をノートブックや任意の Python スクリプトから順に呼び出してください。

---
