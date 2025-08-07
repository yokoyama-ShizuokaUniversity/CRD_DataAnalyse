# CRD Data Analysis

このリポジトリは、Collective Risk Dilemma（CRD）実験のデータ分析および可視化を行うためのPythonプログラムです。
主にoTreeで収集したデータ（`all_apps_wide`形式のCSV）をもとに、グループごとの貢献行動や成功率などを解析・可視化します。

---

## 📦 構成

* `GetData` クラス：CSVデータの読み込みと整形、グループ・貢献データの抽出など
* `GraphPlot` クラス：貢献額・成功率などのグラフを描画し、PNGとして保存（または表示）

---

## 📁 前提ディレクトリ構成

```
プロジェクト/
│
├─ your_data.csv  ← all_apps_wide形式のデータ
├─ main.py        ← このプログラム本体
└─ Figures/       ← グラフ出力ディレクトリ（自動生成されます）
```

---

## 🔧 必要なライブラリ

* Python 3.9 以上推奨
* 必須ライブラリ：

  * pandas
  * matplotlib

```bash
pip install pandas matplotlib
```

---

## 🧐 クラス構成と使い方

### ✅ `GetData` クラス

```python
g = GetData(
    dirpath="path/to/data",
    filename="your_data.csv",
    debug=True  # オプション：処理過程を表示
)
print(g)  # 実験の集計サマリを表示
```

主な機能：

* グループ分割と失敗者の理由抽出
* 成功者の寄付データの抽出
* 途中離脱の特定
* ラウンドごとの貢献額の集計

---

### ✅ `GraphPlot` クラス

```python
gp = GraphPlot(g, savefig=True)
gp.plot_total_contribution()  # 累積貢献額
gp.plot_total_contribution(pledges=True)  # 提案額付き
gp.plot_contribution_per_round()  # 貭aラウンドの貢献推移
gp.plot_success_group_percentage()  # 成功率の棒グラフ
```

出力は `Figures/` ディレクトリに自動保存されます（`savefig=False`なら表示のみ）

---

## 📌 実行例

```bash
python main.py
```

出力例：

```
4人ちゃんと集まったグループ数：5
{1: [101, 102, 103, 104], ...}

4人そろわなかったグループ数：2
{6: [201, 202], ...}

正規に終了しなかった参加者数：3
4人そろわなかったグループの参加者の失敗理由：
201: not enough players
202: browser closed

途中離脱(ドロップアウト)した人：
301: Round 3 で離脱
305: Round 6 で離脱
```

---