# MDPI MAKE 投稿パッケージ（main_paper_MAKE42）

作成日: 2026-07-21
投稿先: MDPI Machine Learning and Knowledge Extraction (MAKE) / Article

## 同梱ファイル

| ファイル | 用途 |
|---|---|
| `main_paper_MAKE42.tex` | 投稿原稿の LaTeX ソース（下記の投稿用修正済み） |
| `main_paper_MAKE42.pdf` | コンパイル済み PDF（29 ページ, A4, pdflatex ×3, エラー・未解決参照なし） |
| `Definitions/` | MDPI 公式クラスファイル一式（mdpi.cls 2026-03-13 版, ロゴ含む） |
| `results/figures/` | 本文で使用する図 5 点（ベクタ PDF） |
| `main_paper_MAKE42_source.zip` | 上記 tex + Definitions + figures をまとめた投稿用 zip（SuSy にアップロード） |
| `supplementary_material.zip` | 査読用補足資料（下記参照） |
| `cover_letter.txt` | カバーレター（投稿システムの Cover Letter 欄に貼り付け） |

## 元の main_paper_MAKE42.tex からの変更点（親ディレクトリの原本は未変更）

MDPI の必須後付項目のうち欠けていた 3 点を追加した：

1. `\supplementary{...}` — 本文 5.1 節で言及している査読用補足資料
   4 部構成（実験スクリプト／解析スクリプト／学習ログ／環境情報）の記載
2. `\institutionalreview{Not applicable.}` — 研究倫理審査（ヒト・動物対象外）
3. `\informedconsent{Not applicable.}` — インフォームドコンセント（同上）

本文・数値・参考文献は一切変更していない。

## supplementary_material.zip の内容（本文 5.1 節の 4 部構成に対応）

- `scripts/experiments/` — (i) 実験スクリプト（Table tab:exp_list と 1 対 1 対応）
- `scripts/models|data|metrics/` — (ii) モデル・ローダー・解析コード（全表・全図を再生成）
- `logs/` — (iii) 学習ログ・数値結果（JSON/CSV, 全シード, 47 ファイル）
- `ENVIRONMENT.txt` — (iv) 環境情報（Python 3.12.3, PyTorch 2.7.1+cu118, シード規約ほか）

## 投稿時の手順（susy.mdpi.com）

1. Journal: MAKE, Article Type: Article を選択
2. Manuscript file: `main_paper_MAKE42_source.zip`（または tex+PDF 一式）をアップロード
3. Supplementary: `supplementary_material.zip` をアップロード
4. Cover Letter 欄に `cover_letter.txt` の内容を貼り付け
5. 著者情報入力時の注意:
   - **ORCID**: tex 内に埋め込み済み（第1ページの著者名横にアイコン表示）。
     投稿システム側でも同じ ORCID を入力すること：
     - Chika Obata: 0009-0001-1209-0918
     - Mizuki Dai: 0000-0002-4070-4585
     - Kenya Jin'no: 0000-0002-0431-5769
   - 対応著者: Kenya Jin'no (kjinno@tcu.ac.jp)
6. 査読者候補（3 名以上）の入力を求められるので事前に準備しておくこと

## 再コンパイル方法

```bash
cd Submit
pdflatex main_paper_MAKE42.tex   # ×3 回（相互参照解決）
```
