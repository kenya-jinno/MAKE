# MDPI MAKE 投稿用ファイル一式（MAKE47）

**論文**: Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and
Variational Autoencoders with Active-Unit Diagnostics
**投稿先**: MDPI *Machine Learning and Knowledge Extraction* (MAKE), Article

## フォルダ内容と投稿システム（SuSy）での用途

| ファイル | 用途 |
|---|---|
| `main_paper_MAKE47.pdf` | **査読用原稿本体**（33ページ、行番号付き submit モード）。SuSy の Manuscript としてアップロード |
| `latex_source_MAKE47.zip` | **LaTeX ソース一式**（tex + `Definitions/`(mdpi.cls 等) + `results/figures/` の全参照図）。LaTeX 投稿時のソース提出用。単体で pdflatex ×3 のコンパイル可を検証済み |
| `supplementary_materials.zip` | **匿名査読用補足資料**（131ファイル：全実験コード、解析スクリプト、設定 JSON、seed 別学習ログ、図表再生成スクリプト、環境情報）。SuSy の Supplementary Materials としてアップロード |
| `main_paper_MAKE47.tex` / `Definitions/` / `results/figures/` | 上記 zip の展開済み実体（参照用） |

## 投稿前の最終確認（済）

- pdflatex ×3 でエラー 0・未定義参照 0（このフォルダ単体で検証済み）
- Abstract 約200語、キーワード10個（規定 3–10）
- References 見出し（`\reftitle`）出力確認済み
- 引用37件 = bibitem 37件、本文初出順に整列
- Back matter: Supplementary Materials / Author Contributions (CRediT) /
  Funding (MDPI 推奨形式) / Institutional Review Board (Not applicable) /
  Informed Consent (Not applicable) / Data Availability /
  Acknowledgments (生成AI利用開示) / Conflicts of Interest / Abbreviations
- 補足資料の匿名性確認済み（著者名・所属・ローカルパスなし）

## 投稿前に著者が行う残作業

1. `supplementary_materials.zip` 内 `ENVIRONMENT.md` の TODO:
   学習マシン（RTX 3070）で `pip freeze > requirements_freeze.txt` を取得し
   zip に追加（skdim / PyTorch / CUDA バージョン含む）
2. 全表の数値とスクリプト出力の一致の最終確認
   （`code/analysis/` のスクリプトで実行可能；スポットチェックでは一致済み）
3. SuSy 上で著者情報・ORCID（C.O.: 0009-0001-1209-0918,
   M.D.: 0000-0002-4070-4585, K.J.: 0000-0002-0431-5769）を入力
4. 受理後: 補足資料を公開 GitHub + Zenodo (DOI) に置き、
   `\supplementary` と `\dataavailability` の URL/DOI を差し替え
