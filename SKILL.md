# SKILL.md — 実験・論文執筆スキル定義

このファイルは Claude Code が `manifold/` ディレクトリで行う
**実験実行・論文改稿・投稿準備**のための手順書である．
作業前に必ず `CLAUDE.md` を全体精読すること．

---

## スキル1：論文改稿（日本語草稿）

> **【必須ルール】新たな原稿を作成するたびに，texファイル名末尾の数字を必ず +1 すること．**
> 例：`main_paper_NN30.tex` → `main_paper_NN31.tex`
> 既存ファイルを絶対に上書き・削除しない．作業前に現行最新番号を必ず確認すること．

**目的**: `main_paper_NN??.tex`（現行：NN30）を改稿し次バージョンを作成する．

### 手順

#### Step 1：状態確認

```bash
# 現行最新バージョンを確認（タイムスタンプでも二重確認する）
ls main_paper_NN*.tex | sort -V | tail -3
ls -lt main_paper_NN*.tex | head -3

# Plan.txt で改訂指針を確認
cat Plan.txt

# 最新 Review*.txt を確認
ls -lt Review*.txt | head -3
```

#### Step 2：新バージョンファイルを作成

```bash
# 例: NN30 → NN31
cp main_paper_NN30.tex main_paper_NN31.tex
# 編集後にコンパイル（参考文献は本文内 thebibliography のため bibtex 不要）
platex main_paper_NN31.tex && platex main_paper_NN31.tex
dvipdfmx main_paper_NN31.dvi
```

**絶対に既存の NN??.tex を上書きしないこと**（バージョン履歴が失われる）．

#### Step 3：改稿チェックリスト

```
□ Plan.txt の最優先事項を反映したか
□ 各指標の分類（頑健/補助的参照/探索的）が一貫しているか
□ AU 飽和定理（定理1）と実験の対応が本文で明示されているか
□ m_knee の限界（グリッド感度）が適切に記述されているか
□ Conv-VAE/自然画像の未解決事項が正直に述べられているか
□ 全数値が results/tables/ の JSON/CSV と一致しているか
□ 全図が results/figures/ から参照されているか
□ 参考文献（本文内 thebibliography）に \cite キーが全て存在するか
```

#### Step 4：コンパイルエラーの対処

```bash
# LaTeX ログを確認（NN31 は例；実際の版番号に読み替える）
grep -n "Error\|Warning" main_paper_NN31.log | head -20

# dvipdfmx で図が見つからない場合
ls results/figures/   # 参照先の存在確認
```

---

## スキル2：実験実行

**目的**: `src/experiments/` のスクリプトを実行し，結果を `results/` に保存する．

### 基本実行パターン

```bash
# GPU が利用可能な場合は自動で使用
python3 src/experiments/<スクリプト名>.py

# 特定の引数が必要な場合（スクリプト内の argparse を確認）
python3 src/experiments/EL_dense_grid_mknee.py --grid_step 1 --m_min 8 --m_max 16
```

### 優先実験（Plan.txt より）

#### 優先度1：Stage 1 妥当性比較実験

```bash
# IsometricAE での ID 推定比較（実装済みスクリプト）
python3 src/experiments/Exp_Iso_ID.py
# 結果: results/tables/IsoID_comparison.json
#        results/figures/IsoID_twonn_comparison.pdf

# 実データ幾何比較
python3 src/experiments/EP_twonn_stability.py
```

**期待される結果**: 入力空間 / 参照 AE / IsometricAE の TwoNN ID を並べて，
κ 改善に伴う ID 推定の変化を定量化する．

#### 優先度2：Conv-VAE 拡張 m グリッド

```bash
# m=64〜256 拡張（既存スクリプト）
python3 src/experiments/EN_conv_vae_extended_m.py

# m=512 拡張
python3 src/experiments/EN_extended_m512.py

# 高 β 実験
python3 src/experiments/Exp_ConvV_HighBeta.py
# 結果: results/figures/ConvV_HiBeta_au_sweep.pdf
```

#### 優先度3：m_knee 安定化（密グリッド実験）

```bash
python3 src/experiments/EL_dense_grid_mknee.py
# 結果: results/tables/EL_dense_grid.json
#        results/figures/EL_mknee_stability.pdf
```

### 結果確認

```bash
# 最新の JSON 結果を確認
ls -lt results/tables/*.json | head -10
python3 -c "import json; d=json.load(open('results/tables/EL_dense_grid.json')); print(d)"

# 最新の図を確認
ls -lt results/figures/*.pdf | head -10
```

---

## スキル3：MDPI MAKE 英語版作成（main_paper_NN?? → main_paper_MAKE??）

**目的**: 最新の日本語草稿 `main_paper_NN??.tex` を MDPI MAKE 投稿形式の英語版
`main_paper_MAKE??.tex` に変換し，`main_paper_MAKE??.pdf` を生成する．

> **【必須ルール】MAKE 版の番号 ?? は翻訳元の NN 版の番号と必ず一致させること．**
> 例：`main_paper_NN30.tex` → `main_paper_MAKE30.tex`
> （MAKE1〜MAKE5 は旧・要約版であり番号体系が異なる．参照用としてのみ使用する．）
> 既存の MAKE??.tex / MAKE??.pdf を絶対に上書き・削除しない．

### Step 1：状態確認

```bash
# 翻訳元となる最新 NN 版と，既存 MAKE 版の最新番号を確認
ls main_paper_NN*.tex | sort -V | tail -1
ls main_paper_MAKE*.tex | sort -V | tail -1

# MDPI クラスファイルの存在確認（Definitions/ に同梱済み）
ls Definitions/mdpi.cls
```

### Step 2：翻訳規約（NN 版と 1:1 対応の完全英訳）

- NN?? の内容・章構成を**忠実に完全英訳**する（要約・省略をしない）．
- 以下は一切変更しない：`\label` / `\ref` / `\eqref` / `\cite` キー・数式・
  `\includegraphics` パス・表中の全数値．
- 用語は直近の MAKE 版の対訳を踏襲する（effective latent dimensionality，
  AU slowdown point $\mknee$，auxiliary reference estimator，exploratory indicator 等）．
- 定理環境は mdpi.cls の大文字環境へ：`theorem`→`Theorem`，`proposition`→`Proposition`，
  `definition`→`Definition`，`remark`→`Remark`，`corollary`→`Corollary`（`proof` はそのまま）．
- 日本語句読点（，．・「」）を英語句読点に置換し，日本語文字を残さない
  （`grep -P '[\x{3040}-\x{9FFF}]' main_paper_MAKE??.tex` で検査）．

**ドキュメントクラスと前置き**（前版 MAKE から引き継ぐ）:
```latex
\documentclass[make,article,submit,pdftex,moreauthors]{Definitions/mdpi}
% \Title / \Author / \AuthorNames / \address / \corres / \abstract / \keyword
% 記号マクロ（\dID, \dz, \dzopt, \dtrue, \Jg, \R, \kap, \AUsat, \dzsat, \mknee, \demb）
```

**本文の後に置く MDPI 必須後付**（前版 MAKE から引き継ぐ）:
```latex
\authorcontributions{...} \funding{...} \dataavailability{...}
\conflictsofinterest{...} \abbreviations{Abbreviations}{...}
\appendixtitles{yes} \appendixstart \appendix
% 付録見出しは \section[\appendixname~\thesection]{Title} 形式
% 参考文献（本文内 thebibliography，MDPI 書式）は付録の後に置く
```

**参考文献は MDPI 書式**（bibtex ではなく本文内 thebibliography）:
```latex
\bibitem{lecun1998}
LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P.
Gradient-based learning applied to document recognition.
\textit{Proc.\ IEEE} \textbf{1998}, \textit{86}, 2278--2324.
```

### Step 3：コンパイル（pdflatex ×3）

```bash
pdflatex main_paper_MAKE30.tex
pdflatex main_paper_MAKE30.tex
pdflatex main_paper_MAKE30.tex   # 相互参照解決のため計3回
```

### Step 4：品質検査

```bash
# エラー・未解決参照ゼロを確認
grep -n "^!" main_paper_MAKE30.log
grep -c "Warning.*undefined" main_paper_MAKE30.log

# 幅超過（Overfull）を確認；30pt 超は要修正
grep "Overfull \\hbox" main_paper_MAKE30.log | sort -t'(' -k2 -rn | head

# NN 版とのラベル・引用整合性
comm -23 <(grep -o '\\label{[^}]*}' main_paper_NN30.tex | sort -u) \
         <(grep -o '\\label{[^}]*}' main_paper_MAKE30.tex | sort -u)
```

**幅の広い表の修正法**（MDPI 公式の全幅表示）:
```latex
\begin{table}[H]
\begin{adjustwidth}{-\extralength}{0cm}   % 左余白側へ 4.61cm 拡張
\centering
\caption{...} \label{...} \small
\begin{tabular}{...} ... \end{tabular}
\end{adjustwidth}
\end{table}
```

### MAKE 版チェックリスト

```
□ NN?? と MAKE?? の番号が一致しているか
□ 全セクション・全表・全図・全付録が NN 版と 1:1 で対応しているか
□ 日本語文字が残っていないか（grep 検査）
□ \label / \ref / \cite / 数値が NN 版と一致しているか
□ Author Contributions / Funding / Data Availability / Conflicts of Interest
□ Abbreviations 表
□ 参考文献が MDPI 書式（Surname, I.; ... \textbf{年}, 巻, 頁）か
□ pdflatex エラー 0・未解決参照 0・Overfull 30pt 超 0
```

---

## スキル4：図の生成・更新

**目的**: 論文用の高品質な図を生成・更新する．

### 既存図の確認

```bash
ls results/figures/*.pdf | sort
# 必要な図がない場合は以下の再生成スクリプトを使用
python3 regenerate_figures.py
```

### 新規図の生成

新しい実験結果の図が必要な場合は，対応する実験スクリプトを実行する：

```bash
# 例: IsometricAE 比較図
python3 src/experiments/Exp_Iso_ID.py
# 結果が results/figures/ に自動保存される

# 例: フレームワーク適用可能性の3段階表（図として生成）
python3 src/utils/generate_applicability_table.py
```

**図の品質基準**:
- フォーマット: PDF（論文用）+ PNG（確認用）
- DPI: ≥ 300（PNG）
- フォントサイズ: 本文と同等（≥ 10pt）
- カラーマップ: 白黒印刷でも判別可能なものを選ぶ

---

## スキル5：査読コメント対応

**目的**: Review*.txt のコメントに対して論文を修正する．

### 手順

#### Step 1：コメント整理

```bash
# 最新のレビューコメントを確認
cat Review15.txt   # 最新版（数字が大きいほど新しい）

# Plan.txt で対応方針を確認
cat Plan.txt
```

#### Step 2：コメントへの回答文書作成

```bash
# 回答文書（Response Letter）を作成
touch response_letter_rN.tex   # N = レビューラウンド番号
```

**回答文書の構成**:
```
1. Thank the reviewers
2. List of major changes
3. Detailed responses to each comment:
   - Reviewer 1, Comment 1: [quote] → [response] → [action taken]
   - ...
4. Summary of all changes
```

#### Step 3：対応内容の確認チェックリスト

```
□ 全コメントに対して response があるか
□ "We agree" / "We respectfully disagree" を明示しているか
□ 変更箇所を論文の該当行番号で示しているか
□ 追加実験が必要なコメントに対して実験を実施したか
□ 論文中の変更箇所がハイライト（\textcolor{blue}{...}）されているか
```

---

## スキル6：投稿準備（MDPI MAKE）

**目的**: MDPI MAKE への投稿ファイルを準備する．

### MDPI MAKE 投稿準備

```bash
# 最新の MAKE 版番号を確認（例：MAKE30）
ls main_paper_MAKE*.tex | sort -V | tail -1

# 投稿用ディレクトリを作成（クラスファイルと図も同梱する）
mkdir -p submission_MAKE/
cp main_paper_MAKE30.tex submission_MAKE/manuscript.tex
cp -r Definitions submission_MAKE/
mkdir -p submission_MAKE/results
cp -r results/figures submission_MAKE/results/

# 投稿用にコンパイル確認（MDPI は PDF 一本での初回投稿も可）
cd submission_MAKE && pdflatex manuscript.tex && pdflatex manuscript.tex && pdflatex manuscript.tex
```

**投稿必須チェックリスト（MDPI MAKE）**:
```
□ mdpi.cls を使用しているか
□ Author Contributions セクション
□ Funding セクション
□ Institutional Review Board Statement（該当なしでも記載）
□ Informed Consent Statement（該当なしでも記載）
□ Data Availability Statement
□ Acknowledgments
□ Conflicts of Interest
□ MDPI の Open Access（CC BY ライセンス）への同意
```

---

## よくあるトラブルと対処

### LaTeX コンパイルエラー

| エラー | 原因 | 対処 |
|--------|------|------|
| `File 'xxx.sty' not found` | パッケージ未インストール | `sudo apt install texlive-xxx` または `\usepackage` を削除 |
| `Undefined control sequence` | コマンド名ミス | `\newcommand` 定義を確認 |
| `Missing $ inserted` | 数式モードの漏れ | エラー行周辺の数式を確認 |
| `Overfull \hbox` | テキストが余白を超過 | `\allowbreak` または表現を短縮 |
| BibTeX が空 | `.bib` のキー名ミス | `\cite{}` のキーと `.bib` のエントリを照合 |

### 実験スクリプトのエラー

```bash
# CUDA メモリ不足
# → バッチサイズを半分にするか CPU に切り替え
CUDA_VISIBLE_DEVICES="" python3 src/experiments/xxx.py

# モジュール not found
pip install torch torchvision numpy matplotlib scikit-learn

# 結果ディレクトリが存在しない
mkdir -p results/figures results/tables
```

### 図ファイルが見つからない

```bash
# results/figures/ を確認
ls results/figures/ | grep "E1"

# 見つからない場合は実験を再実行
python3 src/experiments/E1_synthetic.py
```
