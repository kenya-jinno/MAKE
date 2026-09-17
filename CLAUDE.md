# CLAUDE.md — AE/VAE 有効潜在次元 多指標分析フレームワーク（論文投稿版）

このファイルは **Claude Code** が `manifold/` ディレクトリで作業する際の
プロジェクト規約・論文執筆指針・実験実行手順を定める．

---

## ▶ 最優先タスク：学術論文の完成と投稿

### 投稿先（優先順位順）

| 優先度 | ジャーナル | 形式 | 言語 | ページ上限 |
|--------|-----------|------|------|-----------|
| 1 | **MDPI MAKE** (Machine Learning and Knowledge Extraction) | Article | 英語 | 制限なし |

### ファイル構成と作業フロー

```
日本語草稿（??の最も大きい数値のものが最新。ただしファイル作成タイムスタンプでも確認する）
main_paper_NN??.tex   →（platex + dvipdfmx）→  main_paper_NN??.pdf

→  MDPI 版（MAKE）：NN?? と同じ番号 ?? を用いる
main_paper_NN??.tex の内容を MDPI MAKE 投稿形式（Definitions/mdpi.cls）の英語に翻訳
main_paper_MAKE??.tex →（pdflatex ×3）→  main_paper_MAKE??.pdf
```

### 現在の主要ファイル

| ファイル | 役割 |
|---------|------|
| `main_paper_NN??.tex` | **現行日本語草稿**（バージョン??，??の数値が大きいものが最新版；現在は NN30） |
| `main_paper_NN??.pdf` | 上記のファイルのコンパイル済み PDF |
| `main_paper_MAKE??.tex` | **MDPI MAKE 英語版**（NN?? と番号を一致させる；現在は MAKE30） |
| `main_paper_MAKE??.pdf` | 上記のコンパイル済み PDF（pdflatex） |
| `Definitions/mdpi.cls` | MDPI 公式クラスファイル（MAKE 版で使用） |
| `main_paper.bib` | BibTeX 参考文献データベース（参照用；MAKE 版は本文内 thebibliography を使用） |
| `Plan.txt` | 次版改訂計画（査読対応指針） |
| `Review*.txt` | 各レビューラウンドのコメント |

> **【必須ルール1】改稿するたびに，NN 版と MAKE 版の番号を同一の新番号に揃えること．**
> 新番号は `NN` と `MAKE` の現行最新番号のうち **大きいほうに +1** した値とする．
> 例：NN42 / MAKE48 が現行最新 → 次版は `main_paper_NN49.tex` と `main_paper_MAKE49.tex`．
> **NN 側に欠番が生じてよい**（上記の例では NN43〜NN48 が欠番になる）．
> 英語版のみを修正した回があっても，次の改稿時に日本語版を新番号で起こして番号を揃え直す．
> その際は，日本語版に反映されていない過去の MAKE 版の修正（書誌情報の訂正など）を取り込むこと．
> 既存ファイルを絶対に上書き・削除しない（バージョン履歴が失われる）．
>
> **【必須ルール2】作業前に必ず両系列の現行最新番号を確認すること．**
> 確認コマンド：
> ```bash
> ls main_paper_NN*.tex | sort -V | tail -1
> ls main_paper_MAKE*.tex | sort -V | tail -1
> ```
> （MAKE1〜5 は旧・要約版で番号体系が異なるため，この規約の対象外．）

### NN 版 → MAKE 版 翻訳規約

- NN?? の内容・構成を **1:1 で忠実に完全英訳**する（要約・省略をしない）．
- `\label`・`\ref`・`\eqref`・`\cite` キー・数式・`\includegraphics` パス・数値は一切変更しない．
- 定理環境は mdpi.cls の大文字環境に変換：`theorem`→`Theorem`，`proposition`→`Proposition`，`definition`→`Definition`，`remark`→`Remark`，`corollary`→`Corollary`（`proof` はそのまま）．
- 参考文献は MDPI 書式（例：`LeCun, Y.; Bottou, L. ... \textit{Proc.\ IEEE} \textbf{1998}, \textit{86}, 2278--2324.`）の本文内 `thebibliography` とする．
- 付録見出しは `\section[\appendixname~\thesection]{Title}` 形式，`\appendixtitles{yes}` `\appendixstart` `\appendix` を後付（Author Contributions 等）の後に置く．
- 後付（`\authorcontributions` `\funding` `\dataavailability` `\conflictsofinterest` `\abbreviations`）は MDPI 必須項目．前版 MAKE の記載を引き継ぐ．
- 用語は既存 MAKE 版の対訳（effective latent dimensionality / AU slowdown point $\mknee$ / auxiliary reference estimator 等）を踏襲する．

---

## プロジェクト概要

**研究テーマ**: AE/VAE における有効潜在次元の解釈——内在次元推定・Active-Unit Dynamics・再構成指標に基づく多指標分析フレームワーク

**論文タイトル（日本語草稿）**:
AE/VAE における有効潜在次元の解釈：内在次元推定・Active-Unit Dynamics・再構成指標に基づく多指標分析フレームワーク

**中心的貢献**:
1. **有効次元の多指標分析フレームワーク**（主要貢献）：
   - AU 値（頑健指標），TwoNN/MLE ID（数値安定・補助的参照推定量），MSE エルボー，PCA の組み合わせ
   - 各指標の頑健性・探索性を明示した解釈論
2. **線形 β-VAE の AU 飽和定理**（理論的貢献）：
   - 次元 j が活性となる必要十分条件 λ_j > βτ² を厳密証明（定理1）
   - m_knee のグリッド感度上界（命題2）
3. **実データ幾何比較実験と TwoNN 条件付き安定性命題**：
   - MNIST・Fashion-MNIST で AE/IsometricAE/CAE/DAE を比較（Exp-Real-Geom）
   - 「m >> d̂_ID が TwoNN 安定性の支配的条件」を実証（命題3）
4. **m_knee の定式化・限界解明・安定化戦略**（補助的貢献）：
   - 密グリッド実験（Exp-Mnist-Dense）で m_knee = 10.3 ± 0.9（疎グリッドの 43±15 から改善）

---

## ディレクトリ構成

```
manifold/
├── CLAUDE.md                    # このファイル（Claude Code プロジェクト指示書）
├── SKILL.md                     # スキル定義（実験・論文執筆フロー）
├── main_paper_NN30.tex          # 現行日本語草稿（最新バージョン）★参照元
├── main_paper_MAKE30.tex        # MDPI MAKE 英語版（NN30 と番号一致）
├── Definitions/                 # MDPI 公式クラス（mdpi.cls, mdpi.bst, ロゴ）
├── main_paper.bib               # BibTeX 参考文献データベース
├── Plan.txt                     # 次版改訂計画（重要）
├── Review*.txt                  # レビューコメント
├── IEEE-Trans/                  # IEEE Transactions 形式のファイル（参考）
├── src/
│   ├── models/                  # AE/VAE/CAE/DAE/IsometricAE 実装
│   ├── data/                    # 合成多様体・MNIST ローダー
│   ├── metrics/                 # TwoNN, MLE, AU, CKA, Trustworthiness 等
│   └── experiments/             # 全実験スクリプト（E1〜EZ, Exp_*）
├── results/
│   ├── figures/                 # 生成図（PNG/PDF）
│   ├── tables/                  # 結果表（CSV/JSON）
│   └── *.pth                    # 学習済みモデル重み
└── notebooks/
    └── analysis.ipynb
```

---

## 実験一覧

### 合成多様体実験

| 実験 | スクリプト | 内容 |
|------|-----------|------|
| E1 | `E1_synthetic.py` | MSE 肘点・AU・TwoNN ID vs ボトルネック次元（Swiss Roll, Torus） |
| E2 | `E2_jacobian.py` | デコーダヤコビアン解析（有意特異値数，条件数 κ，TSA 誤差） |
| E3 | `E3_noise_direction.py` | NSR（ノイズ選択性比）分析 |
| EA | `EA_au_saturation.py` | AU 飽和（AE vs VAE）比較 |
| EB | `EB_topology.py` | 位相的最小埋め込み次元と AU 飽和の一致検証 |
| EC | `EC_isometric_ae.py` | IsometricAE による等長性改善 |
| ED | `ED_model_comparison.py` | 4モデル比較（AE/CAE/DAE/VAE） |

### MNIST・実データ実験

| 実験 | スクリプト | 内容 |
|------|-----------|------|
| E4 | `E4_bottleneck_sweep.py` | ボトルネック次元スイープ（m=1〜64） |
| E4_ext | `E4_extended.py` | 長期学習（300エポック，m=1〜256） |
| E4_ms | `E4_multiseed.py` | 多シード再現性検証 |
| E5 | `E5_structure_metrics.py` | 構造保存指標（Trustworthiness 等） |
| E6 | `E6_mnist.py` | MNIST 学習ダイナミクス（大域→局所の時間分離） |
| EP | `EP_twonn_stability.py` | TwoNN 安定性検証（実データ幾何比較） |
| EQ | `EQ_geometry_validity.py` | 幾何学的妥当性 |

### Conv-VAE・自然画像実験

| 実験 | スクリプト | 内容 |
|------|-----------|------|
| EG | `EG_conv_vae.py` | Conv-VAE 基本実験 |
| EH | `EH_conv_vae_300ep.py` | Conv-VAE 300エポック学習 |
| EI | `EI_cifar10_conv_vae.py` | CIFAR-10 Conv-VAE |
| EJ | `EJ_svhn_conv_vae.py` | SVHN Conv-VAE |
| EK | `EK_conv_vae_annealing_vs_fixed.py` | KL アニーリング vs 固定 β 比較 |
| EN | `EN_conv_vae_extended_m.py` | Conv-VAE 拡張 m グリッド（m=64〜256） |
| EN512 | `EN_extended_m512.py` | Conv-VAE m=512 拡張 |
| HiBeta | `Exp_ConvV_HighBeta.py` | 高 β 実験（β_max=10,20）AU 飽和確認 |
| NatImg | `Exp_NatImg_AU.py` | 自然画像での AU 飽和条件探索 |

### 特殊・補助実験

| 実験 | スクリプト | 内容 |
|------|-----------|------|
| EL | `EL_dense_grid_mknee.py` | 密グリッド m_knee 安定化（m=8〜16，1刻み） |
| EM | `EM_theory_bridge.py` | 線形理論と非線形実験の橋渡し（MNIST PCA） |
| EX_ms | `EX_mnist_multiseed.py` | MNIST 多シード実験 |
| EY | `EY_class_restricted.py` | クラス制限 MNIST 実験 |
| EZ | `EZ_tau_robustness.py` | τ_knee 閾値ロバスト性 |
| IsoID | `Exp_Iso_ID.py` | IsometricAE での ID 推定比較 |

---

## LaTeX コンパイル方法

```bash
# 日本語草稿（現行）
cd /home/kjinno/claude/manifold
platex main_paper_NN20.tex
bibtex main_paper_NN20
platex main_paper_NN20.tex
platex main_paper_NN20.tex
dvipdfmx main_paper_NN20.dvi

# 次バージョン作成時（例: NN30 → NN31）
cp main_paper_NN30.tex main_paper_NN31.tex
# 編集後にコンパイル（参考文献は本文内 thebibliography のため bibtex 不要）
platex main_paper_NN31.tex && platex main_paper_NN31.tex && dvipdfmx main_paper_NN31.dvi
```

### MDPI MAKE 英語版のコンパイル

```bash
# MAKE 版は mdpi.cls（Definitions/ 内）を使用し pdflatex でコンパイルする
cd /home/kjinno/claude/manifold
pdflatex main_paper_MAKE30.tex
pdflatex main_paper_MAKE30.tex
pdflatex main_paper_MAKE30.tex   # 相互参照解決のため計3回

# エラー確認
grep -n "^!" main_paper_MAKE30.log | head -20
```

---

## 論文構成（現行 main_paper_NN30.tex ＝ main_paper_MAKE30.tex）

```
Abstract
1. まえがき（Introduction）
   1.1 研究背景：有効潜在次元の解釈という問題
   1.2 本研究の位置づけと貢献
   1.3 「次元」の2種類：局所的固有次元と位相的埋め込み次元
   1.4 リサーチクエスチョン（RQ1〜RQ3）
   1.5 本論文の構成
2. 関連研究（Related Work）
   2.1 表現学習とオートエンコーダ / 2.2 深層表現の内在次元解析
   2.3 Active Units と Posterior Collapse / 2.4 幾何的オートエンコーダ
3. フレームワーク：有効潜在次元の読み取り
   3.1 問題設定と主要記号 / 3.2 読み取りパイプライン / 3.3 m_knee の解釈的意義
4. 理論的背景
   （定理1：線形 β-VAE の AU 飽和，命題2：m_knee グリッド感度，
    4.1 線形理論と非線形実験の対応，4.2 TwoNN 条件付き安定性）
5. 実験
   5.1 実験設定 / 5.2 合成多様体（RQ1-3） / 5.3 MNIST 主実験
   5.4 Fashion-MNIST / 5.5 Conv-AE 拡張 / 5.6 Conv-VAE 拡張 m グリッド
   5.7 高 β AU 飽和誘発 / 5.8 CIFAR-10・SVHN / 5.9 Exp-NatImg-AU
   5.10 実データ幾何比較（Exp-Real-Geom） / 5.11 自然画像拡張スイープと Stable Identification
6. 考察
   6.1 RQ1 への回答と m_knee / 6.2 m_knee 安定化戦略 / 6.3 TwoNN の安定性とバイアス
   6.4 AU Dynamics のメカニズム / 6.5 付随的知見 / 6.6 Conv-VAE AU 飽和遅延 / 6.7 限界
7. むすび（Conclusions）
参考文献（NN 版は本文末尾；MAKE 版は後付・付録の後）
付録 A〜F（証明スケッチ，定理1詳細導出，τ_knee 感度，β感度，TwoNN 詳細，Conv-VAE/自然画像詳細）
```

---

## 指標分類（論文中の核心）

| 指標 | 種別 | 頑健性 | 備考 |
|------|------|--------|------|
| AU 値 | 頑健指標 | σ ≤ 0.5（3シード） | 等長性バイアスなし；主要読み取り |
| TwoNN ID | 補助的参照推定量 | σ ≤ 0.12（数値的に安定） | 等長性バイアスあり；m >> d̂_ID の条件付き |
| MLE ID | 補助的参照推定量 | σ ≤ 0.15 | TwoNN と相補的 |
| MSE エルボー | 探索的指標 | グリッド依存 | 再構成側の参照 |
| m_knee（AU 鈍化点） | 探索的指標 | σ ≤ 0.9（密グリッド） | 疎グリッドでは σ ≈ 15 |

---

## 主要な実験数値（論文中の key numbers）

### 合成多様体（E1）
| データ | m | MSE | AU | TwoNN ID |
|--------|---|-----|----|----------|
| Swiss Roll | 1 | 2.971 | 1 | 1.00 |
| Swiss Roll | **2** | **0.032** | **2** | 1.35 |
| Swiss Roll | 3 | 0.0008 | 3 | 2.01 |
| Torus | **3** | **0.0005** | **3** | 1.99 |

### MNIST（E4_extended, 300 エポック）
- AE m=256: TwoNN ID = 10.03 ± 0.12，MLE = 11.85 → MNIST d_ID ≈ 10〜12
- VAE (β=4) m=12: AU = 8（飽和開始）
- m_knee（疎グリッド）：43 ± 15
- m_knee（密グリッド，Exp-Mnist-Dense）：**10.3 ± 0.9**（3シード）

### Conv-VAE（高 β 実験）
- β_max=10：m_knee ≈ 128（AU 飽和明確）
- β_max=20：m_knee ≈ 96（AU 飽和明確）
- β_max=4：m=512 まで AU 飽和プラトー未観測

### CIFAR-10（Exp-Nat-Cifar）
- 潜在 TwoNN ID (m=64) = 25.21 ± 0.41（Pope らの報告値 ≈ 26 と整合）

### 実データ幾何比較（Exp-Real-Geom，MNIST・Fashion-MNIST）
- 「m >> d̂_ID」が TwoNN 安定性の支配的条件（等長性改善より重要）

### IsometricAE（EC）
| モデル | MSE | 条件数 κ | NSR |
|--------|-----|---------|-----|
| AE | 0.0282 | 1.475 | 0.552 |
| CAE | 0.0296 | 1.531 | 0.059 |
| **IsometricAE** | **0.0290** | **1.040** | 0.753 |
| DAE | 0.0304 | 1.179 | 0.609 |

---

## 参考文献（main_paper.bib）

| キー | 著者 | 内容 | 出典 | 年 |
|------|------|------|------|-----|
| `kingma2014vae` | Kingma & Welling | Auto-Encoding Variational Bayes | ICLR | 2014 |
| `higgins2017betavae` | Higgins et al. | β-VAE | ICLR | 2017 |
| `lucas2019elbo` | Lucas et al. | Don't Blame the ELBO | NeurIPS | 2019 |
| `facco2017twonn` | Facco et al. | TwoNN 内在次元推定 | Sci. Rep. | 2017 |
| `levina2005mle` | Levina & Bickel | MLE 内在次元推定 | NIPS | 2005 |
| `kornblith2019cka` | Kornblith et al. | CKA | ICML | 2019 |
| `alain2014dae` | Alain & Bengio | DAE の理論 | JMLR | 2014 |
| `rifai2011cae` | Rifai et al. | 縮約オートエンコーダ | ICML | 2011 |
| `nazari2023geometric` | Nazari et al. | 幾何的 AE | ICML | 2023 |
| `fefferman2016manifold` | Fefferman et al. | 多様体仮説の検証 | JAMS | 2016 |
| `pope2021intrinsic` | Pope et al. | 画像の内在次元 | ICLR | 2021 |
| `ceruti2014danco` | Ceruti et al. | DANCo | Pattern Recogn. | 2014 |
| `burda2016iwae` | Burda et al. | 重み付き AE | ICLR | 2016 |
| `bengio2013representation` | Bengio et al. | 表現学習レビュー | IEEE TPAMI | 2013 |
| `tishby2000ib` | Tishby et al. | 情報ボトルネック | Allerton | 2000 |
| `vander2009visualization` | van der Maaten & Hinton | t-SNE | JMLR | 2008 |
| `tippingbishop1999ppca` | Tipping & Bishop | 確率的 PCA | JRSS-B | 1999 |
| `ansuini2019intrinsic` | Ansuini et al. | 深層ネットワークの ID | NeurIPS | 2019 |
| `valeriani2023geometry` | Valeriani et al. | 表現の幾何学 | NeurIPS | 2023 |
| `kato2020rdae` | Kato et al. | 等長 AE | ICML | 2020 |

---

## 記号表

| 記号 | 意味 |
|------|------|
| m, d_z | AE ボトルネック次元 |
| m* | 最適ボトルネック次元 |
| m_knee | AU 鈍化点（探索的指標） |
| d_ID, d̂_ID | 内在次元（推定値） |
| d_true | 真の多様体次元 |
| d_emb | 位相的最小埋め込み次元 |
| f: R^N → R^m | エンコーダ |
| g: R^m → R^N | デコーダ |
| J_g(z) | デコーダのヤコビ行列 |
| G(z) = J_g^T J_g | プルバック計量テンソル |
| κ | 条件数（等長性指標） |
| AU | Active Units（活性ユニット数） |
| AU_sat | AU 飽和値 |
| CKA | Centered Kernel Alignment |
| NSR | Noise Selectivity Ratio |
| TSA | Tangent Space Approximation Error |
| λ_j | データの j 番目の固有値 |
| τ² | デコーダノイズ分散 |
| β | VAE の KL 正則化係数 |

---

## 再現性規約

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

TWONN_K = 2
MLE_K = 10
AU_THRESHOLD = 0.01   # δ = 10^{-2}
CKA_SUBSAMPLE = 1000
```

---

## 次版改訂計画（Plan.txt より）

**最優先追加実験**:
1. 実データ IsometricAE で Stage 1 妥当性を直接比較（入力空間 / 参照 AE / IsometricAE の TwoNN 比較）
2. Conv-VAE / 自然画像での m > 64 拡張実験（m=64,96,128,192,256 でプラトー有無を確認）

**次点**:
3. τ_knee の適応化と二階差分法による knee 検出の改善

**記述上の修正**:
- TwoNN ID を「数値的に安定だが幾何学的にはバイアスあり」と安定性と正確性を分けて記述
- フレームワーク適用可能領域の3段階整理表（FC-VAE / Conv-VAE on MNIST / 自然画像）を追加
