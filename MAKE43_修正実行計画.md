# MAKE43 修正実行計画（main_paper_MAKE42 → main_paper_MAKE43）

> **ステータス：完了（2026-08-29 実施）**
> - `main_paper_MAKE43.tex` / `main_paper_MAKE43.pdf`（31ページ）を作成．pdflatex ×3 でエラー 0・未定義参照 0．
> - タイトルは指摘md §2.1 第2案（AU 併記版）を採用：
>   *Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics*
> - 図：`E4_mnist_dualaxis_v43.{pdf,png}`（2×2 パネル分離），`E4_300ep_multiseed_v43.{pdf,png}`（ラベル修正）を
>   `replot_E4_dualaxis_v43.py` / `replot_E4_multiseed_v43.py`（新規，キャッシュ JSON から再描画）で生成し差し替え．
>   Figure fig:ep は tex キャプションのみ修正．旧図・旧スクリプトはすべて残存．
> - 新設表：Table `tab:cost`（学習 run 数の条件別削減率）．
> - 追加文献：`camastra2016intrinsic`（Inf. Sci. 2016, 328, 26–41），`allegra2020data`（Sci. Rep. 2020, 10, 16449）
>   — 引用順と thebibliography 順の一致を検証済み（cited⇔bibitem 完全一致）．
> - Prop. 3 は Proposition 環境を廃止し §4.3「Conditional-Stability Rationale」へ改称，全参照を Section 参照に更新．
> - MAKE42 の tex/pdf は無変更のまま残存．下記チェックリストは全項目実施済み（[x]）．
> - 再実験は不要と判断（§5 のとおり）；オプション実験（複数データ分割 seed 等）は次版へ持ち越し，
>   対応する限定文（T6-3, T10）を本文に記載済み．

本計画書は，`MAKE投稿論文_修正指摘と改善案.md`（以下「指摘md」）の全指摘を
`main_paper_MAKE42.tex` に反映し，`main_paper_MAKE43.tex` / `main_paper_MAKE43.pdf`
を完成させるための実行手順書である．実際の修正作業はこの計画書に従って行う．

> **統一方針（指摘md §1, §16）**：論文全体を「ボトルネック次元を一般的に自動決定する方法」
> ではなく，**「\(\hat d_{\mathrm{ID}}\) を探索窓の参照値とし，AU と再構成誤差で適用可否を
> 診断する条件付き estimate–set–verify プロトコル」**として一貫して記述する．

---

## 0. バージョン規約（必須遵守）

- `cp main_paper_MAKE42.tex main_paper_MAKE43.tex` で複製してから編集する．
- **既存の `main_paper_MAKE42.tex` / `.pdf` を絶対に上書き・削除しない．**
- 番号は必ず「基のファイル +1」（今回 42 → 43）．以降の改稿も同様（43 → 44, …）．
- 作業前確認：`ls main_paper_MAKE*.tex | sort -V | tail -1`
- 修正する図は**旧ファイルを残し**，新ファイル名（`_v43` サフィックス）で
  `results/figures/` に保存し，tex 側の `\includegraphics` パスを差し替える．

## 1. 作業フロー概要

```
Phase A: 図表の再生成（§4）           ← 再学習不要（results/tables/ の JSON から再描画）
Phase B: tex 本文修正（§3 のタスク T1〜T13）
Phase C: コンパイルと検証（§6）
    cd /Users/jinno/Dropbox/Work/claude/manifold/117/manifold
    pdflatex main_paper_MAKE43.tex （×3回，相互参照解決）
    grep -n "^!" main_paper_MAKE43.log ; grep -n "Warning.*undefined" main_paper_MAKE43.log
Phase D: チェックリスト照合（§7）
```

---

## 2. MAKE42 内の主要アンカー（編集箇所の目印）

| 対象 | MAKE42 の位置 |
|---|---|
| タイトル | L90 `\Title{Choosing the Bottleneck Dimension ...}` |
| Abstract | L115 `\abstract{` |
| §1.2 貢献 | L190 / §1.3 次元の2種類 L295 / §1.4 限界 L310 |
| §2 関連研究 | L343 |
| §3 プロトコル（V1/V2 含む） | L411–706（V1/V2 は `\subsection{The Protocol}` L466 内） |
| §4 理論（Prop.1/2/3） | L707–855；`thm:lower_bound` L720，`thm:linear_vae` L753，`thm:twonn_stability` L835 |
| §5.1 実験設定 | L878 / §5.2 合成 L918 / §5.3 MNIST L1056 / §5.4 downstream L1331 / §5.5 real-geom L1413 / §5.6 boundary L1472 |
| §6 考察 | L1635 / §7 結論 L1761 |
| 付録 A 証明 | L1867（A.1 導出 L1869，A.2 TwoNN L1946） |
| 表 | `tab:notation` L437，`tab:ea` L972（=旧 Table 5 相当），`tab:mnist_s1` L1089，`tab:boot_ci` L1125，`tab:mnist_s3` L1206，`tab:fashion_s3` L1314，`tab:downstream` L1391，`tab:real_geom` L1431，`tab:natural` L1615 |
| 図 | `fig:protocol` L507，`fig:e4_dualaxis` L1233，`fig:e4_multiseed` L1246，`fig:ep` L1469，`fig:en` L1531，`fig:hibeta` L2120 |

> 注意：指摘md 中の「Table 5 / 6 / 7 / 8 / 10 / 12」「Figure 2 / 3 / 4」は MAKE42 の PDF 上の
> 通し番号．編集時は必ず `main_paper_MAKE42.pdf` で番号→ラベルの対応を再確認すること
> （暫定対応：T5→`tab:ea`，T6→`tab:mnist_s1`，T7→`tab:boot_ci`，T8→`tab:mnist_s3`，
> T10→`tab:downstream`，T12→`tab:natural`；F2→`fig:e4_dualaxis`，F3→`fig:e4_multiseed`，
> F4→`fig:ep`）．

---

## 3. 修正タスク一覧（tex 本文）

各タスクの**具体的な英文差し替え文案は指摘md の該当節に記載済み**．原則としてその文案を
そのまま（または文脈に合わせ最小修正で）採用する．

### T1. タイトル変更【指摘md §2.1】
- L90 の `\Title{}` を次に変更：
  `An Intrinsic-Dimension-Guided Protocol for Selecting Autoencoder Bottleneck Dimensions`
  （AU を残す場合は第2案 `Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics`）
- ランニングヘッド・`\TitleCitation` 等の付随箇所も同期して変更する．

### T2. Abstract 全面差し替え【指摘md §2.2, §13】
- L115 の `\abstract{}` を指摘md §13 の要旨案で置き換える（分量は MDPI 規定
  200 語程度を目安に必要なら圧縮；主張・限定は削らない）．
- 必須要素：(a) conditional, diagnostic protocol であること，(b) VAE の
  \(m\ge2\hat d_{\mathrm{ID}}\) は保証・最適化原理ではなく「十分に大きな候補＋学習後検証」
  であること，(c) 削減率は「本研究の特定グリッドで約 60–80%」と限定，
  (d) AU は ID/埋め込み次元の一般的推定量ではないこと．
- `\keyword{}` も新タイトルと整合させる（例：intrinsic dimension; bottleneck selection;
  active units; diagnostic protocol; ...）．

### T3. 中心主張の統一【指摘md §2.3】
- Introduction（§1.1–1.2）と Conclusions（§7）に，キーフレーズ
  `conditional diagnostic protocol` / `search-window reference` /
  `model- and training-dependent diagnostic` /
  `not a general estimator of intrinsic or embedding dimension` を一貫使用．
- Conclusions は指摘md §14 の結論文案をベースに書き換える．

### T4. 理論的主張の限定【指摘md §3】
1. **Prop.1（`thm:lower_bound`）**：「self-intersection-free reconstruction」を
   `providing a necessary condition for zero-error, self-intersection-free reconstruction under an idealized continuous-manifold assumption` に変更（Abstract・§1.2・§4.1 の同表現も）．
   トーラス実験は「命題の直接的実証」ではなく「理想化された下限が経験的に現れる事例」と記述．
2. **\(d_{\mathrm{emb}}\ge d_{\mathrm{ID}}\) の扱い（§1.3, §3, §4.1, 結論）**：指摘md §3.1 の
   文案（理想多様体では \(d_{\mathrm{emb}}\ge d\)；実データの \(\hat d_{\mathrm{ID}}\) は
   \(d_{\mathrm{emb}}\) の直接推定量ではなく，マージンの幾何学的動機として解釈）を挿入．
3. **Prop.2（`thm:linear_vae`）**：命題直後に指摘md §3.3 の限定文
   （\(\lambda_j>\beta\tau^2\) は線形ガウス代理モデルでのみ厳密；非線形 VAE の
   座標別活性化定理・AU の定量予測式ではない）を追加．§1.2 の貢献では
   `theoretical motivation` / `interpretive surrogate` と位置づける（§12.2 と連動）．
4. **付録 A.1 の ELBO–相互情報量置換**：指摘md §3.4 の文
   （aggregated-posterior mismatch はここで考える大域最適線形ガウス解に限る；
   任意の線形/非線形 VAE に一般化しない）を追加．
5. **Prop.3（`thm:twonn_stability`）**：`Proposition` 環境をやめ，見出しを
   `4.3 Conditional-Stability Rationale for Latent-Space TwoNN Estimation` とし，
   本文冒頭に `This section provides a heuristic conditional-stability rationale rather than a finite-sample theorem for TwoNN.` を追加．付録 A.2 の「Proof Sketch」も
   「Heuristic argument」等に改称．本文中の `Proposition~\ref{thm:twonn_stability}` 参照を
   全て修正（`grep -n "twonn_stability" main_paper_MAKE43.tex` で漏れ確認）．

### T5. AU の位置づけの厳密化【指摘md §4】
1. `tab:notation`（L437）の AUsat 説明を
   `Saturated number of empirically active coordinates under a specified β, threshold δ, and architecture` に変更．
2. §3.1 の AU 定義部に座標依存性の文（指摘md §4.2：回転・再パラメータ化に非不変；
   事前分布・パラメータ化・学習目的・アーキテクチャに依存）を追加．Limitations（§6.7 相当）
   にも一行追加．
3. トーラスの \(AU_{\mathrm{sat}}=d_{\mathrm{emb}}\) 一致（§5.2, 付録 C.4, 結論）を
   指摘md §4.3 の文案で「経験的観測であり一般的推定性質ではない」と限定．

### T6. Step 1 / Step 3 の分離と安定性主張の限定【指摘md §5】
1. §3.2.1 または §5.3.1 に，参照表現は候補 VAE と独立に学習し，§5.3 の VAE 潜在 TwoNN は
   診断的クロスチェックであって \(\hat d_{\mathrm{ID}}\) の定義に用いない旨を追加（§5.1 文案）．
2. キャプション修正：`tab:mnist_s1` に「reference AE latents used for Step 1（単一seed sweep）」，
   `tab:mnist_s3` に「VAE latents; diagnostic only; not used to set \(\hat d_{\mathrm{ID}}\)」．
3. §5.3.1 に単一seed sweep と 3seed 検証の区別（指摘md §5.2 文案），Limitations にも簡潔に記載．
4. `tab:boot_ci` 付近にサブサンプリング95%区間の意味の限定（指摘md §5.3 文案）を追加．
5. 「stable」の一般化を排除：Abstract・§5.5・§6.2・結論の該当表現を指摘md §5.4 の
   `Under the datasets, architectures, sample sizes, and estimators tested here, ...` 型に変更．

### T7. V1/V2 の位置づけ【指摘md §6】
- §3.2.3（V1/V2 定義部）に：
  `V1 and V2 are operational screening criteria introduced for this study. They are not hypothesis tests, confidence rules, or universally calibrated acceptance criteria.`
- pass が意味しないこと（余剰座標の完全消失・AU と次元の一致・再構成/生成/下流性能の保証
  ではない；MSE・ELBO・下流・可視化と併用する診断）を指摘md §6.2 の文案で明記．

### T8. MNIST 性能・探索コスト主張の限定【指摘md §7】
1. `grid-best` → `best observed`（Abstract, §5.4, 結論）：
  `On the MNIST grid and linear-probe evaluation used in this study, the ID-guided window contained the best observed probe accuracy.`
2. VAE の「nearly identical performance」を定量化・限定（指摘md §7.2：0.9 ポイント差，
   3seed では同等性の正式な主張は不可）．
3. **新規表の追加**（§5.4）：計算削減率の条件別内訳表（指摘md §7.3 の 5 行表を採用；
   キャプションは `Training-run accounting for the specific MNIST grid, epoch count, seed design, and verification procedure used in this study.`）．削減率が「学習 run 数」の比較であり
   GPU 時間等を含まないことを本文に明記．
4. `tab:downstream` キャプションに AE=単一seed / VAE=3seed の非対称性を明記
   （指摘md §7.4 の文案）．

### T9. Conv-VAE・自然画像【指摘md §8】
1. §5.6・付録 C.1 に固定 β 理論と周期的 KL アニーリングの区別
   （指摘md §8.1 文案：\(\beta_{\max}\) は学習スケジュールのパラメータであり Prop.2 の
   固定 β と同一視しない）．アニーリングスケジュール（周期・warm-up・最小 β・周期数・
   関数形）を付録の表または既存図 `fig:hibeta` 周辺に明記．
2. §5.6.2 の3機構（manifold entanglement / high-frequency texture / hierarchical latent）
   冒頭に `The following mechanisms are hypotheses consistent with the observed pattern, not experimentally distinguished explanations.` を追加．manifold entanglement は
   指摘md §8.2 の3候補のいずれかで操作的に定義する．
3. SVHN の raw-input TwoNN と潜在 TwoNN の差の解釈を限定（指摘md §8.3 文案を
   `tab:natural` 直後に追加）．
4. \(\tau_{\mathrm{eff}}^2\) は post-hoc heuristic である旨を §5.6.1 で強化
   （指摘md §8.4 文案）；Abstract・結論にこの解釈を持ち込まない．

### T10. 再現性・データ分割・補足資料【指摘md §9】
1. §5.1 の固定順序サブセット抽出に指摘md §9.1 の限定文
   （順序固有の影響の可能性；独立分割平均としての解釈不可）を追加．
2. 固定サブセット（20,000/3,000）を用いる理由（計算コスト・実験間の同一条件）を明記
   （指摘md §9.2）．
3. Supplementary Materials / Data Availability の記述を
   `Anonymous code, configurations, data indices, seed-specific logs, and figure-generation scripts are provided as supplementary material for review.` に更新（実際に提供可能な範囲と
   整合させる；提供不能な項目は "will be released upon acceptance" のまま残す）．

### T11. 関連研究の補強【指摘md §10】
1. §2 に ID 推定器の限界（有限標本バイアス，非一様サンプリング，ノイズ・曲率，
   距離集中，混合多様体，深層表現での表現依存性）を1段落追加．
   追加候補文献（2〜4件，thebibliography に MDPI 書式で追記）：
   - Camastra & Staiano, "Intrinsic dimension estimation: Advances and open problems," *Inf. Sci.* **2016**.
   - Denti et al. (Hidalgo, heterogeneous ID) *Sci. Rep.* **2022** ほか，混合多様体系1件．
   - 既収載の `ansuini2019intrinsic`・`pope2021intrinsic` をこの文脈で再引用．
   ※ 書誌情報は追加前に必ず原典で確認する．
2. §2 または §3.1 に AU の定義と限界（posterior mean 分散に基づく；KL の座標別分解と
   同一ではない；δ は潜在スケール依存；座標依存；ID 推定量として用いない）を明記
   （指摘md §10.2）．

### T12. 図表の改善【指摘md §11；具体手順は §4 参照】
1. **Table 5（`tab:ea`）**：長い解釈注記を表から脚注/本文へ移し，数値のみの簡潔な構造に変更
   （指摘md §11.1 の推奨構造＋脚注文案 "On the Torus, AU = 2 at m=4 is an over-pruned boundary case; ..."）．
2. **Figure 2（`fig:e4_dualaxis`）**：MSE と AU/TwoNN のパネル分離，seed 数
   （AE=単一，VAE=3）と平均の定義をキャプションに明記，参照線ラベルを
   `Step-1 reference \(\hat d_{\mathrm{ID}}\)` に統一．→ 図を再生成（§4）．
3. **Figure 3（`fig:e4_multiseed`）**：図中ラベルを
   `Rounded Step-1 reference \(\hat d_{\mathrm{ID}}=10\)`（または Primary Step-1 TwoNN reference）
   に変更．→ 図を再生成（§4）．
4. **Figure 4（`fig:ep`）**：キャプションを指摘md §11.4 の文案
   （`This result does not establish that isometry is generally irrelevant ...`）に変更．
   図自体の再生成は不要（tex キャプションのみ）．

### T13. 構成の改善【指摘md §12】
1. Introduction の短縮：§1.2–1.4 の限定・注意の繰り返しを整理し，§1.4（Limitations 要約）
   を約半分に圧縮（詳細は §6 Discussion へ）．Introduction は
   (1) 問題設定 → (2) 提案 → (3) 主要結果 → (4) 主な限定，の4点構成にする．
2. §1.2 の貢献リストを再整理：
   Methodological / Diagnostic / Empirical / **Theoretical motivation** の4分類
   （Prop.2 は理論保証ではなく motivation として記載）．

---

## 4. 図の再生成手順（Phase A）

再学習は**不要**．全て `results/tables/` のキャッシュ済み JSON から再描画する．

| 図 | 生成スクリプト | データ | 修正内容 | 新出力名 |
|---|---|---|---|---|
| Figure 2 | `src/experiments/replot_E4_dualaxis.py` | `E4_extended.json` | MSE パネルと AU/TwoNN パネルを分離（例：2×2 または 1×3 構成）；参照線ラベルを `Step-1 reference $\hat{d}_{\mathrm{ID}}=10$` に | `E4_mnist_dualaxis_v43.pdf/.png` |
| Figure 3 | `src/experiments/replot_E4_multiseed.py` | `E4_300ep_multiseed.json` | 参照線ラベルを `Rounded Step-1 reference $\hat{d}_{\mathrm{ID}}=10$` に変更（線種・候補帯は現状維持） | `E4_300ep_multiseed_v43.pdf/.png` |
| Figure 4 | 再生成不要 | — | tex キャプションのみ修正（T12-4） | 変更なし |

手順：
1. 各 replot スクリプトを複製（例：`replot_E4_dualaxis_v43.py`）して修正・実行
   （既存スクリプトも上書きしない）．
2. `savefig` の出力パスを `_v43` 付き新ファイル名に変更（旧図は残す）．
3. DPI≥300 相当（PDF はベクタなので可）で PNG も併産．
4. `main_paper_MAKE43.tex` の `\includegraphics` パスを新ファイルに差し替え．

## 5. 実験の要否判断

- **必須の再実験：なし**．今回の指摘は全て (a) 文章・キャプション・表構造の修正，
  (b) キャッシュ済み結果からの再プロット，で対応可能．
- **オプション（指摘md §15「可能なら」；今回のスコープ外，将来の版で検討）**：
  - 複数データ分割 seed による補足実験（`E4_multiseed` 系の分割 seed 版）
  - 参照 AE の \(m_{\mathrm{ref}}\)-sweep の複数 seed 再実行
  - δ の潜在スケール依存性の追加解析（`EX_au_delta_sensitivity.json` の拡張）
  - 自然画像での decoder capacity × β の系統的 ablation
  これらを見送る場合，対応する限定文（T6-3, T10 ほか）を本文に必ず残すこと．

## 6. コンパイルと検証（Phase C）

```bash
cd /Users/jinno/Dropbox/Work/claude/manifold/117/manifold
pdflatex -interaction=nonstopmode main_paper_MAKE43.tex
pdflatex -interaction=nonstopmode main_paper_MAKE43.tex
pdflatex -interaction=nonstopmode main_paper_MAKE43.tex
grep -n "^!" main_paper_MAKE43.log
grep -in "undefined" main_paper_MAKE43.log
```

- エラー 0 件，未定義参照 0 件を完了条件とする．
- `mdpi.cls`（`Definitions/`）使用・pdflatex コンパイルは MAKE42 と同一．
- MDPI 後付（\authorcontributions, \funding, \dataavailability, \conflictsofinterest,
  \abbreviations）は MAKE42 から引き継ぎ，T10-3 の変更のみ反映．

## 7. 完了チェックリスト（指摘md §15 準拠）

### 本文（必須）
- [x] タイトルを診断的プロトコルが伝わる表現に変更（T1）
- [x] Abstract：VAE の \(m\ge2\hat d_{\mathrm{ID}}\) が保証でないことを明記（T2）
- [x] \(d_{\mathrm{ID}}\)/\(\hat d_{\mathrm{ID}}\)/\(d_{\mathrm{true}}\)/\(d_{\mathrm{emb}}\)/AU を全節・全図表で区別（T4, T5）
- [x] Prop.1 を理想条件下の必要条件として明記（T4-1,2）
- [x] Prop.2 を線形ガウス代理に限定，非線形への定量適用を否定（T4-3,4）
- [x] Prop.3 を heuristic rationale に改称，参照箇所を全修正（T4-5）
- [x] AU が ID/埋め込み次元の推定量でないことを Abstract と結論に明記（T2, T3）
- [x] AU の座標依存性・回転非不変性を追記（T5-2）
- [x] Step 1 参照 AE と Step 3 候補 VAE の分離を明記（T6-1,2）
- [x] 単一 seed sweep と 3seed 検証の区別を明記（T6-3）
- [x] AE 単一 seed／VAE 3seed の非対称性を表・本文に明示（T8-4）
- [x] 計算削減率の条件別表を新設（T8-3）
- [x] 固定 β 理論と KL アニーリング実験を区別（T9-1）
- [x] 自然画像の3機構を未検証仮説と明記（T9-2）
- [x] SVHN raw-input TwoNN を真値扱いしない（T9-3）
- [x] V1/V2 を運用基準と明記し，pass の非含意を列挙（T7）
- [x] 固定順序サブセットの限定と理由を明記（T10）
- [x] 関連研究に ID 推定器の限界・AU の限界を補強（T11）

### 図表
- [x] Table 5（`tab:ea`）簡潔化＋脚注化（T12-1）
- [x] Figure 2 パネル分離＋キャプション明記＋新図差し替え（T12-2, §4）
- [x] Figure 3 ラベル修正＋新図差し替え（T12-3, §4）
- [x] Figure 4 キャプション修正（T12-4）
- [x] 全図表キャプションで \(\hat d_{\mathrm{ID}}\) 表記に統一

### 体裁
- [x] pdflatex ×3 でエラー・未定義参照なし
- [x] `main_paper_MAKE43.pdf` 生成，MAKE42 の全ファイルは無変更のまま残存
- [x] 図の旧ファイルが `results/figures/` に残存（新図は `_v43`）
- [x] 追加文献の書誌情報を原典確認済み，本文中で全て引用
- [x] 英文表現の最終確認（アメリカ英語，MDPI スタイル，時制の一貫性）
