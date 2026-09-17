# MAKE49 原稿再構成仕様（方針 §7 の実行版）

**前提**: 段階 1〜5 の検証により、原稿の中心的主張の多くが支持されなかった。
本仕様は貢献の枠組みを次のように**組み替える**。

> **旧**: ID 誘導による高速な潜在次元選択法の提案
> **新**: VAE の潜在次元選択における**診断と落とし穴の実証研究**

**2026-09-17 追記**: 段階 7 で B7（GECO+L0）を追加した結果、
当初「残せる貢献の第一」としていた品質条件 $Q$ が GECO に先取りされていることが判明した。
貢献の中心は **「既存 4 手法を共通条件で評価し、各判定規則が何に支配されるかを示したこと」**
および **「閾値を持たない ARD-VAE との対比」** に移る。

この変更は方針 §11 の「診断研究への再構成または追加研究が必要」に対応する。

---

## 1. 章構成（方針 §7.1 を、得られた証拠に合わせて確定）

| 章 | 内容 | 主要な証拠 |
|---|---|---|
| 1. Introduction | 潜在次元選択の実務的問題。既存選択法（FONDUE 等）の存在を認めたうえで、**どの判定規則がどの条件で機能するかが未整理**である点を問いにする | — |
| 2. Related Work | 次元削減、ID 推定、**次元選択（FONDUE, GECO/L0, ARD-VAE）**、剪定、先行会議論文（NOLTA2025）との関係。**FONDUE 付録 G との共通性、および GECO が品質制約つき選択を先取りしていることを冒頭で開示** | §3.2–3.4、段階 7 §2 |
| 3. Protocol and Evaluation Framework | 評価の共通枠組み（品質尺度・費用計上・終了状態 5 種・予算）。**品質条件は GECO の再構成制約と同型であることを明記し、新規性を主張しない** | 設定表 §1, §4、段階 7 §2 |
| 4. Theoretical Background | 位相的必要条件と既知の埋め込み上界を**背景として**簡潔に。危うい一般化を削除 | §4.6, §2.11 |
| 5. Experimental Setup | モデル、**損失規約（reduction を明記）**、尤度（データ別）、分割、seed、比較法、費用計上 | 設定表 §2–§6 |
| 6. Results | 6.1 方法別の選択・品質・費用／6.2 ID 推定の依存性／6.3 閾値感度／6.4 読み取りの安定性／6.5 自然画像 | 段階 4・5 |
| 7. Discussion | 何が機能し何が機能しなかったか、適用範囲、閾値依存性の含意 | 段階 5 総合判断 |
| 8. Conclusions | 支持された内容と限定を短く | — |
| Supplementary | 全グリッド、seed 別ログ、補助感度、事前登録の変更履歴 | — |

**現 Table 3（applicability）は Results 末尾へ移す。** 研究前提の既知条件と実験で分かった適用範囲を同じ表に入れない。

---

## 2. Abstract（草案、記号なし・200 語以内を目標）

> Choosing the latent dimensionality of a variational autoencoder is usually done by grid search.
> Several methods promise a cheaper answer, either from an intrinsic dimension estimate of the data
> or from how many latent variables remain active after training. We asked, under one common
> protocol, which of these signals actually supports the decision. We fixed a quality criterion,
> a budget and a set of termination states in advance, and compared intrinsic-dimension guidance,
> two published dimension-selection algorithms, full grid search, coarse search and fixed
> dimensions on four datasets with five independent seeds each.
>
> Intrinsic-dimension guidance gave no measurable advantage: a plain ascending scan under the same
> quality criterion reached the same or better outcome at lower cost on every dataset, and the
> guidance step declined to apply on two of the three image datasets. The activity-based readout
> did not carry information beyond a matched set of random features. Every acceptance rule we
> examined was governed by a free threshold, and on natural images the verdict reversed with that
> threshold alone. What did transfer was the explicit quality criterion, which improved the
> published selection algorithms substantially. We report these negative results as the main
> contribution, together with the measurements that establish them.

**注意**: 数値は本文に置き、Abstract には入れない。旧「60–80% 削減」は削除済み。

---

## 3. 削除・置換する表現（方針 §7.3 を確定）

| 現在の表現 | 修正後 | 根拠 |
|---|---|---|
| "a selection principle grounded in the data itself has not been established" | 既存の ID ベース選択法（FONDUE）の存在を認め、未解決点を「どの規則がどの条件で機能するか」に具体化 | §3.1 |
| 先行研究は "measure"、本研究は "design" | 次元選択まで扱う先行研究を含む比較に変更 | §3.1 |
| "the elbow appears exactly at $m = d_{\mathrm{emb}}$" | "exactly" を削除。実測 MSE と位相的必要条件を分離 | §2.4 |
| "the protocol return[s] the correct dimension" | 「所定の品質条件下で選んだ候補」 | §2.7 |
| "correctly identified" an over-pruning boundary | 「AU 条件への不適合を示した」 | §2.7 |
| "β too small" という断定的見出し | 「指定した学習条件で活性が維持された」 | §7.3 |
| "Step 1 works"（文献値との一致で判断） | 「推定の安定性・一致を観測した」。真値との一致は主張しない | §7.3 |
| "no automatic truncation" から AE は全次元を使うと説明 | 「明示的な剪定機構を持たない」 | §7.3 |
| "conditional"（24 回） | 8 回程度。具体的な条件そのものを書く | R3-6 |
| **「最大 82%（60–80%）の削減」** | **全削除**（提案法は最も高価だった） | 段階 4 §2.3 |
| **「VAE は 1 候補の学習だけで十分」** | **全削除** | 段階 4 §2.3 |
| **AU の $m$ 依存性・飽和に関する記述** | **全削除**（実効 $\beta=\beta/m$ の交絡） | 段階 1 A2 |
| **V1/V2 を判定規則として使う記述** | 補助分析へ移すか削除 | 段階 5 E4-2 |

---

## 4. 図表（方針 §7.4 を確定。生成済みのものを記載）

| 図表 | 内容 | 状態 |
|---|---|---|
| Table A | NOLTA2025・FONDUE・本研究の 3 列差分 | 方針 §3.4 から作成 |
| Algorithm 1 + 設定表 | 入力・閾値・探索・品質・停止・出力・終了状態 5 種 | 設定表から作成 |
| **Figure A** | train（評価モード）/validation 曲線と選択済み checkpoint の test 値 | **生成済** `MAKE49_E2_{MNIST,dSprites}.pdf` |
| **Table D** | 方法別の $m$（mean±SD）、品質、総費用、終了状態 | **生成済** `MAKE49_TableD_methods.tex` |
| **Figure B** | 品質と総時間、品質と $m$ | **生成済** `MAKE49_FigB_quality_cost.pdf` |
| **Figure C** | 参照表現・標本数による ID の変化 | **生成済** `MAKE49_FigC_id_dependence.pdf` |
| **Figure D** | 閾値（$\delta$）感度 | **生成済** `MAKE49_FigD_delta_sensitivity.pdf` |
| Figure E | 同一 test 画像の再構成比較 | 未作成 |
| Figure F | 読み取り A/B/C の epoch 軌跡（別パネル） | 未作成（データは `stage3_e7a_e2.json` にあり） |
| Table C | 成功・不適合・適用見送り・再探索費用 | 段階 4/5 の終了状態内訳から作成 |
| Table E | 単純診断と AU 追加診断の比較 | **データ済** `stage5_e4_au_value.json`（雑音対照を必ず併記） |

**caption の必須事項**: seed 数、データ分割、評価モード、$\beta$、尤度。
**図中で $\hat d_{\mathrm{ID}}$ を真値のように表示しない。**
既存 14 表は書式を全件監査する（有効数字 3 桁、指数表記統一、±の前後）。

---

## 5. ファイル運用（方針 §8）

- 現行系列: `main_paper_NN47/48` 系と `main_paper_MAKE47` 系。
- 改訂版は **`main_paper_NN49.tex` と `main_paper_MAKE49.tex`** として番号を揃える。
- MAKE48 の書誌修正を NN49 に反映する。
- `CLAUDE.md` の番号規約を更新する。

---

## 6. 本仕様で未確定の事項（著者判断が必要）

1. **枠組みの変更そのもの**。「方法提案」から「実証研究」への再構成は、
   投稿先の適合性に関わる判断である。MAKE（Machine Learning and Knowledge Extraction）は
   実証・再現性研究を受け入れる誌面だが、Editor への説明が必要になる。
2. **B7（GECO/L0）・B8（ARD-VAE）を追加するか**。追加すれば R2-3 への回答が強くなるが、
   別のモデル族の実装が要る。追加しない場合は不足として明記する。
3. **$\beta$ 梯子全体での再評価を行うか**。現在の否定的結論はすべて $\beta=1$ での観測である。
4. **改訂期限**。追加実験が期限内に終わらない場合、方針 §9.3 に従い
   延長依頼を検討する。未完了の実験を実施済みと書かない。
