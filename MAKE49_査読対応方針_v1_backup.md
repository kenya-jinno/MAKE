# MAKE49 — MDPI MAKE 査読コメント対応方針

対象原稿: `submission_MAKE47/main_paper_MAKE47.pdf`（投稿版）
現行最新: `main_paper_MAKE48.tex`（編集部の言語・書誌チェック対応のみ。本文内容は MAKE47 と同一）
改訂版: **`main_paper_MAKE49.tex`**（本文書の計画に基づく査読対応版）

---

## 0. 結論（先に要点）

**判定の読み**: Reviewer 2 が 6 項目すべて「Must be improved」を付けており、実質的に Reject 寄りの Major Revision。Reviewer 1 は Introduction のみ Must、Reviewer 3 は Methods と Conclusions が Must。**採否を分けるのは Reviewer 2 の 3 点（新規性・理論的厳密性・ベースライン比較欠如）であり、ここを実験で埋めない限り再投稿しても通らない。**

**最重要の 4 手**（これだけは必ずやる）:

1. **既存の次元選択手法とのベースライン比較実験を新設**（R2-3 ＝ R3-5）。現状は「自分のモデルの全グリッド探索」としか比較していないという指摘は事実であり、最も正当な批判。
2. **窓幅の係数 2 を Whitney の埋め込み定理から導出**（R3-1 ＝ R2-2 の一部）。現行原稿は「係数 2 は第一原理から導いたものではない」と自ら書いてしまっている。実際には $d \le d_{\mathrm{emb}} \le 2d$ が定理として言えるため、命題を 1 つ追加するだけで「場当たり的ヒューリスティック」という批判の中核が崩せる。
3. **会議版（NOLTA 2025, 4 ページ）との差分を定量表で明示**（R2-1）。MDPI は会議版からの実質的拡張の証明を要求する。
4. **アルゴリズム擬似コードと全閾値の一覧表を追加**（R3-7）。「ad-hoc な寄せ集め」という印象を、再現可能な手続きとして提示し直す。

**所要**: 追加 GPU 計算 約 15〜25 時間、原稿改訂 実働 5〜8 日。MDPI の標準改訂期限（10 日）では足りないため、**編集部に 3〜4 週間の延長を申請すること**を推奨。

---

## 1. 査読スコアの整理

| 評価項目 | R1 | R2 | R3 |
|---|---|---|---|
| English | Fine | Can be improved | Fine |
| Introduction / background | **Must** | **Must** | Can |
| Research design | Can | **Must** | Can |
| Methods description | Can | **Must** | **Must** |
| Results presentation | Can | **Must** | Can |
| Conclusions supported | Yes | **Must** | **Must** |
| Figures and tables | Yes | **Must** | Can |

R1 は体裁・可読性中心（実質 Minor〜Major の境界）、R3 は建設的な Major、R2 は全否定的な Major。R2 の 4 点は他 2 名の指摘と重複する部分（ベースライン、表の体裁）があり、そこを厚く直せば 3 名同時に効く。

---

## 2. 全コメントの分類と優先度

ID 体系: `R{査読者番号}-{項番}`

| ID | 要旨 | 種別 | 優先度 | 追加実験 |
|---|---|---|---|---|
| R2-3 / R3-5 | 既存の次元選択・枝刈り手法との比較がない | 実験 | **P0** | 要 |
| R2-1 | 会議版からの増分が小さい | 記述＋実験 | **P0** | 間接的に要 |
| R2-2 | 新規アーキテクチャ・目的関数がなく理論的裏付けが弱い | 記述＋理論 | **P0** | 一部要 |
| R3-1 | $2\hat d_{\mathrm{ID}}$・V1 閾値・V2 範囲の根拠が恣意的 | 理論＋記述 | **P0** | 不要 |
| R2-4 | Table 4 の数値書式不統一・括弧の破綻 | 体裁 | **P0** | 不要 |
| R3-7 | Step 1–3 の擬似コードと閾値一覧がない | 記述 | **P0** | 不要 |
| R3-2 | MNIST/Fashion-MNIST 以外のデータセット検証 | 実験 | **P1** | 要 |
| R1-9 | train/val/test 分割・学習曲線・シード数増 | 実験 | **P1** | 要 |
| R3-3 | AE 潜在経由 ID 推定のモデル依存性 | 実験 | **P1** | 要 |
| R3-4 | 命題 2 からの主張を限定せよ | 記述 | **P1** | 不要 |
| R1-3 | Introduction の前方参照が多すぎる | 記述 | **P1** | 不要 |
| R1-4 | Introduction に結論が混入 | 記述 | **P1** | 不要 |
| R1-1 | Abstract から記号を排除 | 記述 | **P2** | 不要 |
| R1-2 | 「32 や 64」の慣例に出典を付けよ | 文献 | **P2** | 不要 |
| R1-5 | §1.2 の未定義記号 | 記述 | **P2** | 不要 |
| R1-6 | MST など graph-based 手法の参照追加 | 文献 | **P2** | 不要 |
| R1-7 | 証明末尾の □ の説明 | 体裁 | **P2** | 不要 |
| R1-8 | CUDA のバージョン | 体裁 | **P2** | 不要 |
| R1-10 | DOI 表記の統一 | 体裁 | **P2** | 不要 |
| R3-6 | "conditional" の繰り返しを削減 | 記述 | **P2** | 不要 |
| 全体 | 英文の平易化（R2） | 記述 | **P1** | 不要 |

---

## 3. 各コメントへの対応方針

### 3.1 Reviewer 2（最重要）

#### R2-1: 会議版（Obata & Jin'no, NOLTA 2025, 4 ページ）からの増分が小さい

> "merely adds multi-seed runs, boundary experiments, and empirical screening thresholds."

**評価**: 部分的に正当。現行原稿は §1.2 末尾に 3 行で「substantially extends」と書いているだけで、**差分の証拠を示していない**。MDPI は会議版拡張について実質的な新規性の説明を明示的に求めるため、必ず表で示す。

**対応**:

1. §1.2 の直後に小節「**Relation to the Preliminary Conference Version**」を新設し、下表を置く。

| 要素 | NOLTA 2025（4 頁） | 本論文 |
|---|---|---|
| 手法の形式化 | ID と AU の対応の観察 | 3 段階プロトコル（推定→設定→検証）＋事前登録型 V1/V2 判定規則 |
| 理論 | なし | 命題 1（位相的下界）、**命題 2（Whitney 由来の窓幅、新規追加）**、命題 3（線形 β-VAE 活性化条件）＋付録の完全証明 |
| 失敗診断 | なし | 失敗モード分類表（5 種）＋合成・実データでの実証 |
| 統計的裏付け | 単一シード | 3〜5 シード（改訂で MNIST は 5 シードへ）、ブートストラップ CI、DANCo/ESS 相互検証 |
| ベースライン比較 | なし | **既存 6 手法との比較（改訂で新設）** |
| データセット | MNIST のみ | MNIST, Fashion-MNIST, **dSprites, KMNIST（改訂で追加）**, CIFAR-10, SVHN, 合成多様体 2 種 |
| 適用限界 | なし | Conv-VAE・自然画像での境界条件を 3 シードで定量化 |
| 図表 | 図 3 点 | 図 8 点・表 14 点（うち会議版と共通は 0 点） |

2. Cover letter でも「会議版と共通する図表は 0 点、本文の 90% 以上が新規」と明記する。改訂で新実験を 2 系統足すことで、この主張はさらに強くなる。

3. §1.2 の該当文を、控えめな "extends" から具体的な列挙に置き換える。

---

#### R2-2: 新規アーキテクチャ・目的関数・微分可能機構がなく、既製ツールの寄せ集めで理論的裏付けがない

**評価**: 位置づけの争点。反論すべき部分と、実際に強化すべき部分がある。

**対応（3 本立て）**:

**(a) 貢献の種類を明示的に主張する（反論）**

Introduction 冒頭に「本論文の貢献の種類」を 1 段落で宣言する。要旨は次のとおり。

- 本論文が解く問題は「新しい表現学習モデルの提案」ではなく「**既存モデルを使う実務者が $m$ をどう決め、決めた後どう検証するか**」という運用上の問題である。
- MAKE のスコープ（Machine Learning and **Knowledge Extraction**）は方法論・評価プロトコル・診断手法を明示的に含む。
- 新アーキテクチャを提案しないことは欠陥ではなく設計判断である。プロトコルが特定アーキテクチャに依存すると、汎用の設計指針として機能しない。
- ただしこの主張は「理論が弱くてよい」という意味ではないので、(b) で実際に強化する。

**(b) 理論を実際に厚くする（これが本丸）**

現行原稿は §3.2.2 で自ら「**The factor 2 is not derived from first principles but is a default that worked in our experiments**」と書いており、R2-2 と R3-1 の両方を招いている。これは撤回できる。

- **新規命題（Whitney 括弧）を追加**: 滑らかなコンパクト $d$ 次元多様体は $\mathbb{R}^{2d}$ に埋め込める（強 Whitney 埋め込み定理, $d \ge 1$）。命題 1 の $d_{\mathrm{emb}} \ge d$ と合わせて

  $$d \;\le\; d_{\mathrm{emb}} \;\le\; 2d$$

  が得られる。すなわち **$[\hat d_{\mathrm{ID}},\, 2\hat d_{\mathrm{ID}}]$ は、任意の滑らかコンパクト多様体に対して $d_{\mathrm{emb}}$ を含むことが保証される唯一の自然な区間**であり、係数 2 は恣意的な既定値ではない。残る近似は「$\hat d_{\mathrm{ID}}$ の推定誤差」のみであり、これは別途 Step 1 の CI と相互検証で扱っている。
- この命題により §3.2.2 の自己否定的な文を全面的に書き換える。窓幅の正当化が「実験的に効いた」から「定理で括れる」に変わる。
- 注意書きも忘れず付す: Whitney は理想的な滑らかコンパクト多様体に対する上界であり、実データの $\hat d_{\mathrm{ID}}$ がバイアスを持てば区間もずれる。トーラスで $m \ge 5$ が必要だったのは AU 打ち切りの挙動であって埋め込み可能性ではない（$d_{\mathrm{emb}}=3 \le 4 = 2d$ は満たされている）。この区別を本文に明記する。
- 命題 3（活性化条件 $\lambda_j > \beta\tau^2$）の証明を付録で完全な形に整え、系として「$d^*(\beta) = |\{j : \lambda_j > \beta\tau^2\}|$ は $\beta$ について単調非増加」を追加する。これは V1/V2 の $\beta$ 依存性の根拠になる。

**(c) 「自動次元発見の微分可能機構がない」への応答**

自前で新機構を作るのではなく、**既存の微分可能な自動次元決定機構（ARD-VAE）をベースラインとして実装し、比較する**（→ R2-3 の表 B4）。これにより

- そのような機構は既に存在すること、
- しかし $\beta$ や事前分布の設定に同じだけ敏感で、単体では検証手段を持たないこと、
- 本プロトコルの Step 3 は ARD-VAE に対しても検証器として使えること

を示す。「機構を作らなかった」ではなく「機構を含めて比較し、検証層が独立に必要であることを示した」という構図に変える。これは R2-2 への最も説得力のある応答になる。

---

#### R2-3 ＝ R3-5: 確立された次元選択・枝刈り手法との比較がない

**評価**: 完全に正当。最優先で埋める。

**対応**: 新設 §5.x「**Comparison with Established Dimension-Selection Baselines**」。MNIST と Fashion-MNIST で、各手法が出力する $m$ を比較する。

| ID | ベースライン | 種別 | 出力 | 計算コスト |
|---|---|---|---|---|
| B1 | PCA 累積寄与率 95% / 99% | 線形・解析的 | $m$ | 秒 |
| B2 | Minka の PCA 次元自動選択（Laplace 近似 / BIC） | 線形・確率的 | $m$ | 秒 |
| B3 | Bayesian PCA + ARD（Bishop 1999、既に引用済み） | 線形・確率的 | $m$ | 分 |
| B4 | ARD-VAE（潜在次元ごとに事前分散を学習し関連度で枝刈り） | 非線形・微分可能 | $m$ | 3 シード × 数点 |
| B5 | 広い AE の潜在ユニットを分散・大きさで事後枝刈り | 非線形・事後枝刈り | $m$ | 既存重み再利用 |
| B6 | 検証再構成 MSE のエルボー（全グリッド掃引） | 慣例手法 | $m$ | 全グリッド |
| B7 | 検証 ELBO 最良点（全グリッド掃引） | 慣例手法 | $m$ | 全グリッド |
| B8 | **本手法（ID 誘導窓 ＋ V1/V2 検証）** | 提案 | $m$＋合否 | 窓内のみ |

**報告する列**: 選択された $m$ / 線形プローブ精度 / 検証 MSE / 学習回数 / 事後検証の有無。

**主張の落とし所**（事前に想定される結果に対して、どう書いても誠実になるように設計する）:

- B1/B2/B3 は線形手法なので、非線形 AE/VAE のボトルネックとしては過大評価しやすい（MNIST の PCA 95% は数十次元になるはず）。この差自体が「線形基準では不十分」という知見になる。
- B6/B7 は良い $m$ を出すが全グリッドを要する。本手法との差は**精度ではなく探索コストと事後検証の有無**にある。ここを正直に書く。既に §5.7 でコスト議論はしているので、それをベースライン表に統合する。
- B4（ARD-VAE）が本手法と近い $m$ を出した場合でも問題ない。「データ側の幾何から窓を出す本手法と、モデル側の自動枝刈りは独立に同じ答えに到達した」という三角測量の議論にできる。むしろ望ましい結果。
- **表には必ず「本手法が全ベースラインに勝つ」とは書かない。** 勝敗ではなく「どの手法が何を保証し、何を保証しないか」の整理表として提示する。R3 も R2 も過大主張には厳しいので、控えめな提示のほうが通る。

---

#### R2-4: Table 4 の数値書式が列内で不統一、括弧が破綻

**評価**: 事実。現行 `tab:e1` は同一列に `1.016 ± 0.315` と `(6 ± 2) × 10^{-4}` が混在し、さらにセル内に `← elbow (= d_emb)` が入って括弧が二重になっている。

**対応**:

1. `tab:e1` を全面的に組み直す。
   - 単位を列見出しに括り出す: `Swiss Roll MSE ($\times 10^{-3}$)` のように統一し、セル内は素の小数のみにする。
   - 有効数字を 3 桁に統一する。
   - `← elbow` の注記はセル内から外し、独立の列（例: `Elbow`）か太字＋キャプションでの説明に移す。
2. **全 14 表を横断的に監査する**。R2 は「The draft is very rough」と書いており、1 表だけ直しても印象は変わらない。チェック項目:
   - 同一列内の桁数・指数表記の統一
   - `±` の前後のスペース
   - セル内の括弧のネスト
   - 有効数字の統一（3 桁）
   - 太字の使用基準の統一
   - 表番号と本文参照の一致
3. 監査結果は改訂レターに 1 行で報告する（「全 14 表を統一書式で組み直した」）。

---

### 3.2 Reviewer 1

#### R1-1: Abstract から記号を排除

現行 Abstract は 243 語で、**MDPI の推奨上限（約 200 語）も超えている**。記号（$m$, $\hat d_{\mathrm{ID}}$, $\lambda_j > \beta\tau^2$, $\beta_{\max}$）を全廃したうえで 200 語以内に圧縮する。

書き換え方針:
- 「$m$」→ "the bottleneck dimension"
- 「$\hat d_{\mathrm{ID}}$」→ "an estimate of the data's intrinsic dimension"
- 「$m \in [\hat d_{\mathrm{ID}}, 2\hat d_{\mathrm{ID}}]$」→ "a search window running from that estimate to twice that estimate"
- 「$\lambda_j > \beta\tau^2$」→ 数式は落とし、"a linear-Gaussian rate–distortion surrogate that predicts which latent coordinates survive KL regularization" とする
- 「$\beta_{\max}=4$」→ "the convolutional schedule we tested"
- 数値（60–80%、3 seeds）は残す。読者の判断材料になり、記号ではないため指摘の対象外。

#### R1-2: 「32 や 64 という切りのよい数字」の出典

現状は無出典の主張。以下を候補として引用を付ける（**引用前に各論文の実際の潜在次元設定を必ず確認すること**）。

- Locatello et al., ICML 2019, *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* — 大規模比較で全モデルの潜在次元を 10 に固定しており、「慣例で決める」ことの最良の証拠。
- Ha & Schmidhuber, NeurIPS 2018, *World Models* — 潜在 32。
- Higgins et al., ICLR 2017（既に引用済み [10]）— 10 / 32。
- Kingma & Welling, ICLR 2014（既に引用済み [1]）— MNIST で 3, 5, 10, 20, 200 を掃引。

文の書き換え案: 慣例値そのものを批判するのではなく、「広く使われている大規模比較研究でも潜在次元はデータに依らず固定されている」という事実を出典付きで述べる形にする。そのほうが穏当で、かつ本論文の動機づけとして強い。

#### R1-3: Introduction の前方参照が多すぎる

現状、§1 内に他節への `\ref` が **19 か所**ある（`sec:theory` 4 回、`sec:protocol` 2 回、`sec:exp` 2 回ほか）。

**対応**: §1.1〜§1.3 から前方参照を原則全廃し、末尾の構成案内パラグラフに集約する。目標は §1 全体で 5 か所以内。

- 貢献リスト（§1.2 の enumerate）から各項目末尾の `(Section~\ref{...})` を削除する。どの節に対応するかは構成案内で示せば足りる。
- 表への前方参照（`tab:failure`, `tab:applicability`）も削除し、言葉で「failure-mode taxonomy」「applicability summary」と述べるにとどめる。

#### R1-4: Introduction に結論が混入

**評価**: 正当。§1.2 の貢献リストには具体的な結果値（$\hat d_{\mathrm{ID}} \approx 10$–12、60–80% 削減、3 seeds、$\beta_{\max}=4$ で打ち切りなし等）が詰め込まれており、§1.4「Limitations of This Paper (Summary)」は §6.4 とほぼ全面的に重複している。

**対応**:

1. §1.2 の貢献リストを「**何をやったか**」のレベルに揃える。数値は原則すべて削除し、§5・§7 に移す。たとえば貢献 3 の (a)(b)(c) の詳細（$m \gg \hat d_{\mathrm{ID}}$、$m \approx 2\hat d_{\mathrm{ID}}$ 超で膨張、$\beta_{\max}=4$ で打ち切りなし）は Conclusions に既に同内容があるので、Introduction 側を削る。
2. §1.4「Limitations of This Paper (Summary)」を **3 文の範囲宣言に圧縮**する（検証範囲は合成多様体・MNIST・Fashion-MNIST の全結合系であり、畳み込み系と自然画像は適用限界の探索段階である、という事実のみ）。詳細は §6.4 に一本化する。
3. これにより Introduction は約 1 ページ短くなる。R1-3 の改善と相乗する。

#### R1-5: §1.2 の未定義記号

R1-4 の圧縮と同時に解消する。§1.2 に残す記号は $m$ と $\hat d_{\mathrm{ID}}$ の 2 つだけとし、いずれも初出時に 1 行で言葉による定義を添える。$\lambda_j$, $\tau^2$, $\beta$, $d_{\mathrm{emb}}$, $\mathrm{AU}_{\mathrm{sat}}$, $m_{\mathrm{ref}}$, $m_{\mathrm{ver}}$ は Introduction から全廃し、§3.1 の記号表が初出になるようにする。

#### R1-6: MST など graph-based 手法の参照追加

**評価**: 指摘された文献（DOI: 10.1109/ACCESS.2022.3190505）は

> Gałka, Ł.; Karczmarek, P.; Tokovarov, M. *Isolation Forest Based on Minimal Spanning Tree.* IEEE Access **2022**, *10*, 74175–74186.

で、実際には**次元削減ではなく MST に基づく異常検知**の論文である。そのまま「次元削減手法」として引用すると不正確になるので、**正確な位置づけで引用する**。

**対応**: Related Work に短い段落「Graph-based characterization of data structure」を追加する。

- 主役として **Costa & Hero, IEEE Trans. Signal Process. 2004, 52(8), 2210–2221, DOI 10.1109/TSP.2004.831130, "Geodesic Entropic Graphs for Dimension and Entropy Estimation in Manifold Learning"** を引用する。これは MST・k-NN グラフに基づく内在次元推定器そのものであり、本論文の Step 1 の直接の代替手法として真に関連が深い。查読者の意図（graph-based 手法への言及）を学術的に正しい形で満たす。
- そのうえで「MST に基づくグラフ構造はデータ構造の記述に広く用いられており、異常検知などにも応用されている [Gałka et al. 2022]」として指摘文献を並置する。誇張せず、しかし確実に引用する。
- さらに **Costa & Hero の MST 推定器を Step 1 の第 3 の推定器として実験に加える**（→ R3-3）。引用だけでなく実際に使えば、指摘への応答として格段に強い。

#### R1-7: 証明末尾の □ の説明

最初の `\begin{proof}`（§4.1、命題 1 の証明）の直前に脚注を置く。

> The symbol $\square$ at the end of a proof denotes its completion (Q.E.D.).

加えて §3.1 の記号表にも 1 行追加する。

#### R1-8: CUDA のバージョン

§5.1「Experimental Setup」に環境情報を 1 文追加する。**補足資料の environment info から実際の値を確認して記載すること**（推測で書かない）。記載すべき項目:

- OS、Python、PyTorch、CUDA Toolkit、cuDNN、NVIDIA ドライバ、GPU（RTX 3070、VRAM）
- NumPy / scikit-learn / scikit-dimension 等の主要ライブラリ版

#### R1-9: train/val/test 分割・学習曲線・シード数増

**評価**: 現状は train/test の 2 分割のみで、検証集合がない。過学習の有無を示す図もない。ML 系査読としては妥当な要求。

**対応**:

1. **3 分割の導入**: MNIST / Fashion-MNIST を 20,000 train / 3,000 validation / 3,000 test に変更する。既存の $n_{\mathrm{test}}=3{,}000$ を test として維持し、validation を新たに切り出す。ベースライン B6/B7 の選択、および AU 閾値の確認は validation で行い、報告値は test で出す。この分離は R2-3 のベースライン比較の公平性にも必要。
2. **学習曲線の図を新設**（Figure、代表的な $m \in \{4, 12, 20, 64\}$）:
   - 上段: 再構成 MSE の train / validation / test 曲線 vs エポック
   - 下段: 線形プローブ test 精度 vs エポック（査読者が求めた "Accuracy" に対応する量）
   - 併せて AU vs エポックを重ねると、§6.3 の「大域構造が先、局所が後」の議論とも接続できる。
   - 本文で「train と validation の乖離はエポック 300 時点で MSE 換算 X% 以内であり、過学習は観測されない」と明示する。
3. **シード数の増加**: MNIST の主検証表（`tab:mnist_s3`）と Fashion-MNIST（`tab:fashion_s3`）を **3 シード → 5 シード**（42, 123, 777, 2024, 31337）に拡張する。全グリッド 11 点 × 5 シード = 55 runs。合成多様体は既に 3〜5 シードなので据え置き可。
   - 「Accuracy を使え」という助言については、**オートエンコーダの主タスクは再構成であり分類ではない**ため、MSE を主指標に据えたうえで線形プローブ精度を補助指標として併記する、と丁寧に説明する。これは拒否ではなく、求められた情報を適切な形で提供する対応。

#### R1-10: DOI 表記の統一

現状、37 件中 **DOI 付きは 12 件のみ**（`bengio2013representation`, `fefferman2016manifold`, `facco2017twonn`, `okamoto2023latent`, `jinno2023multimodal`, `dai2024latent`, `dai2026island`, `ceruti2014danco`, `camastra2016intrinsic`, `allegra2020data`, `tippingbishop1999ppca`, `lecun1998`, `johnsson2015ess`）。

**方針: 全件に DOI を付ける**（MDPI 推奨）。

- ジャーナル論文: 出版社 DOI。
- ICLR / NeurIPS / ICML 等の会議論文で DOI がないもの: **arXiv DOI（`10.48550/arXiv.XXXX.XXXXX`）を付与する**。これで全件統一できる。
- DOI が一切存在しないもの（技術報告書、ワークショップ論文など）: 安定 URL とアクセス日を統一書式で記載し、脚注または改訂レターでその旨を述べる。
- MAKE48 で追加済みの DOI と重複しないよう、全 37 件を 1 パスで再点検する。

---

### 3.3 Reviewer 3

#### R3-1: $2\hat d_{\mathrm{ID}}$・V1 閾値・V2 範囲の根拠

**対応（3 分割）**:

**(a) 係数 2** → §3.2.2 の Whitney 命題で解決（R2-2(b) 参照）。「first principles から導いていない」という現行の記述を撤回し、$d \le d_{\mathrm{emb}} \le 2d$ を根拠として提示する。

**(b) V1 閾値 0.9（活性化率）** → 現状は付録に感度範囲 $(0.67, 0.94)$ があるだけ。これを本文に上げ、さらに根拠を補強する。

- **設計上の根拠**: V1 は「打ち切りが起きたか」の二値判定であり、「打ち切りなし」の帰無挙動は $\mathrm{AU} = m$（活性化率 1.0）である。閾値 0.9 は $m \ge 10$ において「少なくとも 1 座標が崩壊した」ことを検出する最小の水準に相当する。この解釈を明記する。
- **経験的分離の可視化**: 本論文の全設定（合成 2 種 × 複数 $m$、MNIST、Fashion-MNIST、Conv-VAE $\beta_{\max} \in \{4,10,20\}$、CIFAR-10、SVHN）の活性化率を 1 枚の散布図にプロットし、合格群と不合格群が閾値 0.9 の周りでどれだけ離れているかを示す。現状「判定は $(0.67, 0.94)$ で不変」という数字はあるので、それを図にする。
- **表現の修正**: 「working default」という自己弁護的な表現を減らし、「帰無挙動 $\mathrm{AU}=m$ からの逸脱を検出する閾値であり、本論文の全設定で判定は区間 $(0.67, 0.94)$ に対して不変」という事実の記述に置き換える。

**(c) V2 の係数 3（上限 $3\hat d_{\mathrm{ID}}$）** →

- Whitney 上界 $2d$ に、Step 1 推定器のバイアス余裕を 1 段分加えた値として位置づける。すなわち「$\hat d_{\mathrm{ID}}$ が真値を 1.5 倍程度過小推定しても上限を割らない」設計であることを述べる。
- 感度区間 $(2.3, 6.8)$（シード別 $(2.3, 6.6)$）を本文の表に移し、下限係数の感度区間 $(0.70, 1.0]$ も併記する。
- **全閾値を 1 表にまとめる**（→ R3-7 の閾値一覧表）。既定値・根拠・判定不変区間・その区間を決めているケースを列にする。

#### R3-2: MNIST/Fashion-MNIST 以外での検証

**対応**: 全結合パイプラインをそのまま使える 2 データセットを追加する。

1. **dSprites**（推奨度: 高）— 生成因子が既知（形状 3 × スケール 6 × 回転 40 × 位置 32 × 32、連続因子は 5 種）。**実画像に近い設定で「正解次元」が存在する唯一の現実的選択肢**であり、合成多様体と実データの間を橋渡しする。Step 1 の $\hat d_{\mathrm{ID}}$ と既知因子数の比較、Step 3 の V1/V2 判定の両方が検証できる。R2-2 の「理論的裏付けが弱い」への実証的応答にもなる。
2. **KMNIST**（推奨度: 中〜高）— MNIST と同形状・同規模で前処理が共通。「MNIST 固有の現象ではない」ことを最小コストで示せる。
3. **（余力があれば）非画像の表形式データ** — 例: UCI Human Activity Recognition（561 次元センサ特徴）。MAKE の読者層に対して「画像専用ではない」ことを示す価値は大きいが、P2 扱いでよい。

いずれも既存の `src/data/loaders.py` に loader を追加し、`Exp_Mnist_Multi_300` 系のスクリプトを流用できる。

Step 3 が自然画像で未解決である点は依然として限界として残るが、**検証済み領域が MNIST 系 2 種から 4 種に広がる**ことで「MNIST だけの話ではないか」という疑いは大きく減る。

#### R3-3: AE 潜在経由の ID 推定のモデル依存性

**評価**: 現行 §5.8（Exp-Real-Geom）で AE/CAE/DAE/IsometricAE の 4 モデル比較を 3 シードで行っているが、査読者は「より体系的に」と求めている。

**対応**: Exp-Real-Geom を拡張する。

- **参照表現を追加**: 現行 4 モデル + **PCA 射影**（線形基準）+ **ランダム射影**（幾何を保存しない対照）+ **入力空間直接推定**。ランダム射影を入れると「ID 推定値が参照表現の学習内容にどれだけ依存するか」の下限対照になり、議論が締まる。
- **推定器を 3 種に**: TwoNN + MLE + **Costa–Hero の MST/測地エントロピーグラフ推定器**（R1-6 の文献をここで実際に使う）。原理の異なる 3 手法の一致・不一致を見る。
- **モデル依存性の定量指標を定義**: 固定 $m_{\mathrm{ref}}$ における参照モデル間の $\hat d_{\mathrm{ID}}$ のばらつき（標準偏差または最大 − 最小）を「model-dependence spread」として報告し、$m_{\mathrm{ref}}$ に対する推移を図示する。「$m_{\mathrm{ref}} \gg \hat d_{\mathrm{ID}}$ でこの spread が縮む」という本論文の主張が、指標として直接読めるようになる。
- MNIST・Fashion-MNIST の 2 データセット × 3 シードで実施。

#### R3-4: 命題 2 からの主張を限定せよ

**評価**: 正当。線形ガウス代理から非線形 VAE への外挿は現状 $\tau^2_{\mathrm{eff}}$ の事後整合性チェックに依存しているだけ。

**対応**:

1. 命題（活性化条件）の直後に「**Scope of Proposition**」という小さな囲み（remark 環境）を置き、仮定と非転移事項を箇条書きで明示する。
   - 仮定: 線形デコーダ、等方ガウス観測ノイズ、対角ガウス事後、大域最適解、固定 $\tau^2$。
   - 転移しない事項: 非線形デコーダでの $\tau^2_{\mathrm{eff}}$ の空間的非一様性、最適化の局所解、学習途中の動的挙動、データが部分多様体の和である場合の固有値スペクトルの解釈。
2. 本文全体で命題 2 を援用している箇所を洗い出し、動詞を統一する。「predicts」「implies」→ **「motivates」「is consistent with」** に置換する。該当箇所は §1.2、§3.2.2、§3.2.3、§4.2、§5.2、§6.4、§7 に分布している。
3. Conclusions の該当文（「a theoretical surrogate, not a quantitative theorem for nonlinear VAEs」）は既に適切なので維持し、同じ強さの限定を本文の各援用箇所にも入れる。
4. 逆に Whitney 命題（新規）は**厳密な定理**なので、こちらは限定せず明確に主張する。「限定すべき主張」と「厳密な主張」を峻別することで、論文全体の主張の信頼性が上がる。

#### R3-5: より強いベースライン（再構成・ELBO・下流検証）

→ R2-3 と統合。ベースライン表の B6（検証 MSE エルボー）、B7（検証 ELBO）、および全手法に対する下流線形プローブ評価が、この指摘に直接対応する。

#### R3-6: "conditional" の繰り返し

現状 **24 回**出現。読みにくさの原因になっている。

**対応**: 8 回程度に削減する。残す箇所は Abstract 1 回、§1.2 で概念を導入する 1 回、§3.4（適用範囲表）1 回、§6 冒頭 1 回、§7 で 2 回、表キャプション 2 回。それ以外は削除するか、具体的な条件そのものを書く（「conditional on the estimator being stable」→「when the multi-seed spread of $\hat d_{\mathrm{ID}}$ is below X」）。

同様に反復している防御的表現も併せて棚卸しする:

- "not an estimator of intrinsic or embedding dimension"（複数回）
- "under the tested settings" / "under our experimental conditions"
- "working default"
- "does not establish / does not guarantee" の連鎖

R2 が「英語を改善せよ」と付けているのは、この防御的反復と長大な挿入句（em-dash の多用）が主因と考えられる。**1 文 1 主張・平均 25 語以内**を目安に §1、§3、§6 を書き直す。

#### R3-7: 擬似コードと閾値一覧

**対応**: 2 点を新設する。

1. **Algorithm 1: ID-Guided Bottleneck Selection**（§3.2 の冒頭、図 1 のパイプライン図の直後）
   ```
   入力: データ X、m グリッド M、β、AU 閾値 δ、シード集合 S
   Step 1: 参照 AE を m_ref ∈ {64,128,256} で学習 → 潜在で TwoNN/MLE
           収束チェック → d̂_ID（主推定量 TwoNN、相互検証 MLE）
   Step 2: AE   → 探索窓 W = [d̂_ID, 2·d̂_ID] ∩ M
           VAE  → m_ver = min{ m ∈ M : m ≥ 2·d̂_ID }（学習前に確定）
   Step 3: |S| シードで学習 → AU(m_ver), MSE(m)
           V1: AU/m_ver ≤ 0.9 ?
           V2: d̂_ID ≤ AU ≤ 3·d̂_ID ?
           両方成立 → pass / 不成立 → 失敗モード表で診断
   出力: 窓 W、検証点 m_ver、V1/V2 判定、失敗モード（不合格時）
   ```
   実際の記載は MDPI の `algorithm` 環境（`algorithm2e` or `algorithmic`）で組む。`mdpi.cls` での利用可否を事前に確認すること。

2. **Table: All decision thresholds**（§3.2.3 内）

   | 記号 | 既定値 | 役割 | 根拠 | 判定不変区間 |
   |---|---|---|---|---|
   | 窓上限係数 | 2 | 探索窓の上端 | Whitney 埋め込み定理 $d_{\mathrm{emb}} \le 2d$ | — |
   | V1 活性化率 | 0.9 | 打ち切り検出 | 帰無挙動 AU $=m$ からの逸脱 | (0.67, 0.94) |
   | V2 上限係数 | 3 | 次数整合の上側 | Whitney 上界 + 推定バイアス余裕 | (2.3, 6.8) |
   | V2 下限係数 | 1 | 過剰枝刈り検出 | $\hat d_{\mathrm{ID}}$ 自身 | (0.70, 1.0] |
   | AU 閾値 $\delta$ | $10^{-2}$ | 活性判定 | 文献既定値 | $\beta \in [2,8]$ で安定 |
   | TwoNN $k$ | 2 | 推定器定義 | 定義により固定 | — |
   | MLE $k$ | 10 | 近傍数 | $k \in \{5,10,20\}$ で安定 | — |

   この表は R3-1 への回答も同時に果たす。

---

## 4. 追加実験計画

| # | 実験名 | 対応コメント | 内容 | 学習回数 | 概算 GPU |
|---|---|---|---|---|---|
| A1 | `Exp_Baselines` | R2-3, R3-5 | B1–B7 のベースライン実装と MNIST/Fashion での $m$ 選択・下流評価 | 約 20（B4 のみ学習、他は解析的 or 既存重み再利用） | 3–5 h |
| A2 | `Exp_Mnist_Multi_300`（拡張） | R1-9 | 3 → 5 シード、3 分割、学習曲線ログ追加 | 55 | 6–8 h |
| A3 | `Exp_Fashion_MS`（拡張） | R1-9 | 同上 | 25 | 3 h |
| A4 | `Exp_dSprites` | R3-2 | dSprites で全プロトコル（Step 1–3）、3 シード | 21 | 3 h |
| A5 | `Exp_KMNIST` | R3-2 | KMNIST で全プロトコル、3 シード | 21 | 3 h |
| A6 | `Exp_Real_Geom`（拡張） | R3-3, R1-6 | 参照モデル 7 種 × 推定器 3 種 × 2 データセット × 3 シード | 既存重み + PCA/RP 追加 | 2 h |
| A7 | 学習曲線図の生成 | R1-9 | A2 のログから train/val/test 曲線と probe 精度曲線 | 0（再解析） | 0.5 h |
| | **合計** | | | **約 140 runs** | **約 20–25 h** |

**注意事項**:

- 全実験で既存の再現性規約（`SEED` 固定、`cudnn.deterministic = True`）を維持する。
- A2/A3 で分割を変更するため、**既存の MNIST/Fashion の全数値が変わる可能性がある**。表 8、9、11、12、および Abstract・Conclusions の数値をすべて再確認・再記載すること。これが改訂作業で最も事故が起きやすい箇所。
- 新規シード（2024, 31337）を追加する場合、既存 3 シードの値は変わらないはずなので、まず既存値が再現することを確認してから 2 シード分を追加実行する。
- A1 の B2（Minka）は `scikit-learn` の `PCA(n_components='mle')` で実装できる。B3 は `sklearn.decomposition` の Bayesian 系または自前実装。B4（ARD-VAE）は既存 `src/models/vae.py` に学習可能な事前分散を追加するだけで実装できる。
- dSprites は 737,280 枚あるため、他実験と条件を揃えて 20,000 / 3,000 / 3,000 にサブサンプルする（サンプリング方法を本文に明記）。

---

## 5. 原稿改訂の作業リスト（節単位）

| 節 | 作業 | 対応 |
|---|---|---|
| Abstract | 記号全廃、200 語以内に圧縮、新実験の結果を 1 文追加 | R1-1 |
| §1 冒頭 | 「貢献の種類」宣言パラグラフを新設 | R2-2(a) |
| §1.1 | 慣例値への出典追加、前方参照削除 | R1-2, R1-3 |
| §1.2 | 貢献リストから数値・前方参照・記号を削除。会議版差分表を新設 | R1-3, R1-4, R1-5, R2-1 |
| §1.3 | 記号を §3.1 に移し、言葉による説明のみ残す | R1-5 |
| §1.4 | 3 文に圧縮（詳細は §6.4 へ一本化） | R1-4 |
| §2 | graph-based 手法の段落を新設（Costa–Hero, Gałka et al.）。次元自動決定の段落に Minka を追加 | R1-6, R2-3 |
| §3.2 | Algorithm 1 の擬似コードを新設 | R3-7 |
| §3.2.2 | 「係数 2 は第一原理由来でない」を撤回し、Whitney 命題を参照する記述に書き換え | R2-2, R3-1 |
| §3.2.3 | 閾値一覧表を新設。V1/V2 の根拠記述を強化。防御的反復を削減 | R3-1, R3-6 |
| §4 | **Whitney 命題を新規追加**（命題 1 の系として）。活性化条件命題に Scope remark を追加。証明末尾 □ の脚注 | R2-2, R3-1, R3-4, R1-7 |
| §5.1 | 3 分割の記載、CUDA/環境情報の詳細、新データセットの設定 | R1-8, R1-9, R3-2 |
| §5.2 | `tab:e1` の書式を全面修正 | R2-4 |
| §5.3 | 5 シードに更新、学習曲線図を追加 | R1-9 |
| §5.4 | Fashion-MNIST を 5 シードに更新 | R1-9 |
| §5.5（新設） | **dSprites / KMNIST の検証結果** | R3-2 |
| §5.6（新設） | **ベースライン比較** | R2-3, R3-5 |
| §5.8 | Exp-Real-Geom の拡張結果（7 参照モデル × 3 推定器 × spread 指標） | R3-3 |
| §6.1 | ベースライン比較の考察に書き換え（現状の「慣例手法との違い」の議論を実測に置き換える） | R2-3, R3-5 |
| §6.4 | §1.4 から移した限界を統合。新データセットで解消した点と残る限界を区別 | R1-4, R3-2 |
| §7 | 新結果を反映。数値を A2/A3 の再実行値に更新 | 全般 |
| 全表 | 14 表の書式監査（桁数・指数・括弧・太字基準） | R2-4 |
| 全文 | "conditional" 24 → 8 回、防御的反復の削減、1 文 25 語目安への短縮 | R3-6, R2 英語 |
| 参考文献 | 全 37 件（＋新規 4〜6 件）に DOI を付与 | R1-10 |

**新規追加が見込まれる参考文献**:

- Costa, J.A.; Hero, A.O. IEEE Trans. Signal Process. **2004**, 52, 2210–2221. DOI 10.1109/TSP.2004.831130
- Gałka, Ł.; Karczmarek, P.; Tokovarov, M. IEEE Access **2022**, 10, 74175–74186. DOI 10.1109/ACCESS.2022.3190505
- Minka, T.P. Automatic choice of dimensionality for PCA. NIPS **2000**.
- Locatello, F. et al. Challenging Common Assumptions... ICML **2019**.
- Ha, D.; Schmidhuber, J. Recurrent World Models... NeurIPS **2018**.
- Matthey, L. et al. dSprites dataset **2017**（R3-2 で使用する場合）
- Clanuwat, T. et al. KMNIST **2018**（R3-2 で使用する場合）
- Whitney 埋め込み定理の典拠（既存の Hatcher [27] で代用できるか要確認。不足なら Lee, *Introduction to Smooth Manifolds* を追加）

---

## 6. Response to Reviewers の構成案

MDPI の標準様式（コメントごとに Response と該当箇所を記す形式）に従う。

**冒頭の総括レター**で、大きな変更 4 点を先に提示する。

> We thank the reviewers for their careful reading. The revision makes four substantive additions: (1) a new baseline comparison against seven established dimension-selection methods (Section 5.6); (2) a first-principles justification of the search-window multiplier via the Whitney embedding theorem, replacing what was previously stated as an empirical default (Proposition 2, Section 4.1); (3) validation on two further datasets, dSprites and KMNIST (Section 5.5); and (4) an explicit algorithm box and a table of every decision threshold with its rationale and invariance interval (Section 3.2). We also rebuilt all 14 tables under a single numeric format and reduced the manuscript's defensive repetition.

**個別回答で気をつけること**:

- **R2-1（新規性）**: 防御的にならず、差分表を示して事実で答える。「会議版と共通する図表は 0 点」「改訂で 2 系統の新実験を追加」を数字で言う。
- **R2-2（理論）**: 「新アーキテクチャを提案しないのは設計判断である」という主張と、「それでも理論は強化した（Whitney 命題）」「微分可能な自動決定機構は ARD-VAE として比較対象に含めた」という実質を必ずセットにする。主張だけでは通らない。
- **R1-9（Accuracy）**: 「AE の主タスクは再構成なので MSE が主指標」と述べたうえで、線形プローブ精度の曲線も追加したことを示す。求めを断らず、適切な形で満たす。
- **R1-6（MST 文献）**: 指摘文献が異常検知論文であることには触れず、graph-based 構造解析の文脈で引用し、かつ Costa–Hero の MST 推定器を**実験に追加した**ことを述べる。引用だけで済ませない。
- **R3-4（命題の限定）**: どの文の動詞をどう変えたかを、代表例 2〜3 件で具体的に示す。
- 全体として、**受け入れた指摘と、理由を付して受け入れなかった指摘を明確に分ける**。すべてに従うと主張が崩れる箇所（例: Step 3 を自然画像で解決せよという含意）は、限界として明記済みであることを丁寧に説明する。

---

## 7. 手順とファイル命名

1. `main_paper_NN43.tex` を `main_paper_NN42.tex` から作成し、日本語側を先に改訂する（プロジェクト規約: 改稿ごとに番号 +1、既存ファイルは上書きしない）。
   - **注意**: 現在 NN 側は NN42、MAKE 側は MAKE48 で番号が乖離している（MAKE47→48 は英語表現と書誌情報のみの修正で日本語版を起こさなかったため）。今回の改訂は内容変更が大きいので、日本語版も起こしたうえで、**MAKE 側は MAKE49 とする**。NN と MAKE の番号一致規約をどう扱うかは著者の判断が必要な点として残る（NN43 = MAKE49 とする対応表を `CLAUDE.md` に追記するのが現実的）。
2. 追加実験 A1–A7 を実行し、`results/tables/` と `results/figures/` に保存する。
3. `main_paper_MAKE49.tex` を作成し、§5 の表・図・数値を新結果に差し替える。
4. `pdflatex` ×3 でコンパイルし、`grep -n "^!" main_paper_MAKE49.log` でエラー 0、未解決引用 0 を確認する。
5. `MAKE49_Response_to_Reviewers.md`（および PDF）を作成する。
6. 変更箇所をハイライトした版（MDPI は差分表示を推奨）を用意する。
7. 補足資料を更新し、新実験のスクリプト・ログ・環境情報を追加する。

**投稿前チェック**:

- [ ] Abstract が 200 語以内かつ記号なし
- [ ] Introduction の前方参照が 5 か所以内
- [ ] 全 14 + 新規表の数値書式が統一されている
- [ ] Abstract・§5・§6・§7 の数値が A2/A3 の再実行結果と一致している
- [ ] 参考文献全件に DOI（または統一書式の安定 URL）がある
- [ ] "conditional" の出現数が 10 回以下
- [ ] Algorithm 1 と閾値一覧表が入っている
- [ ] Whitney 命題の証明（または標準文献への正確な参照）がある
- [ ] ベースライン比較表で本手法の優位を過大主張していない
- [ ] 新規引用がすべて本文中で参照されている

---

## 8. リスクと判断が必要な点

| 論点 | リスク | 推奨 |
|---|---|---|
| ベースライン比較で本手法が B6/B7 に精度で勝たない可能性 | 高い（そもそも探索コストで勝つ手法なので） | 事前に「勝敗表ではなく保証内容の整理表」として設計する。精度同等・コスト削減という結論で十分通る |
| 3 分割導入で既存の全数値が変わる | 中 | 分割変更前後の値を両方記録し、変化が小さいことを確認してから差し替える。大きく変わる場合は改訂レターで説明する |
| dSprites で Step 3 が pass しない可能性 | 中 | pass しなくても「失敗モード分類表で診断できた」という結果は論文の主張（診断プロトコルである）と整合する。むしろ Fashion-MNIST と同型の良い事例になる |
| Whitney 命題がトーラスの $m\ge5$ 観測と矛盾して見える | 低 | トーラスは $d_{\mathrm{emb}}=3 \le 4 = 2d$ で矛盾しない。$m\ge5$ が必要だったのは AU 打ち切りの挙動であり埋め込み可能性ではない、と本文で明確に区別する |
| 改訂期限（通常 10 日）に実験が間に合わない | 高 | **早期に編集部へ延長申請する**（3〜4 週間）。MDPI は理由を示せば通常応じる |
| R2 が再査読で依然として否定的 | 中 | ベースライン比較と Whitney 命題は R2 の指摘に正面から答えるものなので、対応を Response の冒頭に配置して見落とされないようにする |

---

## 9. 実施順序の推奨

```
第 1 週  A1 ベースライン実装・実行（P0、最重要）
        並行して §4 の Whitney 命題執筆と §3.2 の Algorithm 1 作成
        全表の書式監査（計算待ちの間にできる）
第 2 週  A2/A3（5 シード・3 分割）実行 → 数値差し替え
        A6 Exp-Real-Geom 拡張
        Introduction 改稿（R1-3, R1-4, R1-5）、Abstract 書き直し
第 3 週  A4/A5（dSprites, KMNIST）実行
        §5 新設 2 節の執筆、§6 の考察改稿
        参考文献の DOI 統一、"conditional" 削減、英文短縮
第 4 週  全体通読、数値整合確認、Response to Reviewers 執筆
        コンパイル確認、補足資料更新、投稿
```
