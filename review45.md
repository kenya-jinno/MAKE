# MAKE45改訂稿レビュー

対象原稿：

> **Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics**

対象ファイル：`main_paper_MAKE45.pdf`

---

## 総合判定

MAKE45は、前回までの主要な指摘をほぼ適切に反映しており、**内容面・英語面ともにMDPI *Machine Learning and Knowledge Extraction*（MAKE）への投稿を検討できる水準**に達している。

特に、以下の改善は重要である。

- 方法を、一般的な最適ボトルネック次元決定法ではなく、条件付きの探索窓・診断プロトコルとして限定した。
- \(\hat d_{\mathrm{ID}}\)を、唯一の正解や真のIDではなく、探索窓を与える参照値として位置づけた。
- AUを、内在次元・埋め込み次元の推定量ではなく、\(\beta\)、AU閾値、アーキテクチャ、学習条件に依存する診断量として明確にした。
- Proposition 1を理想化されたゼロ誤差・連続写像条件下の必要条件に限定した。
- Proposition 2を線形ガウス代理モデルに限定し、非線形VAEの定量定理ではないことを明記した。
- \(D_j(R_j)\)を residual variance contribution として再定義し、rate–distortion導出の用語を改善した。
- Conv-VAEにおける\(d^*(\beta)>512\)を直接観測値ではなく、代理モデルに基づく整合的解釈として扱った。
- β-strengtheningについて、安定したAU飽和を確認したとは述べず、活性化率低下とより強い切り捨ての証拠として限定した。
- 自然画像における\(m_{\mathrm{ref}}\approx2\hat d_{\mathrm{ID}}\)を、一般規則ではなくprovisional consistency regionとして整理した。
- Step 1の単一seed sweep、固定順序サブセット、V1/V2の運用上の性格、候補VAE潜在表現によるクロスチェックの位置づけを明示した。
- Abstractを、特定のMNIST grid・evaluation protocol、特定の\(\beta_{\max}=4\) Conv-VAE scheduleに限定した。

現段階で残る課題は、論文の中心的なアイデアではなく、**理論記述の最終整合、代理モデル解釈の明示、因果的・保証的に読める表現の軽微な弱化、図表・相互参照の最終確認**である。

---

## 1. 最優先の修正事項

### 1.1 Appendix A.1の「exact」と「negligible」の関係を最終整合化する

**該当箇所：Appendix A.1、約937–944行目**

現行原稿では、概ね次の趣旨が記されている。

> For the particular globally optimized linear-Gaussian solution considered here, the approximation is exact on truncated coordinates ... On active coordinates, the aggregated-posterior mismatch is assumed to be negligible ...

この改訂は大幅な改善である。ただし、Proposition 2の前提では、KL項を相互情報量で近似するため、集約事後分布と事前分布の不一致が無視できる範囲でのみ成立するとされている。したがって、座標種別ごとに、次の区別を明確に維持することが重要である。

- **切り捨て座標**：\(q_\phi(z_j\mid x)=p(z_j)\)であり、その座標の不一致はゼロ。
- **活性座標**：不一致は無視できると仮定するが、厳密なゼロは証明しない。

### 推奨修正文

```text
For the particular globally optimized linear-Gaussian solution considered here, the approximation is exact on truncated coordinates, for which \(q_\phi(z_j\mid x)=p(z_j)\). On active coordinates, the aggregated-posterior mismatch is assumed to be negligible; the PPCA-type variance-matching property motivates this assumption but does not establish exact vanishing. This assumption is not made for arbitrary linear or nonlinear VAEs.
```

### 日本語上の意味

> 本稿で扱う特定の線形ガウス大域最適解では、切り捨てられた座標については近似が厳密に成立する。一方、活性座標については、集約事後分布の不一致が無視できると仮定する。PPCA型の分散一致はこの仮定を動機づけるが、不一致が厳密にゼロであることを単独で証明するものではない。この仮定は任意の線形VAEや非線形VAEには適用しない。

### 修正理由

- 切り捨て座標でのexactnessと活性座標でのassumptionを混同しない。
- 理論の限定範囲を明確にする。
- Section 4.2とAppendix A.1の説明を完全に整合させる。

---

### 1.2 \(d^*_{\mathrm{eff}}(\beta)>512\) を定義するか、記号を削除する

**該当箇所：Section 5.6.1、約691–696行目**

原稿では、以下のような記述がある。

> an effective truncation threshold beyond the tested range (\(d^*_{\mathrm{eff}}(\beta)>512\))

Proposition 2で定義されているのは\(d^*(\beta)\)であり、\(d^*_{\mathrm{eff}}(\beta)\)は新しい記号である。非線形・周期的KLアニーリング実験のための解釈上の記号として導入するなら、そのことを明示する必要がある。

### 推奨修正文

```text
Under the surrogate interpretation, this pattern is consistent with an effective threshold \(d^*_{\mathrm{eff}}(\beta)\), defined here only as an interpretive quantity for the nonlinear experiment, lying beyond the tested range (\(d^*_{\mathrm{eff}}(\beta)>512\)).
```

または、より安全に記号を避ける。

```text
Under the surrogate interpretation, this pattern is consistent with an effective truncation threshold beyond the tested range; this threshold is not directly estimated from the nonlinear experiment.
```

### 日本語上の意味

> 代理モデルに基づく解釈では、この挙動は実効的な切り捨て閾値が検証範囲を超えていることと整合する。この閾値は非線形実験から直接推定した量ではない。

---

### 1.3 Conv-VAEの `clear AU = m pattern` をより正確にする

**該当箇所：Section 5.6.1、約693–696行目**

現行の趣旨：

> the clear AU = m pattern flags the diagnostic condition ...

しかし、\(m\le64\)では完全に\(AU=m\)である一方、\(m=256\)ではAU/mは94%であり、厳密には完全な一致ではない。そこで、観測量に忠実な表現にする方がよい。

### 推奨修正文

```text
The near-diagonal AU–\(m\) relationship flags the diagnostic condition “no substantial truncation observed within the tested range,” which is consistent with \(\beta\) being too small relative to the decoder capacity.
```

### 日本語上の意味

> AUと\(m\)がほぼ対角線上にある関係は、「検証範囲内で大きな切り捨てが観測されない」という診断状態を示しており、これはデコーダ容量に対して\(\beta\)が小さいことと整合する。

---

## 2. Abstractの確認

### 2.1 Abstractは大幅に改善されている

以下の修正は適切であり、現行のまま維持してよい。

- `verifying after training` への変更。
- 合成データはthree to five seeds、MNIST/Fashion-MNISTはthree seedsと明示。
- VAEのpre-specified Step-3 screening ruleと、AE/VAEに共通するreconstruction-error diagnosticsを区別。
- `for the specific MNIST grid and evaluation protocol used here` と評価条件を限定。
- 自然画像ではlatent-space TwoNN estimateが参照ボトルネック増大時に上昇したことを具体化。
- Conv-VAEについて、tested schedule with \(\beta_{\max}=4\)に限定。
- AUを一般的なID・embedding dimension推定量ではないと明記。

### 2.2 “the protocol produced reproducible diagnostics” の範囲

**該当箇所：Abstract、約12–18行目**

現行表現は許容範囲である。ただし、さらに厳密にする場合は、以下のように書ける。

```text
Under the tested settings, the protocol produced reproducible diagnostics on synthetic manifolds and fully connected real-data VAEs; it yielded a pass on MNIST and identified an over-pruning boundary case on Fashion-MNIST.
```

この変更は必須ではないが、再現性主張の適用範囲が明瞭になる。

---

## 3. Introduction・理論動機の表現

### 3.1 Section 1.2の `may be useful` は適切

**該当箇所：Section 1.2、約67–73行目**

次のように修正された点は適切である。

> may be useful when KL-induced truncation is effective—a condition that is itself a hypothesis to be checked by the Step-3 decision rule.

この表現は、VAEのgenerous \(m\)戦略が一般的な保証ではなく、Step 3で検証する仮説であることを明確にしている。維持してよい。

### 3.2 Section 4.2の `Sufficient Side` は少し強い

**該当箇所：Section 4.2見出し**

現行：

```text
4.2. Mechanism on the Sufficient Side: Water-Filling Surrogate Activation Condition
```

非線形VAEへの定量保証ではないことを考えると、`Sufficient Side` はやや保証的に読める。

### 推奨見出し

```text
4.2. Surplus-Dimension Truncation in a Water-Filling Surrogate
```

または、

```text
4.2. Interpretation on the Higher-Capacity Side: Water-Filling Surrogate Activation Condition
```

後者より、前者の方が簡潔で推奨される。

### 3.3 `a generous m is safe for VAEs` が残っている

**該当箇所：Section 4.2、Remark 1**

現行：

> “a generous \(m\) is safe for VAEs.”

本文全体では条件付きであることを強調しているため、この表現だけは少し強い。

### 推奨修正

```text
using a generous \(m\) can be useful when KL-induced truncation is effective
```

または、

```text
this motivates the conservative strategy of using a generous \(m\) and verifying the resulting active-unit count
```

---

## 4. Conv-VAE・自然画像の解釈

### 4.1 Table 2の第1行は改善済み

Table 2の第1行は、因果的断定ではなく、以下のように診断解釈として整理されている。

> Consistent with \(\beta\) being too small relative to decoder capacity, or with the effective truncation threshold lying beyond the tested range.

これは適切である。

### 4.2 β強化を「必要条件」と断定しない

**該当箇所：Section 5.6.1、約713–715行目**

現行：

> identifying “strengthening \(\beta\) according to data and architecture” as a necessary condition for applying the protocol to Conv-VAEs.

今回の実験では、\(\beta\)だけでなく、decoder capacity、KL schedule、データ構造なども影響する。したがって、`necessary condition` は少し強い。

### 推奨修正文

```text
suggesting that architecture- and data-dependent strengthening of \(\beta\) may be required before AU-based verification can be informative for Conv-VAEs
```

### 日本語上の意味

> Conv-VAEにおいてAUに基づく検証を有効にするには、アーキテクチャとデータに応じた\(\beta\)の強化が必要になる可能性が示唆された。

### 4.3 自然画像における \(d^*(\beta)\gg m\)

**該当箇所：Section 5.6.2、約755–758行目**

現行の文はすでに、

> Under the surrogate interpretation, this pattern would correspond to ...

と限定されており、適切である。変更不要である。

### 4.4 TwoNNとglobal metricに関する表現

**該当箇所：Section 5.6.2、約743–752行目**

現行：

> For TwoNN, which assumes a single global metric, ...

この表現は分かりやすいが、TwoNNの前提をやや単純化している。

### より慎重な表現

```text
When TwoNN is applied globally to a dataset with heterogeneous local geometry, such inhomogeneity can complicate the interpretation of neighbor-distance ratios.
```

これは必須ではないが、方法論的にはより正確である。

---

## 5. 実験設計・再現性

### 5.1 Table 6のキャプションは適切

Table 6では、Table 8が参照AEのmulti-seed sweepではなく、candidate-VAE latent cross-checkであることが明記されている。

> the three-seed candidate-VAE latent cross-check (diagnostic only; not used to set \(\hat d_{\mathrm{ID}}\)) is reported in Table 8.

この変更は適切であり、維持すべきである。

### 5.2 primary TwoNN referenceとMLE cross-checkの区別は適切

Section 5.3.1では、

> the pre-specified rounded-TwoNN rule yields the operational primary reference \(\hat d_{\mathrm{ID}}=10\), while the MLE values of 11.3–11.9 provide an order-of-magnitude cross-check

と明記されている。これは大きな改善であり、維持すべきである。

### 5.3 固定順序サブセットの記述

Section 5.1で固定順序サブセットの理由と限界が明記されている。より自然にする場合には、次のように具体化できる。

```text
We use the first \(n_{\mathrm{train}}\) and \(n_{\mathrm{test}}\) samples in the official splits to ensure exact reproducibility.
```

これは軽微な英文改善であり、必須ではない。

### 5.4 標準偏差の明示は適切

次の文は適切であり、維持する。

```text
Unless otherwise stated, all \(\pm\) values in multi-seed experiments denote standard deviations across seeds, not standard errors or confidence intervals.
```

### 5.5 補足資料の最終確認

Data Availability Statementの方針は適切である。実際に査読時の補足資料に以下が含まれることを確認する。

- 全seedの学習ログ。
- データサブセットのindex。
- 乱数seed。
- 設定ファイル。
- AU算出コード。
- TwoNN/MLE/DANCo/ESSの実装・バージョン。
- \(\beta\)アニーリングスケジュール。
- 図表生成スクリプト。
- Table 10のrun数算定スクリプト。
- Python、PyTorch、CUDA、GPU、ライブラリのバージョン。

---

## 6. 英語としての最終評価

### 6.1 全体評価

英語は、研究論文として十分に通用する水準である。重大な文法誤り、意味不明な文、過度に口語的な表現は見当たらない。とくに、理論上の限定と実験的限界を英語で慎重に表現できている。

### 6.2 残る改善余地

ネイティブ校閲レベルでは、次の軽微な改善が可能である。

- 長い文の分割。
- `conditional`、`diagnostic`、`tested`、`surrogate interpretation` の近接反復を軽減する。
- `safe`、`sufficient`、`necessary`のような保証的に読める語を全体で再確認する。
- `same order` を可能な範囲で `operational range` に統一する。
- `the manifold dimension` のような曖昧な総称を、`local intrinsic dimension` または `estimated local intrinsic dimensionality` に置き換える。

### 6.3 表現統一の候補

| 現行または類似表現 | 推奨表現 |
|---|---|
| `a generous \(m\) is safe for VAEs` | `using a generous \(m\) can be useful when KL-induced truncation is effective` |
| `necessary condition for applying the protocol` | `may be required before AU-based verification is informative` |
| `the activation condition predicts` | `the activation condition motivates the expectation` |
| `same order as` | `within the operational range defined relative to` |
| `the model is pushed into` | `the model is consistent with` |
| `deterministic transition` | `transition reproducible across the tested seeds` |

---

## 7. 投稿前チェックリスト

### 必須確認

- [ ] Appendix A.1で、切り捨て座標のexactnessと活性座標のnegligibility assumptionが明確に分離されていることを確認する。
- [ ] \(d^*_{\mathrm{eff}}(\beta)\)を使う場合、それが非線形実験から直接推定した量ではなく、代理モデルに基づく解釈量であることを定義する。
- [ ] Conv-VAEのAU挙動を、\(\beta\)不足の直接検出ではなく、`consistent with`による診断的解釈として維持する。
- [ ] `standard \(\beta\)` が残っていないことを確認する。
- [ ] `clear induced saturation` が残っていないことを確認する。
- [ ] Section 5.3.1でprimary TwoNN referenceとMLE cross-checkが明確に区別されていることを確認する。
- [ ] Table 6からTable 8への参照がcandidate-VAE latent cross-checkを意味することを確認する。
- [ ] Section 5.3.3の\([10,24]\)がMNIST固有の実用的構成であることを維持する。
- [ ] Table 2の診断解釈が因果的断定ではなく `consistent with` として書かれていることを確認する。
- [ ] PDF上で図表、式番号、表番号、参考文献番号、Appendix参照が整合していることを確認する。

### 強く推奨

- [ ] Section 4.2の `Sufficient Side` を、非保証的な表現へ変更する。
- [ ] Section 4.3の `conditions that may support conditional stability` を維持する。
- [ ] Section 4.2 Remark 1に残る `a generous \(m\) is safe for VAEs` をより慎重な表現に変更する。
- [ ] Conv-VAEに対する\(\beta\)強化を `necessary condition` ではなく `may be required` に変更する。
- [ ] TwoNNとsingle global metricに関する記述を、heterogeneous datasetへのglobal applicationという形に弱める。
- [ ] 固定順序データサブセットの制約をLimitationsで維持する。
- [ ] 補足資料の実内容とData Availability Statementが一致することを確認する。

### 最終確認

- [ ] 図中の\(\hat d_{\mathrm{ID}}\)が目視上正しく表示されていることを確認する。
- [ ] 図表の凡例・キャプション・本文で記号が一致していることを確認する。
- [ ] 査読時に匿名補足資料が実際にアクセス可能な状態で提出されることを確認する。
- [ ] 全seedログ、設定、データindex、環境情報、解析コードが補足資料に含まれることを確認する。

---

## 最終判断

MAKE45は、前回までに指摘された主要事項をほぼ適切に反映しており、**MDPI MAKEへの投稿を検討できる状態**にある。現段階で、論文の基本設計や実験構成を大きく変更する必要はない。

投稿前に優先して整えるべき残存点は、次の3点である。

1. **Appendix A.1における、線形ガウス近似のexactnessとactive-coordinate assumptionの最終整合。**
2. **\(d^*_{\mathrm{eff}}(\beta)>512\) の意味を、代理モデル解釈上の量として明示すること。**
3. **Conv-VAEに関する `safe`、`necessary`、`detects` など因果・保証的に読める表現を、`may be useful`、`may be required`、`consistent with`へ統一すること。**

これらを最終確認し、PDF上の数式・図表・相互参照・補足資料を点検すれば、内容面・英語面ともに投稿上の重大な障害は残っていないと判断できる。