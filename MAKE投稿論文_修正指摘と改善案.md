# MDPI *Machine Learning and Knowledge Extraction* 投稿論文：修正指摘と改善案

対象論文：

> **Choosing the Bottleneck Dimension of Autoencoders and Variational Autoencoders from the Intrinsic Dimension of the Data: An ID-Guided Protocol with Active-Unit Verification**

本メモは、投稿予定原稿について、理論的主張、実験設計、結果解釈、再現性、表現・構成の観点から、投稿前に検討すべき修正点を整理したものである。最も重要な方針は、論文全体を「ボトルネック次元を一般的に自動決定する方法」ではなく、**内在次元推定値を探索窓の基準とし、AUと再構成誤差で適用可否を診断する条件付きプロトコル**として一貫して記述することである。

---

## 1. 総合評価

本稿の中心的な着想は明確である。すなわち、推定した内在次元 \(\hat d_{\mathrm{ID}}\) をAE/VAEのボトルネック次元 \(m\) の候補範囲に利用し、VAEについてはActive Units（AU）を用いて、余剰潜在座標が実際に切り捨てられたかを学習後に診断する、という **estimate–set–verify** 型のプロトコルを提案している。

特に、以下は強みである。

- 合成多様体、MNIST、Fashion-MNIST、Conv-VAE、自然画像まで含め、成立例だけでなく適用限界も示そうとしている。
- \(d_{\mathrm{ID}}\)、\(d_{\mathrm{emb}}\)、\(\hat d_{\mathrm{ID}}\)、AUを区別しようとしている。
- 線形ガウス代理モデルでの水満たし型解釈と、実験上のAU診断を明示的に接続している。
- Fashion-MNISTやConv-VAEにおける失敗・境界事例を隠さず、診断結果として報告している。
- 再現性、seed差、推定器間比較、サブサンプリング区間、探索コストなどを意識している。

一方で、投稿前には少なくとも次の点を修正・補強することを推奨する。

1. 理論上の主張をさらに限定する。
2. \(\hat d_{\mathrm{ID}}\)、\(d_{\mathrm{ID}}\)、\(d_{\mathrm{emb}}\)、AUの関係を厳密に整理する。
3. Step 1、Step 3、参照AE、候補VAEの役割を明確に分離する。
4. MNISTでの性能・探索コスト削減を一般化しない。
5. Conv-VAEでの固定 \(\beta\) 理論と周期的KLアニーリング実験の関係を明示する。
6. 自然画像に対する説明を、検証済みの結論ではなく未検証仮説として扱う。
7. seed数、データ分割、補足資料、計算量比較の条件をより明示する。

---

## 2. 最優先の修正事項

### 2.1 タイトルを「一般的な決定法」と誤解されない形にする

**該当箇所：タイトル**

現行タイトル：

> Choosing the Bottleneck Dimension of Autoencoders and Variational Autoencoders from the Intrinsic Dimension of the Data: An ID-Guided Protocol with Active-Unit Verification

`Choosing` は、内在次元から最適なボトルネック次元を直接決定できる方法であるかのような印象を与える。しかし本文の立場は、単一の最適値を出す方法ではなく、探索窓の設定と診断を行う条件付きプロトコルである。

#### 推奨タイトル案

```text
An Intrinsic-Dimension-Guided Protocol for Selecting Autoencoder Bottleneck Dimensions
```

または、AUを強調する場合：

```text
Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics
```

#### 修正意図

- `Selecting` や `Protocol`、`Diagnostics` を用いて、単一値を保証する方法ではないことを示す。
- タイトルを短くし、方法論的な主張を明瞭にする。
- 本文中の「conditional, diagnostic workflow」という立場と整合させる。

---

### 2.2 AbstractにおけるVAEの設計規則を限定する

**該当箇所：Abstract 6–8行目**

> AE: \(m\in[\hat d_{\mathrm{ID}},2\hat d_{\mathrm{ID}}]\); VAE: \(m\ge2\hat d_{\mathrm{ID}}\)

VAEについて \(m\ge2\hat d_{\mathrm{ID}}\) と書くだけでは上限がなく、実際にどのように有限の候補を選ぶのかが不明瞭である。また、本文のConv-VAE結果は、単に \(m\) を大きくするだけではAUの切り捨てが生じないことを示している。

#### 修正文案

```text
For AEs, the protocol recommends searching within a finite window centered on \(\hat d_{\mathrm{ID}}\), whereas for VAEs it recommends selecting a sufficiently generous candidate dimension \(m\ge 2\hat d_{\mathrm{ID}}\) and verifying post-training whether surplus coordinates are inactive.
```

#### 日本語上の意味

> AEでは \(\hat d_{\mathrm{ID}}\) を中心とする有限の探索範囲を用いる。一方VAEでは、\(m\ge2\hat d_{\mathrm{ID}}\) を目安として十分に大きな候補次元を設定し、学習後に余剰座標が不活性化しているかを検証する。

#### 追加すべき注意

- \(m\ge2\hat d_{\mathrm{ID}}\) は十分条件でも最適化原理でもない。
- \(\beta\)、デコーダ容量、学習スケジュール、データの性質によってAUの切り捨ては成立しない。
- VAE側では、候補点を有限の事前規定グリッドから選ぶ実装手順を明示する。

---

### 2.3 中心主張を「一般解」ではなく「条件付き診断プロトコル」に統一する

**該当箇所：Abstract、Introduction、Conclusions 全般**

本文には適切な注意書きが多い一方、タイトル、要旨、中心主張、結論の一部には、「内在次元からボトルネックを選べる」と一般化して読める箇所がある。

#### 推奨する一貫した立場

> \(\hat d_{\mathrm{ID}}\) は、ボトルネック次元の唯一の正解を与えるものではなく、候補範囲の中心または下限側の参照値を与える。VAEにおけるAUは内在次元推定量ではなく、特定の \(\beta\)、閾値、アーキテクチャ、学習条件の下で利用された座標数を示す診断量である。

#### 推奨するキーフレーズ

- `conditional diagnostic protocol`
- `search-window reference`
- `model- and training-dependent diagnostic`
- `not a general estimator of intrinsic or embedding dimension`
- `not a single-number selection algorithm`

---

## 3. 理論的主張に関する修正

### 3.1 Proposition 1における \(d_{\mathrm{emb}}\) と \(d_{\mathrm{ID}}\) の区別

**該当箇所：Section 1.3、Section 3.2.2、Section 4.1、Conclusion**

原稿では、局所内在次元 \(d_{\mathrm{ID}}\) と最小埋め込み次元 \(d_{\mathrm{emb}}\) を区別している点はよい。しかし実データでは、TwoNN/MLEが推定するのは \(d_{\mathrm{emb}}\) ではなく、局所的・表現依存的な内在次元推定値である。

特に次の関係は慎重に扱う必要がある。

```text
\(d_{\mathrm{emb}} \ge d_{\mathrm{ID}}\)
```

理想化された滑らかな多様体の真の次元 \(d\) に対しては \(d_{\mathrm{emb}}\ge d\) と書けるが、実データの \(\hat d_{\mathrm{ID}}\) との関係として直接使うべきではない。

#### 推奨修正文

```text
For an ideal \(d\)-dimensional smooth manifold, the minimum Euclidean embedding dimension satisfies \(d_{\mathrm{emb}}\ge d\). In real datasets, however, \(\hat d_{\mathrm{ID}}\) is an estimator of local intrinsic dimensionality rather than a direct estimator of \(d_{\mathrm{emb}}\). Therefore, the relation between intrinsic and embedding dimensions should be interpreted as a geometric motivation for adding a margin, not as an empirically verifiable inequality involving \(\hat d_{\mathrm{ID}}\).
```

#### 日本語上の意味

> 理想化された \(d\) 次元滑らか多様体では、最小ユークリッド埋め込み次元は \(d_{\mathrm{emb}}\ge d\) を満たす。しかし実データにおける \(\hat d_{\mathrm{ID}}\) は局所内在次元の推定値であり、\(d_{\mathrm{emb}}\) の直接推定値ではない。したがって、両者の関係はマージンを設ける幾何学的な動機として解釈すべきであり、\(\hat d_{\mathrm{ID}}\) を含む経験的に検証可能な不等式として扱うべきではない。

---

### 3.2 “self-intersection-free reconstruction” を理想条件に限定する

**該当箇所：Abstract 8–9行目、Section 1.2、Section 4.1**

現行表現：

> required for self-intersection-free reconstruction

これは、連続写像、コンパクト滑らか多様体、ゼロ再構成誤差という理想条件では妥当である。しかし実際のAE/VAEでは、有限サンプル、有限再構成誤差、ノイズ、最適化不完全性がある。

#### 推奨修正

```text
providing a necessary condition for zero-error, self-intersection-free reconstruction under an idealized continuous-manifold assumption
```

または短く：

```text
required for zero-error continuous reconstruction of an idealized manifold
```

#### 修正意図

- 命題を実ニューラルネットワークの有限誤差再構成に直接適用していないことを明確にする。
- トーラス実験を「命題の直接的実証」ではなく、「理想化された下限が経験的に現れる事例」と位置づける。

---

### 3.3 Proposition 2を非線形VAEに一般化しない

**該当箇所：Section 4.2、Appendix A.1、Abstract、Conclusion**

活性化条件

\[
\lambda_j > \beta\tau^2
\]

は、線形エンコーダ・線形デコーダ・ガウス尤度・特定のrate–distortion近似という強い仮定の下での結果である。非線形VAEにおいて、経験的AUの座標ごとの活性化を定量的に予測する定理ではない。

#### 追加すべき明確な文章

```text
The activation condition \(\lambda_j>\beta\tau^2\) is exact only for the specified linear-Gaussian surrogate. In nonlinear VAEs, it should not be interpreted as a coordinate-wise activation theorem or as a quantitative predictor of empirical AU. It only motivates the qualitative expectation that increasing \(\beta\), or reducing decoder capacity, can promote latent truncation.
```

#### 日本語上の意味

> \(\lambda_j>\beta\tau^2\) という活性化条件は、本稿で定義した線形ガウス代理モデルにおいてのみ厳密である。非線形VAEにおいて、この条件を座標ごとの活性化定理、あるいは経験的AUの定量的予測式として解釈してはならない。これは、\(\beta\) を大きくすることやデコーダ容量を抑えることが潜在座標の切り捨てを促進し得る、という定性的な動機付けを与えるにとどまる。

---

### 3.4 ELBOと相互情報量の置換の説明を慎重にする

**該当箇所：Appendix A.1、810–838行目**

原稿は、

\[
\mathbb{E}_x[\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))]
=I(x;z)+\mathrm{KL}(q_\phi(z)\Vert p(z))
\]

を示し、集約事後分布と事前分布の不一致を落とす近似を明記している。この点はよい。ただし、「線形ガウス設定では大域最適で近似が正確」という記述は、より条件を限定した方がよい。

#### 推奨修正文

```text
Under the particular globally optimized linear-Gaussian solution considered here, the aggregated-posterior mismatch is zero for truncated coordinates and is assumed negligible for active coordinates. This property should not be generalized to arbitrary linear or nonlinear VAEs.
```

#### 修正意図

- 任意の線形VAEで \(q(z)=p(z)\) が成立するような印象を避ける。
- 非線形VAEに対する議論は定性的な代理解釈にとどまることを明確にする。

---

### 3.5 Proposition 3は「proof sketch」ではなく仮説的な安定性の説明として扱う

**該当箇所：Section 4.3、Appendix A.2**

Proposition 3は、局所bi-Lipschitz性と近傍順位保存を仮定し、TwoNN推定値の安定性を説明しようとしている。しかし、本文に具体的な定数、有限標本誤差、推定量の連続性条件、EMDから推定量偏差への厳密な写像が提示されていない。

`Proposition` と呼ぶと、査読者によっては厳密な証明を期待される可能性がある。

#### 推奨

以下のいずれかを検討する。

1. 厳密な仮定と定理・証明を与える。
2. 現状を維持するなら、`Proposition 3` を `Heuristic Stability Argument` または `Conditional-Stability Rationale` に変更する。

#### 推奨見出し案

```text
4.3 Conditional-Stability Rationale for Latent-Space TwoNN Estimation
```

#### 推奨文

```text
This section provides a heuristic conditional-stability rationale rather than a finite-sample theorem for TwoNN.
```

---

## 4. \(\hat d_{\mathrm{ID}}\)、AU、\(d_{\mathrm{emb}}\) の関係を明確化する

### 4.1 AUを“effective dimension”と呼びすぎない

**該当箇所：Table 1**

現行表現：

> AUsat: Read-out of effective dimension (VAE)

AUは \(\beta\)、AU閾値 \(\delta\)、潜在座標のパラメータ化、デコーダ容量、学習過程に依存する。したがって、単独で「実効次元」と呼ぶと、幾何学的または統計的な次元推定量のように読める。

#### 推奨表記

```text
AUsat: Saturated number of empirically active coordinates under a specified \(\beta\), threshold \(\delta\), and architecture
```

#### 日本語上の意味

> 指定された \(\beta\)、AU閾値 \(\delta\)、およびアーキテクチャの下で、経験的に活性な座標数が飽和した値。

---

### 4.2 AUの座標依存性を明記する

**該当箇所：Section 3.1、Section 2のAU説明、Limitations**

AUはposterior meanの各座標の分散に基づく。したがって、潜在空間の任意の回転や再パラメータ化に不変ではない。

#### 追加すべき文章

```text
AU is coordinate-dependent and is not invariant under arbitrary rotations or reparameterizations of the latent space. Its interpretation therefore depends on the chosen prior, parameterization, training objective, and architecture.
```

#### 日本語上の意味

> AUは座標依存量であり、潜在空間の任意の回転や再パラメータ化に対して不変ではない。その解釈は、採用した事前分布、パラメータ化、学習目的、アーキテクチャに依存する。

これは、AUをID推定量として誤解されることを防ぐ上で重要である。

---

### 4.3 トーラスにおける \(AUsat=d_{\mathrm{emb}}\) を一般的対応として読ませない

**該当箇所：Section 5.2、409–456行目；Appendix C.4；Conclusion**

トーラスで \(AUsat=3=d_{\mathrm{emb}}\) が得られたことは興味深い。しかし、結論の表現がやや強い。

#### 現行趣旨

> \(AUsat=d_{\mathrm{true}}\) or \(d_{\mathrm{emb}}\) under sufficient margin

#### 推奨修正文

```text
In the synthetic experiments, the saturated AU coincided with \(d_{\mathrm{true}}\) or \(d_{\mathrm{emb}}\) under the tested optimization and regularization settings. This coincidence is empirical and should not be interpreted as a general estimator property.
```

#### 日本語上の意味

> 合成データ実験では、検証した最適化条件および正則化条件の下で、飽和AUが \(d_{\mathrm{true}}\) または \(d_{\mathrm{emb}}\) と一致した。ただし、この一致は経験的な観測であり、AUが一般にこれらの次元を推定する性質を意味しない。

---

## 5. Step 1・Step 3・参照表現の実験設計

### 5.1 Step 1で使う参照AEとStep 3で評価するVAEを明確に分離する

**該当箇所：Section 3.2.1、Section 5.3.1、Table 6、Table 8**

MNISTでは、参照AE潜在表現によるTwoNN/MLE推定と、VAE潜在表現上のTwoNN値が近接して報告されている。そのため、読者が「Step 1の \(\hat d_{\mathrm{ID}}\) はAE由来なのか、VAE由来なのか」を混同する可能性がある。

#### 追加すべき文章

```text
The reference representation used for Step 1 must be trained independently of the candidate VAE models used in Step 3. The VAE latents reported in Section 5.3 are diagnostic cross-checks and are not used to define \(\hat d_{\mathrm{ID}}\).
```

#### 日本語上の意味

> Step 1で用いる参照表現は、Step 3で評価する候補VAEとは独立に学習する。Section 5.3で示すVAE潜在表現のTwoNN値は診断的なクロスチェックであり、\(\hat d_{\mathrm{ID}}\) の定義には用いない。

#### 表の修正

- Table 6のキャプションに「reference AE latents used for Step 1」と明記する。
- Table 8のキャプションに「VAE latents; diagnostic only; not used to set \(\hat d_{\mathrm{ID}}\)」と明記する。
- 図中の `dID = 10` を、`Primary Step-1 TwoNN reference \(\hat d_{\mathrm{ID}}=10\)` に変更する。

---

### 5.2 Step 1の単一seed sweepと3seed検証を区別する

**該当箇所：Section 1.2、Section 5.3.1、Table 6、Table 7、Table 8、Conclusion**

Table 6の \(m_{\mathrm{ref}}\)-sweep は単一seedである。一方、本文はMNIST Step 1の安定性を3seedで支持している。現行の書き方では、参照AEの完全な \(m_{\mathrm{ref}}\)-sweepが3seedで実施されたように読める可能性がある。

#### 推奨追記

```text
The \(m_{\mathrm{ref}}\)-convergence sweep itself was performed with one seed, whereas uncertainty was assessed by subsampling and the final latent estimate was reproduced across three VAE seeds. A full multi-seed reference-AE sweep over all \(m_{\mathrm{ref}}\) values was not performed.
```

#### 日本語上の意味

> \(m_{\mathrm{ref}}\) に対する収束sweep自体は単一seedで実施した。一方、不確実性はサブサンプリングで評価し、最終的な潜在表現の推定値は3つのVAE seedで再現した。すべての \(m_{\mathrm{ref}}\) に対する参照AEの完全な複数seed sweepは実施していない。

これはLimitationsにも簡潔に書くべきである。

---

### 5.3 サブサンプリング95%区間の意味を限定する

**該当箇所：Section 5.3.2、Table 7**

サブサンプリング区間は、固定された学習済み表現における標本変動を示す。初期値、学習確率性、データ分割、モデル構造、推定器バイアスは含まない。

#### 追加すべき文章

```text
These intervals quantify subsampling variability conditional on the trained reference representation. They do not include variation due to model initialization, training stochasticity, dataset resampling at the population level, or estimator bias.
```

#### 日本語上の意味

> これらの区間は、学習済み参照表現を固定した条件でのサブサンプリング変動を示すものである。モデル初期値、学習確率性、母集団からのデータ再標本化、推定器バイアスによる変動は含まれない。

---

### 5.4 “stable” を一般化しない

**該当箇所：Abstract、Table 3、Section 5.5、Section 6.2、Conclusion**

現行の「stable only through reference representations with \(m\gg\hat d_{\mathrm{ID}}\)」という表現は、今回のデータセット・アーキテクチャ・推定器の範囲を超えて一般化しているように読める。

#### 推奨修正文

```text
Under the datasets, architectures, sample sizes, and estimators tested here, the estimate became numerically stable only after using a reference bottleneck substantially larger than the working estimate.
```

#### 日本語上の意味

> 本研究で検証したデータセット、アーキテクチャ、サンプル数、推定器の範囲では、参照ボトルネックを作業推定値より十分に大きくした後にのみ、推定値の数値的安定性が得られた。

---

## 6. V1/V2判定規則の位置づけ

### 6.1 V1/V2を統計的検定のように見せない

**該当箇所：Section 3.2.3、242–261行目**

現行のV1/V2は、プロトコルの実用的な判定規則として有用である。一方、

\[
\mathrm{V1}: \frac{AU(m_{\mathrm{ver}})}{m_{\mathrm{ver}}}\le0.9
\]

\[
\mathrm{V2}: \hat d_{\mathrm{ID}}\le AU(m_{\mathrm{ver}})\le3\hat d_{\mathrm{ID}}
\]

における0.9、1、3の係数は、今回の実験を分離するための運用上の値であり、一般に校正済みの閾値ではない。

#### 追加すべき文章

```text
V1 and V2 are operational screening criteria introduced for this study. They are not hypothesis tests, confidence rules, or universally calibrated acceptance criteria.
```

#### 日本語上の意味

> V1およびV2は本研究のために導入した運用上のスクリーニング基準であり、仮説検定、信頼区間に基づく判定、または普遍的に校正された合格基準ではない。

### 6.2 V1/V2のpassが意味しないことを明示する

以下も明記するとよい。

- V1のpassは、すべての余剰座標が完全に消失したことを意味しない。
- V2のpassは、AUと \(d_{\mathrm{ID}}\) または \(d_{\mathrm{emb}}\) が一致することを意味しない。
- V1/V2のpassは、再構成性能、生成品質、下流性能、識別可能性を保証しない。
- V1/V2はMSE、ELBO、下流タスク、可視化などと併用する診断指標である。

#### 推奨文

```text
A V1/V2 pass indicates only that the observed AU is compressed and of the same order as the Step-1 reference under the specified conditions. It does not establish equality between AU and intrinsic or embedding dimension, nor does it guarantee reconstruction quality, generative quality, or downstream utility.
```

---

## 7. MNISTの性能・探索コスト主張

### 7.1 “grid-best accuracy” を限定的に述べる

**該当箇所：Abstract 13–16行目、Section 5.4、Conclusion**

現行の表現：

> attaining the grid-best downstream linear-probe accuracy inside the window

この結果は、MNIST、特定のグリッド、線形ロジスティック回帰、固定サブセット、AEでは単一seed、VAEでは3seedという限定条件に基づく。

#### 推奨修正文

```text
On the MNIST grid and linear-probe evaluation used in this study, the ID-guided window contained the best observed probe accuracy.
```

#### 日本語上の意味

> 本研究で用いたMNISTの探索グリッドおよび線形プローブ評価に限れば、ID-guided探索窓の中に、観測された最高のプローブ精度が含まれていた。

`grid-best` よりも `best observed` を用いる方が、偶然の観測値を一般原理と見なさない表現になる。

---

### 7.2 “nearly identical performance” を定量化して限定する

**該当箇所：Section 5.4、VAE結果**

VAEの最高平均精度は \(89.1\pm0.4\%\)（\(m=12\)）、検証点 \(m=20\) は \(88.2\pm0.4\%\) であり、差は0.9ポイントである。3seedでは、統計的同等性を主張するには十分ではない。

#### 推奨修正文

```text
The verification point was within 0.9 percentage points of the best observed mean accuracy, although the three-seed sample is too small to support a formal equivalence claim.
```

#### 日本語上の意味

> 検証点の平均精度は、観測された最高平均精度から0.9ポイント以内であった。ただし、3seedのみであるため、両者が統計的に同等であるという正式な主張はできない。

---

### 7.3 計算削減率を表にする

**該当箇所：Section 5.4、567–577行目**

計算削減率は複数の条件に依存するため、文章だけでは読みにくい。以下のような表を追加することを推奨する。

| Path | Full-grid runs | ID-guided runs | Step-1 seed count | Reduction |
|---|---:|---:|---:|---:|
| AE, single-seed Step 1 | 33 | 12 | 1 | 64% |
| VAE, single-seed Step 1 | 33 | 6 | 1 | 82% |
| VAE + MLE-shifted verification | 33 | 9 | 1 | 73% |
| AE, three-seed Step 1 | 33 | 18 | 3 | 45% |
| VAE, three-seed Step 1 | 33 | 12 | 3 | 64% |

#### 推奨キャプション

```text
Training-run accounting for the specific MNIST grid, epoch count, seed design, and verification procedure used in this study.
```

また、削減率はGPU時間、推定器計算時間、解析時間を含まない「学習run数」の比較であることを明示する。

---

### 7.4 AE側の線形プローブは単一seedであることを目立たせる

**該当箇所：Table 10、Section 5.4**

AEのprobe accuracyは単一seed、VAEのprobe accuracyは3seedである。この非対称性は重要である。

#### 追加すべきTable 10キャプション文

```text
Because AE results are based on a single reference-AE seed, the AE probe accuracies should be interpreted descriptively rather than as multi-seed performance estimates.
```

#### 日本語上の意味

> AEの結果は単一の参照AE seedに基づくため、AEのプローブ精度は複数seedによる性能推定値ではなく、記述的な結果として解釈すべきである。

---

## 8. Conv-VAEと自然画像に関する修正

### 8.1 固定 \(\beta\) の理論と周期的KLアニーリングを区別する

**該当箇所：Section 5.6、Appendix C.1**

理論部のProposition 2は固定 \(\beta\) を仮定する。一方、Conv-VAEの実験ではKL cyclic annealingと \(\beta_{\max}\) を用いる。

この両者を直接同一視すると、理論と実験の接続が過度に単純化される。

#### 追加すべき文章

```text
The surrogate analysis assumes a fixed \(\beta\), whereas the convolutional experiments use cyclical KL annealing. Therefore, \(\beta_{\max}\) is reported as a training-schedule parameter and should not be interpreted as the fixed-\(\beta\) value appearing in Proposition 2.
```

#### 日本語上の意味

> 代理モデルの解析は固定 \(\beta\) を仮定しているが、畳み込みVAEの実験では周期的KLアニーリングを用いている。したがって、\(\beta_{\max}\) は学習スケジュールのパラメータとして報告しており、命題2に現れる固定 \(\beta\) と同一視してはならない。

さらに、補足資料または本文で、アニーリング周期、warm-up、最小 \(\beta\)、周期数、スケジュール関数を図または表で明示する。

---

### 8.2 自然画像に対する説明を「未検証仮説」と明示する

**該当箇所：Section 5.6.2、634–657行目**

以下の説明は興味深いが、直接検証されていない。

- manifold entanglement
- high-frequency texture and effective noise variance
- hierarchical latent structure

#### 追加すべき導入文

```text
The following mechanisms are hypotheses consistent with the observed pattern, not experimentally distinguished explanations.
```

#### 日本語上の意味

> 以下の機構は、観測された傾向と整合する仮説であり、本研究の実験によって相互に識別された説明ではない。

また、`manifold entanglement` については、少なくとも以下のいずれかとして操作的に定義する。

- 異なるクラスや領域における局所内在次元の不均一性。
- 複数の部分多様体の合併としての分布構造。
- 単一グローバル距離に基づくTwoNN推定との不整合。

---

### 8.3 SVHNのraw-input推定値を真値のように扱わない

**該当箇所：Section 5.6.2、Table 12**

SVHNでは、raw-input TwoNNが10.09、潜在空間TwoNNが27.08であり、大きな差がある。これを潜在空間でのID inflationと解釈するには、入力空間のTwoNN自体が信頼できることが必要である。

#### 追加すべき文章

```text
The discrepancy between raw-input and latent-space estimates on SVHN should not by itself be interpreted as latent-space inflation, because the raw-input estimate may also be affected by ambient-dimensional concentration and heterogeneous local geometry.
```

#### 日本語上の意味

> SVHNにおける入力空間と潜在空間の推定値の差だけから、潜在空間での過大推定と結論づけるべきではない。入力空間側の推定値も、周囲次元による距離集中や不均一な局所幾何の影響を受けている可能性がある。

---

### 8.4 \(\tau_{\mathrm{eff}}^2\) の後付け解釈をさらに弱める

**該当箇所：Section 5.6.1、607–616行目**

Conv-VAEにおける \(\tau_{\mathrm{eff}}^2\) の説明は、理論的に推定した量ではなく、観測AUに合わせた後付けの解釈である。本文にもheuristicとあるが、要旨・結論にこの種の解釈が強く出ないよう注意する。

#### 推奨文

```text
The discussion of \(\tau_{\mathrm{eff}}^2\) is a post-hoc heuristic interpretation intended to connect the observed behavior with the linear surrogate; it is not an independently estimated physical or probabilistic parameter of the nonlinear decoder.
```

---

## 9. 再現性・データ分割・補足資料

### 9.1 固定順序でのサブセット抽出を説明する

**該当箇所：Section 5.1、376–381行目**

現行記述：

> Train/test subsets use the first \(n_{\mathrm{train}}/n_{\mathrm{test}}\) samples of the official splits in fixed order.

完全再現性のためには理解できるが、順序依存性を持つ可能性がある。

#### 追加すべき文章

```text
We use the first samples of the official splits to ensure exact reproducibility. Because this choice may introduce order-specific effects, the reported results should not be interpreted as estimates averaged over independent data partitions.
```

#### 日本語上の意味

> 完全な再現性を確保するため、公式分割の先頭サンプルを用いた。ただし、この選択はデータ順序固有の影響を導入する可能性があるため、結果を独立した複数データ分割にわたる平均性能推定値として解釈してはならない。

可能であれば、補足資料で異なるランダムサブセットまたはデータ分割seedによる確認を追加する。

---

### 9.2 データセット全体ではなく固定サブセットを用いる理由を明記する

MNIST/Fashion-MNISTでは、標準の全データではなく、20,000 train / 3,000 testを用いている。

#### 追加すべき文章

```text
We intentionally use fixed subsets rather than the complete official training set to control computational cost and ensure consistent comparisons across experiments.
```

#### 日本語上の意味

> 計算コストを抑え、実験間で同一条件を保つため、公式学習データ全体ではなく固定されたサブセットを用いた。

---

### 9.3 Supplementary Materialsを査読時に利用可能にする

**該当箇所：Section 5.1、ConclusionのSupplementary Materials、Data Availability Statement**

現行では、受理後にGitHub/Zenodoへ公開するとしている。可能であれば、査読時点で匿名化済みの補足資料を投稿システムに添付する方がよい。

#### 最低限含めるべきもの

- 全実験スクリプト。
- 全seedのJSONログ。
- 使用データサンプルのインデックス。
- モデル定義と設定ファイル。
- optimizer、学習率、batch size、epoch数。
- KLアニーリングスケジュール。
- AU算出コード。
- TwoNN、MLE、DANCo、ESSの実装・バージョン。
- PyTorch、CUDA、GPU、Python、ライブラリのバージョン。
- 各表・図の生成スクリプト。

#### 推奨表現

```text
Anonymous code, configurations, data indices, seed-specific logs, and figure-generation scripts are provided as supplementary material for review.
```

---

## 10. 関連研究の補強

### 10.1 内在次元推定器の限界に関する文献を追加する

**該当箇所：Section 2、Section 6.2**

本稿の中核は内在次元推定であるため、TwoNN/MLEの原典だけでなく、以下に関する関連研究を補強するとよい。

- 有限標本バイアス。
- 非一様サンプリング。
- ノイズ・曲率の影響。
- 高次元空間での近傍距離集中。
- 異なる局所次元を持つ部分多様体の混合。
- 深層表現でのID推定の表現依存性。

本文ではこれらを実験的に認識しているため、関連研究節でも理論的・方法論的な位置づけを補うと説得力が高まる。

---

### 10.2 AUの定義・限界を関連研究で明示する

**該当箇所：Section 2、Section 3.1**

次の事項を明記する。

- AUはposterior meanのデータ分散に基づく。
- posterior varianceやKL寄与の座標別分解と同一ではない。
- AU閾値 \(\delta\) は潜在スケールに依存し得る。
- AUは座標依存量である。
- AUを内在次元推定量として用いるわけではない。

これにより、「AUを何として測っているのか」がより明確になる。

---

## 11. 表・図・キャプションの改善

### 11.1 Table 5を簡潔にする

**該当箇所：Table 5**

表内に長い解釈注記が混在しているため、数値の読み取りが難しい。数値を表に残し、`saturation = ...`、`over-pruned` などは脚注または本文に移す。

#### 推奨構造

| Dataset | \(m\) | AE coordinate usage | VAE AU |
|---|---:|---:|---:|
| Swiss Roll | 2 | 2 | \(2.0\pm0.0\) |
| Swiss Roll | 4 | 4 | \(2.0\pm0.0\) |
| Torus | 4 | 4 | \(2.0\pm0.0\) |
| Torus | 6 | 6 | \(3.0\pm0.0\) |

#### 脚注文案

```text
On the Torus, AU = 2 at \(m=4\) is an over-pruned boundary case; AU = 3 becomes stable from \(m=5\) in the fine-grid experiment.
```

---

### 11.2 Figure 2を分割または軸表示を改善する

**該当箇所：Figure 2**

MSE、AU、TwoNN IDを同じ図に載せているため、軸や単位を取り違えやすい。

#### 推奨

- MSEとAU/TwoNNを別パネルに分ける。
- 各曲線について平均値かseed平均かをキャプションで明記する。
- AEとVAEでseed数が異なることをキャプションに明記する。
- \(d_{\mathrm{ID}}\) ではなく、`Step-1 reference \(\hat d_{\mathrm{ID}}\)` と表示する。

---

### 11.3 Figure 3のラベルを \(\hat d_{\mathrm{ID}}\) に統一する

**該当箇所：Figure 3**

図中の `dID = 10` は真の内在次元のように見える。

#### 修正案

```text
Primary Step-1 TwoNN reference \(\hat d_{\mathrm{ID}}=10\)
```

または、短く：

```text
Rounded Step-1 reference \(\hat d_{\mathrm{ID}}=10\)
```

---

### 11.4 Figure 4で等長性の一般的不重要性を示唆しない

**該当箇所：Figure 4、Section 5.5**

`regardless of isometry` は、等長性が一般に無関係であるように読める可能性がある。

#### 推奨キャプション文

```text
TwoNN convergence was similar across the tested reference models despite large differences in the decoder-Jacobian condition number. This result does not establish that isometry is generally irrelevant to intrinsic-dimension estimation.
```

---

## 12. 構成の改善

### 12.1 Introductionをやや短縮する

**該当箇所：Section 1.2–1.4**

序論では以下が複数回繰り返される。

- 一般解ではない。
- MNISTではpass。
- Fashion-MNISTはboundary case。
- Conv-VAE・自然画像には限界がある。
- AUはID推定量ではない。
- knee detectionは探索的である。

慎重な姿勢は長所だが、中心的な貢献が埋もれやすい。

#### 推奨構成

Introductionでは以下に絞る。

1. 問題設定：ボトルネック次元は経験則・試行錯誤で選ばれる。
2. 提案：\(\hat d_{\mathrm{ID}}\) に基づく探索窓とAU/MSE診断。
3. 主要結果：合成データとMNISTでの成立例、Fashion-MNIST・Conv-VAE・自然画像での境界。
4. 主な限定：一般解ではなく、条件付きの診断プロトコルである。

詳細なLimitationsはSection 6へ移すか、Section 1.4を半分程度に圧縮する。

---

### 12.2 “Contribution” と “Theoretical motivation” を分ける

**該当箇所：Section 1.2**

新規性と理論的根拠を分けて書くとよい。

#### 推奨する貢献の整理

1. **Methodological contribution**：\(\hat d_{\mathrm{ID}}\) に基づくestimate–set–verify型のボトルネック設計プロトコル。
2. **Diagnostic contribution**：AUとMSEを併用した、余剰次元の切り捨て・過剰正則化・topological obstructionの診断。
3. **Empirical contribution**：synthetic、MNIST、Fashion-MNIST、Conv-VAE、CIFAR-10、SVHNを通じた適用範囲の検証。
4. **Theoretical motivation**：理想化されたtopological lower boundと、線形ガウスrate–distortion代理モデル。

特にProposition 2は、一般の非線形VAEの理論保証ではなく、`theoretical motivation` または `interpretive surrogate` として位置づけるのが安全である。

---

## 13. 要旨の修正文案

以下は、現行の主張を保持しつつ、適用範囲と限定をより明確にした要旨案である。

```text
The bottleneck dimension \(m\) is a fundamental design choice in autoencoders (AEs) and variational autoencoders (VAEs), but it is often selected by convention or trial and error. We propose a conditional, diagnostic protocol that uses an estimate \(\hat d_{\mathrm{ID}}\) of the intrinsic dimension as a data-dependent guide for selecting \(m\). For AEs, we recommend searching over a finite range near \(\hat d_{\mathrm{ID}}\), typically \([\hat d_{\mathrm{ID}},2\hat d_{\mathrm{ID}}]\); for VAEs, we recommend using a sufficiently generous candidate dimension and verifying whether surplus coordinates become inactive. The protocol combines intrinsic-dimension estimation, bottleneck selection, and post-training verification using active units (AU) and reconstruction error. Its geometric motivation is a topological lower bound for zero-error continuous reconstruction of an idealized manifold, while its VAE interpretation is supported by a linear-Gaussian rate–distortion surrogate with activation condition \(\lambda_j>\beta\tau^2\). On synthetic manifolds and fully connected VAEs trained on MNIST and Fashion-MNIST, the protocol provided a useful diagnostic under the tested settings. On MNIST, the ID-guided procedure reduced the number of training runs for the specific grid considered here by approximately 60–80% while retaining the best observed linear-probe accuracy within the tested window. However, the method is conditional rather than universal: intrinsic-dimension estimates depend on the reference representation and estimator, AU depends on \(\beta\), threshold, and architecture, and AU verification was not resolved for the tested natural-image Conv-VAEs. These results support the use of \(\hat d_{\mathrm{ID}}\) as a search-window reference and AU as a model- and training-dependent diagnostic, rather than as a general estimator of intrinsic or embedding dimension.
```

---

## 14. 結論の修正文案

```text
This study introduced a conditional estimate–set–verify protocol for selecting AE and VAE bottleneck dimensions. The protocol uses intrinsic-dimension estimates from reference representations to define a candidate range and uses AU and reconstruction-error behavior to diagnose whether the selected VAE bottleneck has been effectively truncated. The synthetic experiments demonstrated that the procedure can detect both ordinary saturation and a topological boundary case in which the default margin is insufficient. On MNIST, the protocol produced a consistent diagnostic under the tested fully connected architecture and reduced the number of training runs relative to the specific full-grid experiment. On Fashion-MNIST, it correctly identified a lower-bound violation of the AU consistency criterion rather than returning an unconditional pass.

These results should not be interpreted as establishing that \(\hat d_{\mathrm{ID}}\) estimates the topological embedding dimension or that AU estimates intrinsic dimension. The former is estimator- and representation-dependent, while the latter depends on the regularization strength, AU threshold, decoder capacity, and architecture. In particular, the linear-Gaussian activation condition is a theoretical surrogate and not a quantitative theorem for nonlinear VAEs. Conv-VAEs and natural-image experiments further showed that sufficient truncation and stable AU saturation cannot be assumed at standard \(\beta\). Future work should therefore focus on calibrated decision thresholds, independent data and seed splits, multi-seed reference-model sweeps, stronger validation on natural images, and theory for nonlinear posterior collapse.
```

---

## 15. 投稿前チェックリスト

### 必須修正

- [ ] タイトルの `Choosing` を、診断的プロトコルであることが伝わる表現へ変更する。
- [ ] Abstractで、VAEの \(m\ge2\hat d_{\mathrm{ID}}\) が保証ではないことを明記する。
- [ ] \(d_{\mathrm{ID}}\)、\(\hat d_{\mathrm{ID}}\)、\(d_{\mathrm{true}}\)、\(d_{\mathrm{emb}}\)、AUを全節・全図表で厳密に区別する。
- [ ] Proposition 1を理想条件下の必要条件として明記する。
- [ ] Proposition 2を線形ガウス代理モデルに限定し、非線形VAEへの定量的適用を否定する。
- [ ] Proposition 3を厳密定理として扱うか、heuristic stability rationaleに改称する。
- [ ] AUが内在次元・埋め込み次元の推定量ではないことをAbstractとConclusionにも入れる。
- [ ] AUの座標依存性、潜在空間の回転・再パラメータ化に対する非不変性を追記する。
- [ ] Step 1の参照AEとStep 3の候補VAEを明確に分離する。
- [ ] Table 6の単一seed sweepとTable 8の3seed VAE結果を混同しない。
- [ ] AEの単一seed評価とVAEの3seed評価を表・本文で明示する。
- [ ] 計算削減率を条件別の表に整理する。
- [ ] 固定 \(\beta\) の理論と周期的KLアニーリング実験を区別する。
- [ ] 自然画像に対する説明を未検証仮説として明記する。
- [ ] SVHNのraw-input TwoNN値を真値のように扱わない。

### 可能なら追加する修正

- [ ] 複数のデータ分割seedによる補足実験を追加する。
- [ ] 参照AEの \(m_{\mathrm{ref}}\)-sweepを複数seedで再実行する。
- [ ] seedごとの線形プローブ精度を補足資料に掲載する。
- [ ] \(\delta\) の潜在スケーリング依存性を議論する。
- [ ] V1/V2閾値の事前根拠、またはより広いデータでの校正を検討する。
- [ ] KLアニーリングの詳細スケジュールを図・表で提示する。
- [ ] 補足資料としてコード、ログ、データインデックス、設定ファイルを査読時に提供する。
- [ ] クラス条件付き・領域別のID推定、または局所ID分布の分析を追加する。
- [ ] 自然画像におけるdecoder capacityと \(\beta\) の系統的ablationを増やす。

---

## 16. 最終的な投稿方針

本稿は、次のような主張に統一すると最も強い。

> 本研究は、内在次元推定値からAE/VAEの唯一の最適ボトルネック次元を決定する一般解を与えるものではない。代わりに、\(\hat d_{\mathrm{ID}}\) を探索窓の参照値として用い、VAEではAUと再構成誤差を併用して、余剰潜在次元の切り捨て、過剰正則化、推定不足、トポロジー由来の不足マージンなどを診断する、条件付きのestimate–set–verifyプロトコルを示す。

この位置づけであれば、MNISTでの成功例、Fashion-MNISTでの境界診断、Conv-VAEと自然画像における限界が、矛盾ではなく「適用条件と失敗モードを可視化した実証」として統一的に説明できる。
