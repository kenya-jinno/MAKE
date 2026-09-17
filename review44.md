# MAKE44改訂稿レビュー：残存修正事項

対象原稿：

> **Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics**

対象ファイル：`main_paper_MAKE44.pdf`

## 前提

Figure 2およびFigure 3の図中ラベルについて、PDFテキスト抽出では `d_{\mathrm{ID}}=10` のように見える箇所があるが、目視確認の結果、実際のPDF表示では正しく

\[
\hat d_{\mathrm{ID}}=10
\]

とレンダリングされていることを確認済みである。したがって、**Figure 2およびFigure 3のハット記号に関する指摘はOCR／PDFテキスト抽出上の問題であり、修正対象から除外する。**

本メモでは、それ以外の残存修正事項のみを整理する。

---

## 総合評価

MAKE44は、内容面・論文英語の両面で、MAKE投稿に向けて十分に準備された状態に近い。前回の主要な懸念であった以下の点は、概ね適切に修正されている。

- 手法を「一般的な最適次元決定法」ではなく、条件付きの探索窓・診断プロトコルとして限定した。
- \(\hat d_{\mathrm{ID}}\) を探索窓の参照値として位置づけ、真のIDや埋め込み次元の推定値と混同しないようにした。
- AUをID推定量ではなく、\(\beta\)、AU閾値、アーキテクチャ、学習条件に依存する診断量として明記した。
- Proposition 1を理想化条件下の必要条件に限定した。
- Proposition 2を線形ガウス代理モデルに限定した。
- Step 1の単一seed制約と固定順序データサブセットの制約を明示した。
- V1/V2を統計的検定ではなく、運用上のスクリーニング規則として位置づけた。
- Conv-VAEにおける固定\(\beta\)理論と周期的KLアニーリング実験を区別した。
- 自然画像に関する説明を、未検証仮説として適切に限定した。
- \(D_j(R_j)\) を残余分散寄与として定義し直し、rate–distortionの説明を改善した。
- \(d^*(\beta)>512\) を直接観測ではなく、代理モデルとの整合的解釈として表現した。

現時点での残存課題は、論文の中心的アイデアや英語全体の質ではなく、**理論記述の最終整合、観測事実と診断解釈の分離、要旨の実験条件への限定、図表・相互参照の最終確認**である。

---

## 最優先の修正事項

### 1. Appendix A.1の “exact at the global optimum” をさらに整合化する

**該当箇所：Appendix A.1、約928–937行目**

現行原稿では、次の趣旨が並んでいる。

> In the linear-Gaussian setting the approximation is exact at the global optimum.

一方で、すぐ後に、活性座標については

> on active coordinates it is assumed to be negligible at this optimum

と述べている。

活性座標のaggregated-posterior mismatchについて厳密なゼロを証明していないなら、「近似が大域最適で厳密に成立する」という一般的な表現は強すぎる。切り捨て座標と活性座標を明確に分けて記述することを推奨する。

### 推奨修正文

```text
For the particular globally optimized linear-Gaussian solution considered here, the approximation is exact on truncated coordinates. On active coordinates, the aggregated-posterior mismatch is assumed to be negligible; the PPCA-type variance-matching property motivates this assumption but does not by itself establish exact vanishing.
```

### 日本語上の意味

> 本稿で扱う特定の線形ガウス大域最適解では、切り捨てられた座標については近似が厳密に成立する。一方、活性座標については、集約事後分布の不一致が無視できると仮定する。PPCA型の分散一致はこの仮定を動機づけるが、不一致が厳密にゼロであることを単独で証明するものではない。

### 修正理由

- `exact` と `assumed negligible` の論理的緊張を解消する。
- Proposition 2が線形ガウス代理モデルの限定的な解析であるという論文全体の立場と整合させる。
- 理論面での過剰主張を避ける。

---

### 2. Conv-VAEにおいてAUが因果状態を直接「検出」するような表現を弱める

**該当箇所：Section 5.6.1、約685–689行目**

現行の趣旨：

> the AU verification detects the state “\(\beta\) too small relative to the decoder capacity” through the clear signal \(AU=m\)

AUから直接観測されるのは、検証範囲において明確な切り捨てが見られないこと、またはAUが\(m\)に近いことである。そこから「\(\beta\)がデコーダ容量に対して小さい」とするのは、代理モデルおよび実験知見に基づく診断解釈であり、因果状態の直接検出ではない。

### 推奨修正文

```text
The AU pattern flags the diagnostic condition “no substantial truncation observed within the tested range,” which is consistent with \(\beta\) being too small relative to decoder capacity.
```

### 日本語上の意味

> AUの挙動は、「検証範囲内で大きな切り捨てが観測されない」という診断状態を示しており、これはデコーダ容量に対して\(\beta\)が小さい可能性と整合する。

### Table 2も合わせて変更する場合

最初の行の `Diagnostic interpretation` を次のようにする。

```text
Consistent with \(\beta\) being too small relative to decoder capacity, or with the effective truncation threshold lying beyond the tested range.
```

---

### 3. Abstractの “standard \(\beta\)” を実験条件に限定する

**該当箇所：Abstract、約20–23行目**

現行：

> for Conv-VAEs the AU truncation fails at standard \(\beta\)

`standard \(\beta\)` は一般的な概念に見えるが、実験では周期的KLアニーリングと\(\beta_{\max}=4\)を用いている。本文の慎重な立場に合わせ、要旨も実験条件に限定することを推奨する。

### 推奨修正文

```text
for the tested Conv-VAE schedule with \(\beta_{\max}=4\), AU-based truncation was not observed within the tested range
```

必要であれば続けて、

```text
whereas increasing \(\beta_{\max}\) reduced the activation ratio without establishing order consistency
```

と書く。

### 日本語上の意味

> 検証した\(\beta_{\max}=4\)のConv-VAE学習スケジュールでは、検証範囲内でAUに基づく明確な切り捨ては観測されなかった。一方、\(\beta_{\max}\)を増加させると活性化率は低下したが、IDとのorder consistencyは確立されなかった。

---

### 4. Abstractの “inflate on natural images” を具体化する

**該当箇所：Abstract、約19–21行目**

現行：

> intrinsic-dimension estimates were numerically stable only through reference representations with \(m\gg\hat d_{\mathrm{ID}}\) and inflate on natural images

ここでは、何がどの条件で増大するかがやや曖昧である。自然画像では、参照ボトルネックを大きくするとlatent-space TwoNN推定値が増大するという実験結果を明示するとよい。

### 推奨修正文

```text
on natural images, latent-space TwoNN estimates increased when the reference bottleneck exceeded the provisional consistency region
```

または、より短く：

```text
on natural images, TwoNN estimates increased beyond the provisional consistency region as the reference bottleneck grew
```

---

## 強く推奨する修正事項

### 5. Section 1.2の `can be expected to apply` をさらに条件付きにする

**該当箇所：Section 1.2、約67–70行目**

現行：

> the conservative operating strategy of using a generous candidate dimension followed by AU-based verification can be expected to apply

この書き方は、広い条件で適用可能という印象を残す。本文では、Conv-VAEや自然画像で成立しないことを示しているため、Step 3による検証が必要な仮説であると明示する方がよい。

### 推奨修正文

```text
The conservative operating strategy of using a generous candidate dimension followed by AU-based verification may be useful when KL-induced truncation is effective, a condition that must be checked by Step 3.
```

または、

```text
The conservative operating strategy is a hypothesis to be tested by the Step-3 diagnostic.
```

---

### 6. Section 4.3の `sufficient conditions` を弱める

**該当箇所：Section 4.3、約404–411行目**

現行：

> a sketch of sufficient conditions for conditional stability

Section 4.3自身が有限標本定理ではないと明示しているため、`sufficient conditions` は少し強い。

### 推奨修正文

```text
a heuristic description of conditions that may support conditional stability
```

### 日本語上の意味

> 条件付き安定性を支え得る条件についてのヒューリスティックな説明。

---

### 7. Section 5.3.1でprimary TwoNN referenceとMLE cross-checkを明確に分ける

**該当箇所：Section 5.3.1、約524–533行目**

現行では、

> yielding the reference value \(\hat d_{\mathrm{ID}}\approx10\text{–}12\)

と書かれている。

しかし、プロトコルのprimary ruleは丸めたTwoNN値\(\hat d_{\mathrm{ID}}=10\)であり、MLEの11.3–11.9はクロスチェックである。両者を一つのreference valueとして書くと、少し曖昧になる。

### 推奨修正文

```text
The pre-specified rounded-TwoNN rule yields the operational primary reference \(\hat d_{\mathrm{ID}}=10\), while MLE values of 11.3–11.9 provide an order-of-magnitude cross-check.
```

### 日本語上の意味

> 事前指定された丸めTwoNN規則により、実用上のprimary referenceは\(\hat d_{\mathrm{ID}}=10\)となる。一方、MLEの11.3–11.9は同程度の値であることを確認するクロスチェックとして用いる。

---

### 8. Table 6キャプションの「3-seed reproducibility」を正確化する

**該当箇所：Table 6キャプション**

現行では、Table 8が参照AE潜在表現そのものの3seed再現性を示すように読める余地がある。実際にはTable 8は候補VAE潜在表現における診断的クロスチェックである。

### 推奨修正文

```text
The three-seed candidate-VAE latent cross-check is reported in Table 8.
```

### 日本語上の意味

> 3seedによる候補VAE潜在表現のクロスチェックはTable 8に示す。

---

### 9. 自然画像節の \(d^*(\beta)\gg m\) を代理モデル解釈として明示する

**該当箇所：Section 5.6.2、約746–752行目**

現行：

> As a result \(d^*(\beta)\gg m\) and truncation vanishes

これは自然画像実験から直接測定された量ではなく、線形ガウス代理モデルに基づく解釈である。

### 推奨修正文

```text
Under the surrogate interpretation, this pattern would correspond to an effective active-direction threshold larger than the tested \(m\)-range, resulting in little or no observable truncation.
```

---

### 10. Table 3の表現をさらに慎重にする

#### Synthetic manifoldsのOverall欄

現行：

> Ground-truth validated with an adaptive margin diagnostic

これはかなり改善されているが、初期のデフォルト窓だけで常に正解に到達したわけではない。

### 推奨表現

```text
Validated on synthetic ground truth when adaptive margin enlargement is allowed
```

または、

```text
Ground-truth illustrated under the tested settings with adaptive margin enlargement
```

#### Conv-VAE on MNISTのOverall欄

現行：

> Applicable only with strong \(\beta\) (order consistency open)

切り捨ては\(\beta\)だけでなく、デコーダ容量、KLアニーリング、潜在構造にも依存する。

### 推奨表現

```text
Conditional applicability after architecture- and schedule-dependent truncation is induced; order consistency remains open
```

または、

```text
Conditional applicability after sufficient truncation is induced; order consistency remains open
```

---

## 英語表現の仕上げ

### 11. `post training` を修正

**該当箇所：Abstract、7–8行目**

現行：

```text
verifying post training whether
```

### 推奨修正

```text
verifying after training whether
```

または、

```text
performing post-training verification of whether
```

前者の方が自然で簡潔である。

---

### 12. Abstractの実験限定をもう一段明確にする

**該当箇所：Abstract、16–18行目**

現行：

> on the specific MNIST grid used here

これは適切な限定だが、評価プロトコルも限定されるため、次の方がより明確である。

### 推奨修正文

```text
for the specific MNIST grid and evaluation protocol used here
```

ここでevaluation protocolには、固定データサブセット、seed設計、線形プローブが含まれる。

---

### 13. “same order” をV2の運用帯域に合わせる

**該当箇所：Section 3.2.3、約275–299行目**

現行：

> evidence of a compression to the same order as the Step-1 intrinsic-dimension estimate

`same order` は数学的にはかなり広い意味を持ち得る。V2の具体的帯域は\([\hat d_{\mathrm{ID}},3\hat d_{\mathrm{ID}}]\)であるため、運用上の帯域として表現する方が明確である。

### 推奨修正文

```text
evidence that AU lies within the operational range defined relative to the Step-1 reference
```

---

## 内容・英語の最終評価

### 内容面

MAKE44は、提案の新規性、理論的動機、合成データでの原理確認、MNISTでのmulti-seed診断、Fashion-MNISTの境界事例、Conv-VAEおよび自然画像における適用限界が一貫して整理されている。

特に次の立場は、現在十分に明確である。

> \(\hat d_{\mathrm{ID}}\) は唯一の最適ボトルネック次元を与えるものではなく、探索窓の参照値である。AUは内在次元や埋め込み次元の推定量ではなく、指定した\(\beta\)、AU閾値、アーキテクチャ、学習条件における潜在座標の利用状況を診断する量である。

この立場なら、MNISTでの成立例、Fashion-MNISTのboundary case、Conv-VAEと自然画像における失敗・未解決性を矛盾なく説明できる。

### 英語面

英語は論文投稿レベルに達している。重大な文法誤りは見当たらず、概念上の限定も英文で明確に表現されている。

最終的に改善余地があるのは、次の点である。

- 長い一文の分割。
- `conditional`、`diagnostic`、`tested`、`surrogate interpretation` の近接反復の軽減。
- primary TwoNN referenceとMLE cross-checkの役割の整理。
- 因果的な断定を `consistent with`、`flags`、`under the surrogate interpretation` に寄せる。
- 図表中、本文中、キャプション中の記号の最終整合確認。

---

## 投稿前チェックリスト

### 必須

- [ ] Appendix A.1の `exact at the global optimum` を、切り捨て座標と活性座標で分けて記述する。
- [ ] Conv-VAEのAUパターンを、\(\beta\)不足の直接検出ではなく、診断的に整合する条件として表現する。
- [ ] Abstractの `standard \(\beta\)` を、\(\beta_{\max}=4\)のKLアニーリング実験に限定する。
- [ ] Abstractの `inflate on natural images` を、latent-space TwoNN値が参照ボトルネック増大時に上昇したこととして具体化する。
- [ ] Section 5.3.1のprimary TwoNN referenceとMLE cross-checkを明確に区別する。
- [ ] Table 6キャプションで、Table 8がcandidate-VAE latent cross-checkであることを明示する。
- [ ] `post training` を `after training` または `post-training` に修正する。

### 強く推奨

- [ ] Section 1.2の `can be expected to apply` を、Step 3で検証すべき条件付き仮説として表現する。
- [ ] Section 4.3の `sufficient conditions` を `conditions that may support` に弱める。
- [ ] 自然画像節の\(d^*(\beta)\gg m\)を代理モデル解釈として明示する。
- [ ] Table 3のSynthetic / Conv-VAEのOverall表現を、adaptive margin・architecture・schedule依存性に合わせてさらに慎重にする。
- [ ] V2の `same order` を、Step-1 referenceに相対的に定義されたoperational rangeへ言い換える。
- [ ] `for the specific MNIST grid used here` を `for the specific MNIST grid and evaluation protocol used here` に変更する。

### 最終確認

- [ ] 図中の\(\hat d_{\mathrm{ID}}\)表示が、目視上すべて正しいことを確認する。
- [ ] 図表番号、引用番号、Appendix参照、式番号を再コンパイル後のPDFで総点検する。
- [ ] 補足資料が匿名査読時に実際に利用可能であることを確認する。
- [ ] JSONログ、設定ファイル、データインデックス、seed、図表生成スクリプトが整合していることを確認する。

---

## 最終判定

MAKE44は、内容面・英語面ともにMAKE投稿を検討できる水準にある。重大な論理上の欠陥や、論文英語として投稿を妨げるレベルの問題は見当たらない。

ただし、投稿直前に優先して修正すべき残存点は次の3つである。

1. **Appendix A.1における線形ガウス近似の `exact` と `assumed negligible` の整合。**
2. **Conv-VAEのAU結果から因果状態を直接検出したように読める表現の弱化。**
3. **Abstractを含む本文で、実験上の条件・範囲により明示的に限定すること。**

これらを仕上げ、PDF上の図表・数式・相互参照を最終確認すれば、MAKE投稿用原稿として十分に整った状態と判断できる。