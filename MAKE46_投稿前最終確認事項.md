# MAKE46 投稿前最終確認事項

対象原稿：

> **Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics**

対象ファイル：`main_paper_MAKE46.pdf`

本メモは、最終投稿前に確認・必要に応じて修正すべき事項だけを整理したものである。現行原稿には投稿を妨げる重大な問題は見当たらないが、以下を最終確認すると、理論的な限定、Conv-VAEの解釈、図表・補足資料との整合性がさらに明確になる。

---

## 1. Conv-VAEのAU診断表現

**該当箇所：Section 5.6.1、Figure 5キャプション**

Figure 5キャプションには、以下の表現が残っている。

> the failure mode “\(\beta\) too small relative to decoder power” detected by the AU signal

AUから直接観測されるのは、検証範囲においてAUが\(m\)に近く、明確な切り捨てが観測されないことである。\(\beta\)がデコーダ容量に対して小さいことは、代理モデルと実験結果に基づく診断解釈であり、AUだけによる因果状態の直接検出ではない。

### 推奨修正文

```text
Figure 5. Boundary condition (Exp-Conv-M256): the MNIST Conv-VAE with \(\beta_{\max}=4\) keeps \(AU=m\) (no truncation) up to \(m=64\) and remains highly active at \(m=256\). The AU pattern indicates that no substantial truncation is observed within the tested range, which is consistent with insufficient \(\beta\) relative to decoder capacity. The reference value \(\hat d_{\mathrm{ID}}=10\) was obtained from the independently trained reference-AE procedure of Step 1, not from the Conv-VAE latents shown here.
```

### 修正意図

- 観測事実：`no substantial truncation observed`。
- 診断解釈：`consistent with insufficient \(\beta\)`。
- 因果的断定：`detected` を避ける。

---

## 2. 「generous \(m\) is safe」の表現

**該当箇所：Section 4.2、Remark 1**

現行文に、以下のような表現が残っていないか確認する。

> a generous \(m\) is safe for VAEs

論文全体では、余剰次元の切り捨てが、\(\beta\)、アーキテクチャ、デコーダ容量、学習スケジュールに依存することを明確にしている。そのため、`safe` は保証的に読める可能性がある。

### 推奨修正文

```text
This motivates the conservative strategy of using a generous \(m\) and verifying the resulting active-unit count, which can be useful when KL-induced truncation is effective.
```

または、より短く：

```text
Using a generous \(m\) can be useful when KL-induced truncation is effective.
```

### 修正意図

- VAEで大きな\(m\)を使うことが一般に安全だと断定しない。
- Step 3のAU診断が必須であるという本稿の中心立場に整合させる。

---

## 3. 合成多様体での代理モデルとの対応

**該当箇所：Section 5.2、VAE AU verificationの説明**

現行文に、以下のような強い表現が残っていないか確認する。

> the surrogate prediction ... is confirmed on ground truth in this setting

合成多様体での実験は重要な原理確認であるが、線形ガウス代理モデルの予測が非線形VAE一般で証明されたことを意味しない。

### 推奨修正文

```text
Under this synthetic setting, the observed behavior is consistent with the surrogate expectation that a VAE can truncate surplus dimensions when a generous \(m\) is used and AU is read out.
```

### 日本語上の意味

> この合成データ条件では、十分に大きな\(m\)を用いた場合にVAEが余剰次元を切り捨て、AUがその結果を反映するという代理モデル上の期待と、観測結果は整合している。

### 修正意図

- `confirmed` を `consistent with` に弱める。
- 特定の合成条件下での結果であることを明示する。

---

## 4. Table 3の適用範囲表現

### 4.1 Synthetic manifolds

**該当箇所：Table 3、Overall列**

現行：

```text
Validated on synthetic ground truth when adaptive margin enlargement is allowed
```

この表現は十分慎重であり、現行のままでも問題ない。さらに限定する場合は以下を推奨する。

```text
Validated on the tested synthetic manifolds when MSE-triggered margin enlargement is allowed
```

### 修正意図

- トーラスではデフォルト窓\([2,4]\)だけでは不十分だった。
- MSE cross-checkによるマージン拡大を含めて適用可能であったことを明確にする。

### 4.2 Conv-VAE on MNIST

**該当箇所：Table 3、Overall列**

現行：

```text
Conditional applicability after architecture- and schedule-dependent truncation is induced; order consistency remains open
```

これは概ね適切である。より慎重にする場合は以下のようにする。

```text
Conditional applicability when sufficient truncation can be induced under the chosen architecture and training schedule; order consistency remains open
```

### 修正意図

- どのアーキテクチャ・スケジュールでも切り捨てが誘導できると保証しない。
- \(\beta\)、デコーダ容量、KL schedule、データ構造への依存を残す。

---

## 5. 図・表・参考文献・補足資料の最終確認

### 5.1 図表と記号

以下を最終PDFで目視確認する。

- [ ] Figure 2、Figure 3、Figure 5で、\(\hat d_{\mathrm{ID}}=10\)が正しく表示されている。
- [ ] Figure 5の参照値が、候補Conv-VAE潜在表現ではなく、独立に学習したreference AE由来であることがキャプションで明確である。
- [ ] AU、TwoNN ID、MSEの軸・凡例・seed数の表記が図とキャプションで一致している。
- [ ] \(d_{\mathrm{ID}}\)、\(\hat d_{\mathrm{ID}}\)、\(d_{\mathrm{emb}}\)、AU、\(AUsat\)、\(d^*(\beta)\)、\(d^*_{\mathrm{eff}}(\beta)\)の記号が本文・表・図で一貫している。
- [ ] Figure 2–5、Figure A1のキャプションと本文中の参照番号が一致している。

### 5.2 表・式・引用の相互参照

- [ ] Table 1–13、Table A1–A4の番号と本文中の参照が一致している。
- [ ] Equation (1)–(4)、(A1)–(A3)への参照が一致している。
- [ ] 参考文献番号が本文初出順に整列している。
- [ ] 新規に追加した著者らの先行研究引用が、本文の主張を適切に支えている。
- [ ] MNIST、Fashion-MNIST、CIFAR-10、SVHN、TwoNN、MLE、DANCo、ESS、cyclical annealing、hierarchical VAEの引用番号が正しい。

### 5.3 補足資料の整合性

Data Availability StatementおよびSupplementary Materialsに記載した内容が、匿名査読用の添付ファイルに実際に含まれていることを確認する。

- [ ] Table A4の全実験に対応する実験スクリプト。
- [ ] 全seedのJSON学習ログ。
- [ ] モデル設定ファイル。
- [ ] 乱数seed。
- [ ] データサブセットのindex。
- [ ] AU計算コード。
- [ ] TwoNN、MLE、DANCo、ESSの実装情報・バージョン。
- [ ] KL cyclic annealingのスケジュール実装。
- [ ] Figure 2–5、Figure A1および全表の生成スクリプト。
- [ ] run数削減率を再現するTable 10生成コード。
- [ ] Python、PyTorch、CUDA、GPU、主要ライブラリのバージョン情報。
- [ ] 匿名化により著者名、所属、ローカルパス、private repository URLなどが補足資料に残っていない。

---

## Appendix A.1の最終確認

投稿前に、次の区別が維持されていることを確認する。

- [ ] 切り捨て座標では\(q_\phi(z_j\mid x)=p(z_j)\)であり、その座標のaggregated-posterior mismatchはゼロ。
- [ ] 活性座標ではmismatchを `assumed negligible` とし、exact vanishingを証明したとは書かない。
- [ ] 非線形VAEでは、線形ガウス代理モデルの活性化条件を定量予測・座標別定理として使わない。
- [ ] \(d^*_{\mathrm{eff}}(\beta)\)を使用する場合、非線形・周期的KL annealing実験に対する解釈量であり、直接推定量ではないと明記する。

---

## 最終判断

現行のMAKE46原稿には、投稿を妨げる重大な問題は見当たらない。上記の表現調整は、投稿可否を左右するものではなく、査読者による「過度な一般化」または「因果的断定」の懸念をさらに減らすための最終仕上げである。

特に、以下の3点を確認または修正すれば、投稿時の完成度が高まる。

1. Figure 5キャプションの `detected` を、AUパターンに基づく診断的な表現にする。
2. `a generous \(m\) is safe for VAEs` に相当する表現を条件付きに統一する。
3. 合成多様体における代理モデルとの一致を `confirmed` ではなく `consistent with` と表現する。

これらに加え、最終PDFの相互参照と匿名補足資料の内容を確認すれば、MAKE投稿用原稿として十分に整った状態である。