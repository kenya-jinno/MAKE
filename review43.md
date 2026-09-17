# MAKE43改訂稿レビュー

対象原稿：

> **Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and Variational Autoencoders with Active-Unit Diagnostics**

本レビューは、MDPI *Machine Learning and Knowledge Extraction*（MAKE）投稿予定の `main_paper_MAKE43.pdf` を対象とする。

## 総合評価

MAKE43改訂稿は、前回の主要コメントに対して概ね適切かつ十分に対応できている。特に、以下の改善は重要である。

- タイトルが、一般的な最適ボトルネック次元決定法ではなく、ID-guided selectionとAU diagnosticsを表すものに変更された。
- 要旨と序論で、方法が **conditional, diagnostic protocol** であることを明示した。
- \(\hat d_{\mathrm{ID}}\) を唯一の正解や真の内在次元ではなく、探索窓の参照値として位置づけた。
- AUを、内在次元や埋め込み次元の推定量ではなく、\(\beta\)、AU閾値、アーキテクチャ、学習条件に依存する診断量として明確化した。
- \(d_{\mathrm{ID}}\)、\(\hat d_{\mathrm{ID}}\)、\(d_{\mathrm{emb}}\)、AUの区別を強化した。
- Proposition 1を、理想化された連続多様体・ゼロ再構成誤差の下での必要条件として限定した。
- Proposition 2を、線形ガウスrate–distortion代理モデルに限定し、非線形VAEへの定量的定理ではないと明記した。
- Step 1の参照AEとStep 3の候補VAEを明確に分離した。
- \(m_{\mathrm{ref}}\)-sweepが単一seedであること、固定順序サブセットを使用することを明示した。
- V1/V2が統計的検定ではなく、運用上のスクリーニング基準であることを明記した。
- Conv-VAEにおいて、固定\(\beta\)の理論と周期的KLアニーリングを区別した。
- 自然画像に関する説明を、実験で識別された結論ではなく未検証仮説として整理した。
- 探索run数削減の比較を表形式で提示した。

現段階で残る主要課題は、論文の中心的アイデアそのものではなく、**数式の正確性、観測事実と代理モデル解釈の分離、図中表記の整合性、表現の過度な一般化**である。

---

## 最優先の修正事項

### 1. Appendix A.1：\(D_j(R_j)\) の定義を修正

**該当箇所：Appendix A.1、887–890行目**

現行原稿では、

> Let \(R_j\ge0\) be the rate ... and \(D_j(R_j)\) the corresponding distortion reduction; the Gaussian rate–distortion function gives \(D_j(R_j)=\lambda_j e^{-2R_j}\).

と記述されている。

しかし、

\[
D_j(R_j)=\lambda_j e^{-2R_j}
\]

は通常、「歪み減少量」ではなく、**残余歪み（residual distortion）**、またはその固有方向に残る残余分散寄与として解釈される。

### 推奨修正文

```text
Let \(R_j\ge0\) be the rate allocated to eigendirection \(j\), and let \(D_j(R_j)\) denote its residual variance contribution. Under the Gaussian rate–distortion relation,
\[
D_j(R_j)=\lambda_j e^{-2R_j}.
\]
```

続けて、以下を追加するとよい。

```text
Here \(R_j\) is measured in nats. The objective \(L_j\) is written up to additive constants and is maximized with respect to \(R_j\ge0\).
```

### 修正理由

- 式の意味と記号の定義を一致させる。
- rate–distortion解析の基礎的な意味論を明確にする。
- 査読者が数式の正当性を確認しやすくする。

---

### 2. Conv-VAEの \(d^*(\beta)>512\) を直接観測のように書かない

**該当箇所：Section 5.6.1、668–671行目**

現行原稿では、

> at \(\beta_{\max}=4\) the truncation threshold satisfies \(d^*(\beta)>512\)

と書かれている。

実験から直接観測できるのは、\(m\le512\)の範囲で明確な切り捨てが確認されなかったことまでである。\(d^*(\beta)\) は線形ガウス代理モデルにおける理論量であり、非線形・周期的KLアニーリングを使うConv-VAE実験から直接推定された量ではない。

### 推奨修正文

```text
The experiments do not observe substantial truncation within \(m\le512\). Under the surrogate interpretation, this pattern is consistent with an effective truncation threshold beyond the tested range.
```

記号を使うなら、次でもよい。

```text
The observed AU pattern is consistent, under the surrogate interpretation, with an effective truncation threshold exceeding the tested range, \(d^*_{\mathrm{eff}}(\beta)>512\).
```

### 日本語上の意味

> 実験では \(m\le512\) の範囲で大きな切り捨ては観測されなかった。代理モデルに基づけば、この傾向は、実効的な切り捨て閾値が検証範囲を超えていることと整合的である。

---

### 3. β-strengtheningの結果を「飽和」と断定しない

**該当箇所：Section 5.6.1、683–685行目；Appendix C.1、987–993行目**

現行原稿では、

> a clear induced saturation

とある。

\(\beta_{\max}=10,20\)でAU/mが低下していることは明確である。しかし、Table A3とFigure A1では、AU自体は\(m\)とともに増えており、検証範囲内で安定したAU plateauが確認されたとは言いにくい。

### 推奨修正文

```text
a clear reduction in the activation ratio and evidence of stronger truncation within the tested range
```

または、より慎重に：

```text
stronger truncation within the tested range, although stable AU saturation was not established
```

### 日本語上の意味

> 検証範囲内で活性化率の明確な低下と、より強い切り捨ての兆候が得られた。ただし、AUの安定した飽和が確認されたわけではない。

---

### 4. Figure 2・3・5の \(d_{\mathrm{ID}}\) 表記を \(\hat d_{\mathrm{ID}}\) に統一

**該当箇所：Figure 2、Figure 3、Figure 5の図中ラベルおよび凡例**

本文では推定値を\(\hat d_{\mathrm{ID}}\)と区別しているが、図中には次のような表記が残っている。

```text
Step-1 reference dID = 10
```

```text
dID = 10 (FC-VAE)
```

これは真の内在次元、あるいは候補FC-VAEから得られた値のように読める可能性がある。

### 推奨ラベル

```text
Rounded Step-1 reference \(\hat d_{\mathrm{ID}}=10\)
```

短くするなら：

```text
Step-1 ref. \(\hat d_{\mathrm{ID}}=10\)
```

Figure 5には、必要に応じてキャプションへ次を追加する。

```text
The reference value was obtained from the independently trained reference-AE procedure, not from the Conv-VAE latents shown here.
```

---

## 論理・方法に関する修正

### 5. `safe-side operation` を学術的表現へ変更

**該当箇所：Section 1.2、Section 3.2.2、Section 4.2**

現行表現：

> safe-side operation “generous \(m\) + AU diagnosis”

これは少し口語的であり、安全性を保証するようにも読める。

### 推奨修正文

```text
the conservative operating strategy of using a generous candidate dimension followed by AU-based verification
```

または、短く：

```text
a conservative operating strategy
```

---

### 6. Abstractの `three steps operate under a reproducible decision rule` を限定

**該当箇所：Abstract、13–14行目**

現行表現：

> the three steps operate under a reproducible decision rule

VAEのStep 3にはV1/V2があるが、AEのMSE確認は定性的である。また、Step 1の\(m_{\mathrm{ref}}\)-sweepは単一seedを含む。したがって、三段階すべてが同一の意味で「decision rule」によって機能すると読めないようにする。

### 推奨修正文

```text
The protocol produced reproducible diagnostics using a pre-specified Step-3 screening rule for VAEs and complementary reconstruction-error diagnostics.
```

または、データセット別に明記する。

```text
On synthetic manifolds and fully connected MNIST VAEs, the protocol produced reproducible diagnostics; on Fashion-MNIST it identified an over-pruning boundary case, while its applicability remained unresolved for the tested Conv-VAEs on natural images.
```

---

### 7. Abstractの `3+ seeds each` を具体化

**該当箇所：Abstract、12–13行目**

現行：

> MNIST and Fashion-MNIST (3+ seeds each)

### 推奨修正文

```text
On synthetic manifolds using three to five seeds and on fully connected VAEs trained on MNIST and Fashion-MNIST using three seeds, ...
```

### 修正理由

- 合成多様体と実データVAEのseed数を正確に分けられる。
- `3+`という曖昧な書き方を避けられる。

---

### 8. Section 5.3.3：AE探索窓 \([10,24]\) の算出規則を明文化

**該当箇所：Section 5.3.3、557–559行目**

現行：

> From \(\hat d_{\mathrm{ID}}\approx10\text{–}12\), the search window is \(m\in[10,24]\)

現状では、下限10と上限24がどう導かれたかを読者が完全に再現できない。

### 推奨修正文

```text
Using the rounded primary TwoNN reference \(\hat d_{\mathrm{ID}}=10\), the default AE window would be \([10,20]\). In the practical comparison reported here, we use the broader window \([10,24]\) to incorporate the MLE cross-check range and the tested grid.
```

一般規則として明示するなら、例えば次のように書ける。

\[
m_{\min}=\operatorname{round}(\hat d_{\mathrm{TwoNN}}),
\qquad
m_{\max}=\left\lceil 2\hat d_{\mathrm{MLE}}\right\rceil.
\]

ただし、この式を一般プロトコルにするのではなく、MNISTでの実用的な窓の設定として限定する方が安全である。

---

### 9. Section 4.1：\(\hat d_{\mathrm{ID}}\) を形式的下限のように書かない

**該当箇所：Section 4.1、341–342行目**

現行：

> \(\hat d_{\mathrm{ID}}\) provides the reference point “below this is insufficient in principle”

この表現は、\(\hat d_{\mathrm{ID}}\)が実データに対する形式的下限であるように読める。しかし本文では、\(\hat d_{\mathrm{ID}}\)は推定値であり、\(d_{\mathrm{emb}}\)の推定値ではないと明示している。

### 推奨修正文

```text
\(\hat d_{\mathrm{ID}}\) provides a practical reference point below which insufficient representation capacity is more likely under the tested conditions; it is not a formal lower bound for real datasets.
```

---

## Proposition 2とAppendix A.1に関する技術的確認

### 10. rate–distortion目的関数の定義を再確認

**該当箇所：Appendix A.1、883–895行目**

以下の式は査読者が特に確認しやすい部分である。

\[
L_j(R_j)
=
-\frac{1}{2\tau^2}\lambda_j e^{-2R_j}
-\beta R_j
+\mathrm{const.}
\]

\[
R_j^*
=
\max\left(0,\frac{1}{2}\log\frac{\lambda_j}{\beta\tau^2}\right).
\]

投稿前に、以下を実装・理論ノートと照合する。

- \(L_j\)は最大化対象か。
- \(R_j\)の単位はnatか。
- \(\lambda_j\)は中心化済み入力共分散の固有値か。
- \(\tau^2\)はガウス尤度における観測ノイズ分散か。
- \(\beta\)の位置が実装した\(\beta\)-VAE目的関数と一致するか。
- \(D_j(R_j)\)が残余歪みか、歪み減少量か。
- \(d^*(\beta)\)と潜在座標数制約\(m\)の関係が一貫しているか。

---

### 11. 集約事後分布の不一致が活性座標でゼロという主張

**該当箇所：Appendix A.1、903–911行目**

現行：

> the aggregated-posterior mismatch vanishes on truncated coordinates and is zero at this optimum on active ones

この主張を維持するなら、対象とする特定の線形ガウス大域最適解について十分な導出を補足すべきである。十分に示せない場合は、以下のように弱める方が安全である。

```text
At the particular solution considered here, the aggregated-posterior mismatch is zero on truncated coordinates and is assumed to be negligible on active coordinates. This assumption is not made for arbitrary linear or nonlinear VAEs.
```

---

### 12. `underestimating truncation` を明瞭に言い換える

**該当箇所：Section 4.2、390–392行目；Appendix A.1、915–917行目**

現行：

> the surrogate errs on the side of underestimating truncation

これは「活性次元数を過小評価する」という逆の意味に読まれる可能性がある。

### 推奨修正文

```text
The surrogate tends to predict weaker truncation than would be induced by the additional aggregated-posterior mismatch; equivalently, it may overestimate the number of active dimensions relative to the nonlinear model.
```

---

## 実験結果の解釈に関する修正

### 13. トーラスの “self-intersecting representation” を弱める

**該当箇所：Section 5.2、477–480行目**

現行：

> the model is pushed into a self-intersecting two-dimensional representation (an immersion)

MSE、AU、トーラスの埋め込み次元に基づく解釈としては理解できるが、実際に写像の自己交差を直接可視化・検証したわけではない。

### 推奨修正文

```text
The model is consistent with a two-dimensional representation that cannot provide a globally self-intersection-free embedding of the torus; geometrically, this may correspond to an immersion-like or self-overlapping representation.
```

---

### 14. トーラスの “deterministic” を再現性表現に変更

**該当箇所：Section 5.2、470–474行目**

現行：

> the transition at \(m=5\) is deterministic

5seedで同じ結果が得られたことは有用だが、理論的・一般的な決定性を意味するわけではない。

### 推奨修正文

```text
The transition at \(m=5\) was identical across all five tested seeds under this training setup.
```

または、

```text
The transition at \(m=5\) was reproducible across all five tested seeds.
```

---

### 15. Table 3の `Fully applicable` を条件付きにする

**該当箇所：Table 3、Synthetic manifoldsのOverall欄**

現行：

> Fully applicable (ground-truth validated)

トーラスでは、デフォルト探索窓\([2,4]\)が不足し、MSE cross-checkに基づく追加マージンが必要だった。そのため、デフォルト手順だけで完全に成功したように見せない方がよい。

### 推奨修正文

```text
Ground-truth validated with an adaptive margin diagnostic
```

または、

```text
Applicable under the tested settings, provided that the MSE cross-check can trigger margin enlargement
```

---

### 16. Fashion-MNISTのV2下限違反を因果断定にしない

**該当箇所：Table 2、Section 5.3.6**

V2下限違反は過剰正則化と整合的であるが、それを唯一の原因と断定したわけではない。Step 1参照AEが、VAEによって刈り込まれる方向を含む可能性もある。

### 推奨

Table 2の列名を `Likely cause` から `Diagnostic interpretation` に変更する。

| Symptom | Diagnostic interpretation | Follow-up |
|---|---|---|
| AU saturates below \(\hat d_{\mathrm{ID}}\) | Consistent with excessive regularization or with the Step-1 reference containing directions pruned by the VAE | Sweep \(\beta\), inspect MSE and downstream metrics, and treat the interval between AU and \(\hat d_{\mathrm{ID}}\) as a diagnostic range |

---

## 自然画像に関する修正

### 17. \(m_{\mathrm{ref}}\approx2\hat d_{\mathrm{ID}}\) を一般規則として書かない

**該当箇所：Section 5.6.2、700–703行目、742–746行目**

自然画像において、\(m_{\mathrm{ref}}\approx2\hat d_{\mathrm{ID}}\)付近で既報値と整合したことは興味深いが、一般的な推奨値として扱うべきではない。

### 推奨修正文

```text
For the natural-image experiments considered here, \(m_{\mathrm{ref}}\approx2\hat d_{\mathrm{ID}}\) behaved as a provisional consistency region rather than as a validated selection rule.
```

---

### 18. Allegra et al.に対する `exactly` を削除

**該当箇所：Section 5.6.2、720–722行目**

現行：

> Allegra et al. [16] model exactly this heterogeneous-local-ID structure

`exactly` はやや強い。

### 推奨修正文

```text
Allegra et al. [16] provide a related treatment of heterogeneous local intrinsic dimensionality.
```

または、

```text
Allegra et al. [16] study a closely related heterogeneous-local-ID structure.
```

---

## 図表・記号の整合性

### 19. Table 1の \(d_{\mathrm{emb}}\ge d_{\mathrm{ID}}\) の表記を改善

**該当箇所：Table 1**

現行：

> Topological minimum embedding dimension (≥ \(d_{\mathrm{ID}}\))

本文では\(\hat d_{\mathrm{ID}}\)と\(d_{\mathrm{emb}}\)の直接比較を避けているため、Table 1でも理想多様体の真の次元\(d\)との関係を示す方が一貫する。

### 推奨表記

```text
Minimum Euclidean embedding dimension of the specified manifold; for an ideal \(d\)-dimensional manifold, \(d_{\mathrm{emb}}\ge d\)
```

---

### 20. Figure 2のsingle-seed sweepと主要3seed判定の役割を明記

**該当箇所：Figure 2キャプション**

Figure 2はsingle-seed (42) sweepであり、Table 8とFigure 3が3seedの主な判定根拠である。この関係をキャプションで明示するとよい。

### 追加推奨文

```text
The single-seed VAE sweep is included for visualizing the dependence on \(m\); all principal VAE verification decisions are based on the independent three-seed results in Table 8 and Figure 3.
```

---

## 英文表現・構成上の修正

### 21. `The contributions ... are organized into three contributions` を修正

**該当箇所：Section 1.2、68–70行目**

現行：

> The contributions of this paper are organized into three contributions and one theoretical motivation:

### 推奨修正文

```text
The contributions and theoretical motivation of this paper are organized as follows:
```

---

### 22. Section 4のタイトルを `Theoretical Motivation` に統一

**該当箇所：Section 4タイトル**

現行：

> Theoretical Grounding of the Selection Principle

本文では、Proposition 1は理想条件下の必要条件、Proposition 2は線形ガウス代理モデル、Section 4.3はheuristic rationaleである。`grounding` より `motivation` の方が論文の実態に整合する。

### 推奨タイトル

```text
4. Theoretical Motivation and Interpretive Surrogates
```

または、

```text
4. Geometric and Probabilistic Motivation for the Protocol
```

---

### 23. `DANCo is more precise` を条件付きにする

**該当箇所：Section 2、153–156行目**

現行：

> DANCo [14] is more precise but computationally expensive

DANCoが一般に常に精密であるとは限らない。

### 推奨修正文

```text
DANCo can provide a useful complementary estimator but is computationally more expensive; we therefore use it only as an auxiliary cross-check.
```

---

### 24. `the manifold dimension` の曖昧さを減らす

**該当箇所：Introduction、40–45行目など**

本文の中心は、\(d_{\mathrm{ID}}\)、\(d_{\mathrm{emb}}\)、\(\hat d_{\mathrm{ID}}\)の区別である。`the manifold dimension` は便利だが、それらを曖昧にする。

### 推奨表現

```text
the appropriate bottleneck scale should be informed by a data-side geometric quantity, namely an estimate of local intrinsic dimensionality
```

---

## 再現性・統計表現

### 25. 標準偏差と標準誤差を区別

**該当箇所：Section 5.1、全てのmean±std表記**

以下をSection 5.1に追加する。

```text
Unless otherwise stated, all \(\pm\) values in multi-seed experiments denote standard deviations across seeds, not standard errors or confidence intervals.
```

---

### 26. Fashion-MNISTの `identical` と `unanimous` を慎重に表現

**該当箇所：Section 5.3.6、Table 9**

実際に各seedの整数AUがすべて7なら、以下の方が明確である。

```text
All three tested seeds yielded AU = 7 at \(m=20\).
```

表示値の丸めによる一致なら、以下にする。

```text
The displayed AU was 7.0 for all three seeds at \(m=20\).
```

`unanimous` の代わりには、次を推奨する。

```text
All three tested seeds received the same operational verdict.
```

---

## 投稿前チェックリスト

### 最優先

- [ ] Appendix A.1で \(D_j(R_j)\) を `residual variance contribution` に変更する。
- [ ] rate–distortion目的関数の符号、最大化方向、\(R_j\)の単位、\(\tau^2\)の定義を再確認する。
- [ ] 活性座標における集約事後分布の不一致ゼロという主張を、証明するか `assumed negligible` に変更する。
- [ ] Section 5.6.1の \(d^*(\beta)>512\) を、代理モデルとの整合的解釈に書き換える。
- [ ] `clear induced saturation` を、活性化率低下・より強い切り捨ての証拠という表現に変更する。
- [ ] Figure 2、Figure 3、Figure 5の \(d_{\mathrm{ID}}=10\) を \(\hat d_{\mathrm{ID}}=10\) に統一する。
- [ ] Abstractの `3+ seeds each` を具体的なseed設計へ変更する。
- [ ] Abstractの `three steps operate under a reproducible decision rule` を、VAEのV1/V2と補助的MSE診断に限定する。
- [ ] Section 5.3.3で、AE探索窓\([10,24]\)の算出方法を再現可能な形で明記する。
- [ ] Table 1の\(d_{\mathrm{emb}}\ge d_{\mathrm{ID}}\)を、理想多様体の真の次元\(d\)に関する表記へ変更する。

### 強く推奨

- [ ] `safe-side operation` を `conservative operating strategy` に変更する。
- [ ] Section 4のタイトルを `Theoretical Motivation and Interpretive Surrogates` に変更する。
- [ ] `The contributions ... are organized into three contributions` を修正する。
- [ ] `DANCo is more precise` を条件付き表現に変更する。
- [ ] トーラスの `self-intersecting representation` を `consistent with ...` に弱める。
- [ ] トーラスの `deterministic` を `reproducible across all five tested seeds` に変更する。
- [ ] Table 3の `Fully applicable` を、adaptive marginを含む条件付き表現に変更する。
- [ ] 自然画像の\(m_{\mathrm{ref}}\approx2\hat d_{\mathrm{ID}}\)を暫定的整合領域として表現する。
- [ ] `Allegra et al. ... exactly` の `exactly` を削除する。
- [ ] seed間標準偏差と標準誤差・信頼区間を明示的に区別する。
- [ ] Figure 2のsingle-seed sweepとTable 8/Figure 3の3seed主要検証を明確に分離する。

### 可能であれば追加

- [ ] 参照AEの\(m_{\mathrm{ref}}\)-sweepを複数seedで実施する。
- [ ] 複数データ分割seedでMNISTの主要結果を確認する。
- [ ] Conv-VAEの\(\beta_{\max}\)×decoder capacity ablationを追加する。
- [ ] 全seedのlinear-probe accuracyを補足資料に掲載する。
- [ ] コード・ログ・データインデックスを含む補足資料が匿名査読時に利用可能であることを確認する。

---

## 最終判定

MAKE43改訂稿は、前回の主要指摘を十分に反映しており、現時点では「内在次元から唯一の最適ボトルネック次元を決定する一般解」ではなく、次のような主張として一貫して読める。

> \(\hat d_{\mathrm{ID}}\) はボトルネック次元の唯一の最適値を与えるものではなく、探索窓の参照値である。AUは内在次元や埋め込み次元の推定量ではなく、指定された \(\beta\)、AU閾値、アーキテクチャ、学習条件の下での潜在座標利用を診断する量である。

この位置づけは、MNISTでの成功例、Fashion-MNISTの境界事例、Conv-VAEと自然画像における未解決性を一貫して説明できている。

投稿前に特に確認すべき残存課題は以下の6点である。

1. Appendix A.1のrate–distortion式の用語と導出。
2. \(d^*(\beta)>512\) を直接観測と誤解されない表現への変更。
3. β-strengtheningで「飽和」と断定しないこと。
4. 図中の\(d_{\mathrm{ID}}\)と\(\hat d_{\mathrm{ID}}\)の統一。
5. AE探索窓\([10,24]\)の算出規則の明文化。
6. 表・図・参考文献の相互参照の最終確認。

これらを修正すれば、概念的な問題はほぼ解消され、MAKEへの投稿原稿として十分に整った状態に近づく。