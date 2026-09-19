# MAKE52 原稿修正記録

作成日: 2026-09-19

`main_paper_MAKE51.tex` を基に、`MAKE52_実験検証結果.md`、対応するJSON、
実行コードおよびGECO/ARD原著を照合し、MAKE52のTeX・PDFを作成した。
今回の作業でモデルの再学習は行っていない。MAKE51、提供された実験コード・
生の結果JSON・検証結果文書は変更していない。

## タイトル

**Intrinsic-Dimension-Guided Bottleneck Selection for Variational Autoencoders with Active-Unit Diagnostics**

MAKE47のタイトルから **Autoencoders and** だけを削除した。
それ以外の表現を維持し、MAKE51で追加した Revisited 等は採用していない。

## 主な反映箇所

| 箇所 | 内容 |
|---|---|
| Abstract、Introduction、Discussion、Conclusions | ID開始幅に今回の条件で追加的な利益が観測されないことを主結論へ反映 |
| Section 3.5 / Algorithm 1 | ID-gated midpointを定義し、従来のbank-free midpointとの違いを明記 |
| Section 5.5 | 追加36学習runの設定、原著との差、native/転用、ELBO・時間計測の定義 |
| Section 6.2 / Table 5 | 開始幅の対照100条件、ID採用25条件・fallback75条件の層別結果 |
| Section 6.5 / Tables 9–11 | HC対ARM、ARDの厳密Jacobian、中心化2条件、native品質と転用品質 |
| Section 6.10 | ARMのper-example損失差とバッチ平均の順序に関する監査 |
| Appendix F | 全seedのゲート分布、制約軌跡、4データの曲線、test MSE、計測済み時間 |
| Appendix G、Data Availability | MAKE52補足資料のファイル対応と再生成手順 |

## 検証結果文書をそのまま採用しなかった点

### 1. 「真のELBO」の完了とは扱わない

`geco_arm.py::eval_pass` の `mc=32` はMSEだけを事後標本で平均する。
`rec` は常にposterior meanで計算され、`kl` はゲートを反映しない標準正規priorに
対する値である。ARDにも同じ評価関数が使われ、学習したprior分散を反映しない。
またGaussianの正規化定数も当該フィールドには含まれない。
保存値でも `eval_mean` と `eval_sampled_K32` の `rec` と `elbo` は一致する。

したがって新しい4データ×3seedのログを「nativeモデルのMC ELBO」や
「ELBOによる幅選択の追加検証」と記述していない。利用したのはMSE、
ゲート、関連度、曲線および計測時間である。妥当なMC ELBOによる幅選択は、
MAKE51で別途実施したMNIST・1seed・12幅の実験を維持した。

### 2. ARMは構成要素を検証したローカル実装

ARM推定量を導入しても、初期化・制約を初めて満たすまでの処理・KLの正規化などに
原著との相違が残る。ローカル実装は最初から確率0.5のゲートを更新し、masked KLの
和を使う。原著Algorithm 1との全面的な同一性や再現完了は主張しない。
[GECO/L0原著](https://arxiv.org/html/2003.10901v3)

追加12runの最終判定は次のとおり。分母は各データ3seedであり、代表seedだけの表を
3seed全体の結論に置き換えていない。

| データ | 訓練制約達成 | native mean Q | native sampled Q | 転用先Q |
|---|---:|---:|---:|---:|
| MNIST | 2/3 | 3/3 | 0/3 | 3/3 |
| Fashion-MNIST | 2/3 | 3/3 | 0/3 | 3/3 |
| dSprites | 0/3 | 0/3 | 0/3 | 3/3 |
| CIFAR-10 | 2/3 | 1/3 | 0/3 | 1/3 |

dSpritesは全seedの全64ゲートが確率0.1〜0.9に留まる。HCの64.0とARMの54.0を
「4データすべて選択一致」「修正後はすべて二値化し制約も満たす」とは記述しない。
またsampled再構成を比較した目標はdeterministic anchor由来であり、
sampled anchorを用いた公平な幅選択実験とは区別した。

### 3. ARDの中心化と既存の正規化補正

原著はprior-locationパラメータをゼロに設定しており、sample meanによる中心化が
原著再現に必須という扱いはしない。ゼロ中心の分散更新を主な比較とし、
sample-centred更新を感度分析として両方掲載した。両実装ともKL内のprior平均はゼロ。
Gaussian閉形式KLは原著にも用いられるため、その点を逸脱としない。
[ARD-VAE原著、Section 3.3](https://arxiv.org/pdf/2501.10901v1)

CIFAR-10のraw countはゼロ中心26.7±0.6、sample中心26.3±0.6。
ただしMAKE51の有限ステップスコア正規化ですでに14.2から26.0へ修正されている。
今回の学習・prior更新条件も異なるため、14.2→26.3の全差を厳密Jacobianだけの
効果とは帰属していない。過去の値と補正値を残し、追加値を別表として示した。

ARDの `eval_full` は64/256幅の未剪定モデルであり、選択されたraw countまで
剪定したnativeモデルの品質ではない。通常VAEへの次元転用と明確に区別した。

### 4. 曲線・費用・重みの範囲

新しい36runの曲線は5 epochごとの60点を確認した。ただし過去の通常VAEの
未保存training曲線を回復したわけではない。ARDの2000点のtraining-pool評価には、
prior更新専用1800点とパラメータ学習用200点が含まれる。

`train_seconds` には定期的な評価が含まれる。ARMの最終評価、ARDのJacobian時間を
別掲したが、ARDの最終native評価時間などは未計測。完全なend-to-end費用とはしない。
追加剪定モデルのcheckpointはコード上保存されておらず、補足ZIPにも含まれない。
MAKE51の別実験のMNIST checkpointは引き続き同梱した。

### 5. ARM単体検証とFONDUE-VARの区別

提供された単体検証を実行し、10万標本で相対誤差0.0085、100万標本で0.0037、
勾配の符号8/8一致を確認した。バッチ平均対照の相関は0.6761。
これは線形損失での推定量の検証であり、学習全体の原著同一性の証明ではない。
FONDUE-VARのゼロ出力は別の退化現象であり、同種の実装バグとして数えていない。

## 再生成・整合性確認

- 100条件の選択一致、ID採用判定・fallback・参照費用の一致を確認。
- 統合JSONと各データJSONの一致、seed集合、全曲線のepochを確認。
- ARMの確率閾値による次元数と、ARDのscore累積99%による次元数を再計算。
- 閾値が同一seedのanchorから得られること、転用先のvalidation/test値がキャッシュと一致することを確認。
- 現行の再集計: `src/make52/summarize_revision.py`。
- TeX組立: `src/make52/build_manuscript.py`。
- 補足ZIP作成・SHA-256検証: `src/make52/package_supplement.py`。

独立参照bankでの追試、全データ・複数seedの妥当なMC ELBO幅選択、原著手順全体の
再現、完全な時間計測、漏洩を除いた合成データ再実行は今回の追加結果で完了したとは
扱っていない。参考文献全30件の外部書誌照合、査読返答書のMAKE52版の作成、
補足資料の外部公開も今回の原稿作成には含めていない。

最終PDFは41ページ。全ページをPNG化して目視確認し、表・図の欠けや重なりが
ないことを確認した。最終LaTeXログに未定義参照・文献警告・Overfullはなく、
タイトルがMAKE47から指定語句のみを削除したものと一致することも自動確認した。
