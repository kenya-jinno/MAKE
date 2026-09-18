# MAKE50 改訂内容と残る検証事項

作成日：2026-09-18  
対象：`main_paper_MAKE49.tex`／`main_paper_MAKE49.pdf`  
修正指針：`MAKE49_manuscript_review.md`  
生成物：`main_paper_MAKE50.tex`、`main_paper_MAKE50.pdf`（26ページ）

レビューの指摘に対し、本文の修正だけでなく、保存済みの設定、実験コード、JSON記録を照合し、表・図と予測解析を再作成した。新しいVAE学習は実施していない。元のMAKE49原稿・実験記録は変更していない。

## 主要な対応

| レビューの論点 | MAKE50での対応 | 主な掲載箇所 |
|---|---|---|
| β感度と判定閾値の混同 | βによる学習・品質目標の変化と、同一checkpointのAU閾値変更を区別。βの5値と変動幅の定義を明記 | §6.3、Table 7、Figures 3・6 |
| ARDの平坦な結果 | 選択段階ではβを受け取らず、同じ設定が繰り返されていたことを確認。「感度ゼロ」の順位づけから除外。原典のGaussian近似・99%基準を訂正し、局所実装の関連度計算との差も明記 | §2.1、§6.3、Appendix B.1 |
| 感度順位・「2桁」の過大な主張 | データセット間の順位逆転を記述し、普遍的順位や桁数の主張を削除 | Abstract、Introduction、§6.3、Discussion、Conclusions |
| AUと無作為特徴量の比較 | 同じdataset–seedブロックを対応させた差を算出。100通りの無作為特徴量も比較。確率の数値飽和による順位消失を確認し、decision scoreでAUROCを再計算 | §6.5、Table 10、Table A8 |
| 実験仕様不足 | 分割件数、seed、モデル構造、optimizer、尤度、停止条件、グリッド、環境情報、ID設定を明記 | §5、Table 2、Appendix B |
| 疑似コード不足 | anchor、参照AE、信頼性判定、候補順、fallback、予算、退化出力まで記載。実装上到達しない拡張分岐も開示 | Algorithm 1、§3.4 |
| 品質と費用の解釈 | seed別anchor由来の実際の品質条件を明記し、品質達成率・test MSE・SDを追加。欠けていたanchor時間を補い、未計測のID計算等と区別 | Tables 3–5、Table A1 |
| 終了状態と次元0 | fallback、予算超過、品質達成、圧縮を別々に集計。次元0に存在しないモデルのMSEを割り当てない | §3.3、Table 11、Tables A2–A3 |
| 最終モデルの定義 | 生の残存座標数、グリッドに丸めた幅、評価した通常VAEを区別。剪定モデルそのものの比較ではないことを明記 | §1、§5.2、Table A7 |
| Fashion-MNISTの欠落 | 4データセット×16手法×5設定×5seedの1,600記録を確認。主結果・fallback件数を全データセットで提示 | Tables 3–4、A2–A4 |
| 学習曲線・選択基準・readout結果 | 保存済みのtrain/validation曲線、ELBO proxy／probeによる選択、最終test性能、AU／変数型の軌跡とSDを掲載 | §6.4–6.5、Tables 8–9、Figures 4–5 |
| NOLTA・MAKE47との関係 | NOLTAの4 FC・2 CNN構造と4 seeds、MAKE47に既存の環境記述があったことを訂正。3段階の差分を整理 | §2.2、Table 1、§5.3 |
| 実装問題の分類 | 2件のスケール不整合とFONDUE-VARの退化出力を分離。「1.7 units以内」の裏付け不足の主張を削除 | §6.6 |
| 理論・合成多様体 | Whitneyの直接出典と線形Gaussianモデルの出典を整理。AUと位相を同一視しない。除去前後MSEを追加し、分割前標準化による漏洩も開示 | §4、Appendix C、Table A11 |
| 事前指定・再現性 | 内部指定と公開事前登録を区別。設定変更・今回の事後解析を明示。分割indexを規則から再構成し、出典ファイルのSHA-256と生成手順を保存 | §5.3、Appendix D |
| 図表・参考文献 | 6図を再生成し、パネル配置・凡例・表示名を統一。Fuらの著者・DOI、NOLTA書誌等を訂正し、URLのはみ出しを修正 | 全体 |

## 再解析によって変わった重要な解釈

- MNIST主設定のID誘導715秒対昇順走査182秒という結果は維持される。ただし費用は保存済みの学習時間等を帰属させた値であり、完全なwall-clock測定ではない。他手法では欠落していたanchor時間を追加したため、MAKE49と費用が異なる。
- ID信頼性判定はMNISTだけで通過し、Fashion-MNIST、dSprites、CIFAR-10はfallbackとなる。参照AE群は共有されており、5回の独立な全手順再現とは扱わない。
- ARDの選択段階は全β文脈で固定重み1だった。加えて、保存コードの有限差分スコアは原典のJacobianに基づく関連度と同一ではない。結果は「ARD adaptation」として限定した。
- AU解析はβ=1と宣言済みグリッドに限定して再構成した。AU追加対基底の平均AUROC差は−0.024（95%区間−0.102～+0.027）。AU対1回の無作為特徴量対照は+0.026（−0.040～+0.133）、100回の対照平均に対しては−0.007（−0.064～+0.039）。従来の正の改善値は原解析として残すが、検証済みの正の効果とは位置づけていない。
- 上記区間はdatasetとseedブロックを再標本化する探索的な階層bootstrapで、開発側のモデル再学習の不確実性は含まない。AUの無情報性や対照との同等性は主張しない。
- 合成多様体記録では標準化が分割前に行われていたため、汎化性能の証拠として扱わず、尺度・座標除去に関する記述的実験に限定した。

## 未実施・復元できない事項

以下は本文の限界として明記した。追加実験を実施したような記述にはしていない。

1. ARDの選択重みを正しく変え、原典に対応した関連度計算で学習し直す比較。
2. 原典のARMを用いるGECOと、各剪定法が返すモデル自体の性能比較。
3. βを変えても品質目標を固定する介入、およびFONDUE停止閾値・ARD 99%基準の感度評価。
4. ID参照群の再学習を含む独立反復、完全なID計算・評価費用、未保存checkpointや元のtest予測の復元。
5. 到達不能なID探索拡張分岐を修正した後の新規実験。
6. 未実施のMNIST-CNN予測条件、合成多様体の分割後標準化による再実験。

今回の指定成果物は英語版MAKE50であり、日本語版NN49と査読回答書は更新していない。

## 生成と確認

- 数値・図の生成：`src/make50/rebuild_review.py`
- 本文・表の組み立て：`src/make50/build_manuscript.py`
- 監査済み1,600記録：`results/make50/method_results.csv`
- 解析結果と出典hash：`results/make50/`
- PDFが参照する図：`results/figures/MAKE50/`

プロジェクト直下で、NumPy・SciPy・Matplotlib・scikit-learnを備えたPythonを使用する。

```sh
python src/make50/rebuild_review.py
python src/make50/build_manuscript.py
pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE50.tex
pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE50.tex
```

生成済みのTeXには表を埋め込んでいるため、通常の再コンパイルではPython処理は不要。既存の`Definitions/`と上記図PDFが必要になる。PDF全26ページを画像化して体裁を確認し、未解決参照・欠落引用・はみ出し警告がないことを確認した。
