# MAKE51 修正内容と追加検証

作成日：2026-09-19

MAKE50_revision_review.md を反映し、main_paper_MAKE51.tex と PDF（34ページ）を生成した。MAKE50およびMAKE49の原稿・元の実験記録は変更していない。

## 実施した作業

| 論点 | MAKE51での対応 | 掲載箇所 |
|---|---|---|
| 旧ID手順の構造的費用差 | 固定windowと昇順探索の費用差を式で示し、ID情報の価値を識別した結果とは扱わない | Equation 6 (p. 9) |
| 探索設計 | IDから開始する局所探索、IDなしのmidpoint対照、最小幅を認証する変種を定義。6手順×100条件を再評価 | Algorithm 1 (p. 5); Table 3 (p. 9); Table 4 (p. 10) |
| ID係数・信頼性閾値 | 5係数×4種類のR閾値×5種類のA閾値×100条件を全件評価。既定値の変更や都合のよい条件の選抜は行わない | Section 6.2 (p. 10); Table A4 (p. 23); Table A5 (p. 24) |
| 費用の出典 | 同じkeyの価格を単一snapshotに統一。旧値・新値・完全key・出典を3159行のCSVに記録 | Appendix B.1 (p. 24) |
| 共有参照AEの費用 | 初回利用、構築済みbank利用、25条件への費用配分を分ける | Table 5 (p. 11) |
| ARDの有限差分 | 保存スコアqと分散pからq/pを再構成し、刻み幅の正規化後の99%基準・転用幅・品質を再評価 | Equation A2 (p. 29); Table A13 (p. 29) |
| GECOのnative挙動 | MNIST 1 seed、300 epochsのHC版を追加実行。制約・gate・乗数・native再構成と次元転用を比較 | Figure A1 (p. 28); Table A12 (p. 28) |
| 真のELBO | MNIST全12幅を1 seedで新規学習。各proxy選択checkpointでK=32のMC ELBOを評価し、容量選択後にtestを評価 | Section 5.4 (p. 8); Table 11 (p. 16); Table A18 (p. 31) |
| 選択モデルの曲線 | 実際の昇順Q選択candidateのvalidation曲線を4データセットで追加。新しいMNIST幅6・64のtrain/validation曲線も保存 | Figure 3 (p. 14); Figure A2 (p. 32) |
| FONDUE | 原論文の探索更新式を記録済みgap系列上で独立に再計算。全100経路・返却値の一致を確認 | Appendix C.2 (p. 27) |
| 補足資料 | 設定、元のJSON、全解析CSV、コード、依存関係、split、今回のcheckpoint、manifestをZIPで提供 | Appendix F (p. 32) |
| 本文・書誌 | Introductionから費用の具体値を移動。AU課題の天井効果、population/sample variance、probe前処理、epoch校正のループを明記。dSprites校正はN/Aに訂正。指定graph論文を追加 | 本文・付録・参考文献 |

## 主な新しい結果

- 既定のID-localは全100条件でfull-grid最小適格幅と一致した。ただし3データセットでは昇順fallbackであり、全条件でIDが有効だったという意味ではない。
- midpoint対照は98/100条件で一致。10,000件の閾値感度では155件が最小適格幅を見逃したが、品質条件は全件達成した。最大の幅の超過は56。これらは独立した10,000実験ではなく、既存記録を使った再評価である。
- MNIST主設定のcanonical費用は旧window 672秒、ID-local 636秒、昇順177秒。旧715対182秒のledger値は監査付録に残し、同じ価格系列とは扱っていない。
- ARDの正規化後、CIFAR-10のraw countは平均14.2から26.0へ変化し、幅24へ転用すると5/5で品質を満たした。ただし有限刻み幅・事前分布更新の違いは残る。
- 新しいMNIST 1 seedではMSE、proxy、MC ELBOの全基準が幅12を選んだ。test MSEは0.020434。1 seedでの一致は基準の一般的同等性を示さない。
- 新しいHC診断では最終制約値は+0.044995、gate数7、転用幅6。native評価とordinary VAEの評価を分離した。

## 残る範囲

原著ARMおよび原著ARD全体の再現は完了していない。旧ARD/GECOの結果を主比較から外し、実装監査の付録に置いた。ARDの変更は保存された有限差分の正規化であり、厳密なJacobianや学習更新の原著対応を回復したものではない。HCの追加実行と真のELBO比較はいずれもMNISTの1 seedに限る。

元のMAKE49のcheckpoint、CIFAR-10のepoch別train曲線、完全なID計算時間は復元していない。元の選択候補にはvalidation曲線だけが残っており、train曲線を捏造・補間していない。合成多様体は分割前標準化の旧監査結果を保持し、主結論には使っていない。追加の分割後標準化実験は未実施。

査読者への回答案は MAKE51_response_to_reviewers.md に整理した。一部の比較検証は限定的であり、すべての査読要求が完了したとは記述していない。

## 再生成・配布

MAKE51_reproducibility.zip を展開し、README.md の手順に従う。保存済み数値からの再集計には画像データもPyTorchも不要。新規学習の再実行は別手順とし、公式MNISTデータが必要となる。

TeXを直接再コンパイルする場合は、プロジェクト直下から pdflatex を2回実行する。Definitions/ と results/figures/MAKE51/ が必要。すべての表はTeX本体に埋め込まれている。

## 出力の検証

最終PDFの全34ページを画像化して確認した。未定義の引用・参照、組版のはみ出し警告、未置換の埋込み記号はない。補足ZIPを別のフォルダに展開し、画像データやネットワークを使わずに再集計・再組版を実行したところ、TeXおよび58個の数値・表ファイルが元の出力と完全に一致した。

新たに保存した幅6・12・64とnative HCのcheckpointからvalidation MSEを再評価し、幅12では32個のMC ELBO draw値も再計算して保存値との一致を確認した。詳細は results/make51/build_verification.json に記録した。
