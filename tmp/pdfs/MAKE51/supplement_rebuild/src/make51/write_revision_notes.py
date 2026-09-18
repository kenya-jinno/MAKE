"""Write revision and reviewer-response notes using the final LaTeX labels."""
from pathlib import Path
import csv, json, re, subprocess

ROOT=Path(__file__).resolve().parents[2]
aux=(ROOT/'main_paper_MAKE51.aux').read_text()
labels={k:(num,page) for k,num,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
def loc(label):
    n,p=labels[label]
    kind={'sec':'Section','app':'Appendix','tab':'Table','fig':'Figure','alg':'Algorithm','eq':'Equation'}[label.split(':')[0]]
    return f'{kind} {n} (p. {p})'
info=subprocess.check_output(['pdfinfo',str(ROOT/'main_paper_MAKE51.pdf')],text=True)
pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
hc=json.loads((ROOT/'results/make51/model_validation/native_hc_s42.json').read_text())
vtext=json.loads((ROOT/'results/make51/validation_text.json').read_text())
with (ROOT/'results/make51/cost_lineage.csv').open() as f:lineage_count=len(list(csv.DictReader(f)))

notes=f"""# MAKE51 修正内容と追加検証

作成日：2026-09-19

MAKE50_revision_review.md を反映し、main_paper_MAKE51.tex と PDF（{pages}ページ）を生成した。MAKE50およびMAKE49の原稿・元の実験記録は変更していない。

## 実施した作業

| 論点 | MAKE51での対応 | 掲載箇所 |
|---|---|---|
| 旧ID手順の構造的費用差 | 固定windowと昇順探索の費用差を式で示し、ID情報の価値を識別した結果とは扱わない | {loc('eq:dominance')} |
| 探索設計 | IDから開始する局所探索、IDなしのmidpoint対照、最小幅を認証する変種を定義。6手順×100条件を再評価 | {loc('alg:replay')}; {loc('tab:replay1')}; {loc('tab:replay2')} |
| ID係数・信頼性閾値 | 5係数×4種類のR閾値×5種類のA閾値×100条件を全件評価。既定値の変更や都合のよい条件の選抜は行わない | {loc('sec:idsensitivity')}; {loc('tab:coefficient')}; {loc('tab:reliability')} |
| 費用の出典 | 同じkeyの価格を単一snapshotに統一。旧値・新値・完全key・出典を{lineage_count}行のCSVに記録 | {loc('app:timing')} |
| 共有参照AEの費用 | 初回利用、構築済みbank利用、25条件への費用配分を分ける | {loc('tab:amortization')} |
| ARDの有限差分 | 保存スコアqと分散pからq/pを再構成し、刻み幅の正規化後の99%基準・転用幅・品質を再評価 | {loc('eq:ardcorrected')}; {loc('tab:ard_normalized')} |
| GECOのnative挙動 | MNIST 1 seed、300 epochsのHC版を追加実行。制約・gate・乗数・native再構成と次元転用を比較 | {loc('fig:hc')}; {loc('tab:native_hc')} |
| 真のELBO | MNIST全12幅を1 seedで新規学習。各proxy選択checkpointでK=32のMC ELBOを評価し、容量選択後にtestを評価 | {loc('sec:newvalidation')}; {loc('tab:mcselection')}; {loc('tab:mcgrid')} |
| 選択モデルの曲線 | 実際の昇順Q選択candidateのvalidation曲線を4データセットで追加。新しいMNIST幅6・64のtrain/validation曲線も保存 | {loc('fig:selectedcurves')}; {loc('fig:newcurves')} |
| FONDUE | 原論文の探索更新式を記録済みgap系列上で独立に再計算。全100経路・返却値の一致を確認 | {loc('app:adaptations')} |
| 補足資料 | 設定、元のJSON、全解析CSV、コード、依存関係、split、今回のcheckpoint、manifestをZIPで提供 | {loc('app:inventory')} |
| 本文・書誌 | Introductionから費用の具体値を移動。AU課題の天井効果、population/sample variance、probe前処理、epoch校正のループを明記。dSprites校正はN/Aに訂正。指定graph論文を追加 | 本文・付録・参考文献 |

## 主な新しい結果

- 既定のID-localは全100条件でfull-grid最小適格幅と一致した。ただし3データセットでは昇順fallbackであり、全条件でIDが有効だったという意味ではない。
- midpoint対照は98/100条件で一致。10,000件の閾値感度では155件が最小適格幅を見逃したが、品質条件は全件達成した。最大の幅の超過は56。これらは独立した10,000実験ではなく、既存記録を使った再評価である。
- MNIST主設定のcanonical費用は旧window 672秒、ID-local 636秒、昇順177秒。旧715対182秒のledger値は監査付録に残し、同じ価格系列とは扱っていない。
- ARDの正規化後、CIFAR-10のraw countは平均14.2から26.0へ変化し、幅24へ転用すると5/5で品質を満たした。ただし有限刻み幅・事前分布更新の違いは残る。
- 新しいMNIST 1 seedではMSE、proxy、MC ELBOの全基準が幅12を選んだ。test MSEは0.020434。1 seedでの一致は基準の一般的同等性を示さない。
- 新しいHC診断では最終制約値は{hc['curves'][-1]['constraint_ma_sse']:+.6f}、gate数{hc['raw_count']}、転用幅{hc['grid_width']}。native評価とordinary VAEの評価を分離した。

## 残る範囲

原著ARMおよび原著ARD全体の再現は完了していない。旧ARD/GECOの結果を主比較から外し、実装監査の付録に置いた。ARDの変更は保存された有限差分の正規化であり、厳密なJacobianや学習更新の原著対応を回復したものではない。HCの追加実行と真のELBO比較はいずれもMNISTの1 seedに限る。

元のMAKE49のcheckpoint、CIFAR-10のepoch別train曲線、完全なID計算時間は復元していない。元の選択候補にはvalidation曲線だけが残っており、train曲線を捏造・補間していない。合成多様体は分割前標準化の旧監査結果を保持し、主結論には使っていない。追加の分割後標準化実験は未実施。

査読者への回答案は MAKE51_response_to_reviewers.md に整理した。一部の比較検証は限定的であり、すべての査読要求が完了したとは記述していない。

## 再生成・配布

MAKE51_reproducibility.zip を展開し、README.md の手順に従う。保存済み数値からの再集計には画像データもPyTorchも不要。新規学習の再実行は別手順とし、公式MNISTデータが必要となる。

TeXを直接再コンパイルする場合は、プロジェクト直下から pdflatex を2回実行する。Definitions/ と results/figures/MAKE51/ が必要。すべての表はTeX本体に埋め込まれている。
"""
(ROOT/'MAKE51_revision_notes.md').write_text(notes)

response=f"""# MAKE51: draft response mapping to the reviewers

Prepared on 19 September 2026 from the comment summaries in MAKE50_revision_review.md. Items below are paraphrases, not quotations of the original letters. Locations refer to the generated MAKE51 PDF ({pages} pages). This file is a draft for the authors; nothing has been submitted or sent.

## Overall change in scope

We have changed the central comparison from a fixed-window expense contrast to a retrospective comparison of explicitly defined search policies. The original contrast is now an accounting audit. No new architecture or universal ID guarantee is claimed. Original V1/V2 activity checks are absent from selection, and AU remains a diagnostic. Modified pruning implementations are separated from the main comparison; their limitations are not presented as completed source-method validation.

## Reviewer 1

| Item | Response and location |
|---|---|
| R1-1: Abstract variables | The revised Abstract states the study and results without relying on undefined mathematical variables. |
| R1-2: Typical choice of widths 32/64 | The unsupported general statement remains deleted; no replacement claim about a universally typical grid has been introduced. |
| R1-3: Forward references | Introduction contains only a short structural guide. |
| R1-4: Detailed conclusions in Introduction | Numerical time contrasts were removed from Introduction. They now appear in {loc('sec:replayresults')} as historical accounting and canonical replay results. |
| R1-5: Definitions | The target, population variance convention, sample SD convention, ID checks, local search, inference unit and cost accounting are stated explicitly in Sections 3–5. |
| R1-6: Simple/graph methods | PCA, Isomap, LLE and graph-based dimension estimation are retained. Gałka et al., DOI 10.1109/ACCESS.2022.3190505, is now cited as graph-based anomaly scoring. We distinguish that objective from VAE width selection; no unrequested anomaly-detector implementation is claimed. |
| R1-7: Proof notation | The earlier proof section and Q.E.D. issue were removed. The remaining theory is background; the new expense identity is tied to stated algorithmic conditions. |
| R1-8: CUDA | {loc('sec:provenance')} reports the historical CUDA 11.8 environment. {loc('sec:newvalidation')} separately identifies the CPU environment for new experiments. |
| R1-9: Splits, repetition, curves and test | Fixed split sizes and five-seed historical results remain. {loc('fig:selectedcurves')} adds actual selected-candidate validation histories, including CIFAR-10. {loc('fig:newcurves')} gives new MNIST train/validation histories. Historical natural-image train histories remain unavailable; no complete recovery is claimed. |
| R1-10: Bibliography | DOI formatting and line wrapping were retained and the requested graph citation was added. DOI-less proceedings entries were supplemented with primary source links where checked. |

## Reviewer 2

**R2-1 and R2-2: Increment beyond NOLTA and contribution.**
{loc('tab:history')} distinguishes the earlier six architectures/four seeds from the present question. {loc('eq:dominance')} states why exhausting a window makes the old cost difference structurally nonnegative. The new contribution is the measured relationship among policy, information acquisition, quality, exact minimum recovery and candidate requests. {loc('alg:replay')} and {loc('tab:replay1')}–{loc('tab:replay2')} compare six policies on 100 recorded conditions. ID-local matches the grid minimum in 100/100 at inherited defaults; the midpoint control matches in 98/100. The full 10,000-row factorial contains 155 nonminimal outputs. All analyses are explicitly retrospective and do not imply a universal policy ranking.

**R2-3: Existing selection/pruning methods.**
The FONDUE description-based implementation remains, with a source-pseudocode control-flow check on all 100 recorded paths. This check does not assert training-framework or published-benchmark parity. The inherited ARD and GECO rows were removed from the main comparison. {loc('eq:ardcorrected')} corrects the finite-step normalization using saved scores and prior variances; {loc('tab:ard_normalized')} gives revised transferred outcomes. {loc('fig:hc')} and {loc('tab:native_hc')} add a newly trained native HC diagnostic and distinguish it from the transferred ordinary VAE.

Source-ARD prior-update reproduction and original ARM benchmarking remain incomplete. The HC experiment is one MNIST seed; the manuscript does not state that this resolves the entire established-method comparison request.

**R2-4: Presentation.**
The final PDF was compiled, rendered and visually inspected. Tables use consistent uncertainty/denominator conventions. Legacy history is identified as audit material; source files, dependencies and actual results are included in the accompanying reproducibility ZIP.

## Reviewer 3

| Item | Response and location |
|---|---|
| R3-1: Factor two and V1/V2 | The factor and reliability cutoffs are internally specified heuristics, not theorem-derived guarantees. {loc('sec:idsensitivity')}, {loc('tab:coefficient')} and {loc('tab:reliability')} report their effects, including decision changes and nonminimal outputs. V1/V2 are not selection branches. |
| R3-2: Additional datasets | All four datasets remain represented, with CIFAR-10 identified as the natural colour-image condition. Rejections and fallback are retained. |
| R3-3: Reference-model dependence | {loc('tab:id')} reports the three-width/three-seed bank, shared-reference limitation and estimator disagreement. Cold, warm and batch reuse costs are separated in {loc('tab:amortization')}. |
| R3-4: Linear versus nonlinear theory | The theory remains background. No claim equates AU counts with topological dimension or transfers a linear guarantee to nonlinear finite-sample reconstruction. |
| R3-5: MSE, ELBO and downstream baseline | The original MSE/proxy/probe results remain in {loc('tab:selection')}. A genuine posterior-sampling ELBO evaluation was added after a new complete MNIST grid at seed 42: {loc('tab:mcselection')} and {loc('tab:mcgrid')}. K=32, evaluation seed and likelihood constant are stated. Capacity is selected among proxy-selected checkpoints, not among ELBO-selected epochs. Full four-dataset/five-seed ELBO replication remains open. |
| R3-6: Repeated conditional/history language | The main claims now concern specified policies and their measurable outputs; detailed prior implementation issues are concentrated in the audit appendices. |
| R3-7: Algorithm and thresholds | {loc('alg:replay')} defines all revised branches. The old unreachable expansion branch is retained only in {loc('alg:legacy')} as a historical fixed-window audit. |

## Corrected AU analysis

The earlier +0.108 value is an archived result affected by probability saturation and ambiguous expanded-cache row extraction. The current score-ranking analysis reports AU-minus-baseline −0.024, with hierarchical-bootstrap 95% interval [−0.102, +0.027]. The paired random-feature comparisons and their limitations are retained. Baseline AUROC values 0.950, 0.980 and 1.000 are stated to make the prediction task's ceiling explicit. Neither equivalence nor absence of AU information is claimed.

## Reproducibility and unfinished work

{loc('app:timing')} and the {lineage_count}-row cost_lineage.csv connect anchor/final keys to the two timing snapshots. Canonical replay does not mix their prices. The accompanying MAKE51_reproducibility.zip contains analysis inputs, outputs, scripts, fixed dependency versions, split indices, new model checkpoints and a verified SHA-256 manifest. Historical checkpoints and missing timing components are not reconstructed. Exact pruning-source replication, independent reference-bank experiments, a broader ELBO study, natural-image train histories and leakage-corrected synthetic retraining remain outside the completed revision.
"""
(ROOT/'MAKE51_response_to_reviewers.md').write_text(response)
print(f'Wrote revision notes and response mapping for {pages} pages.')
