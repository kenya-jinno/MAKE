# MAKE51: draft response mapping to the reviewers

Prepared on 19 September 2026 from the comment summaries in MAKE50_revision_review.md. Items below are paraphrases, not quotations of the original letters. Locations refer to the generated MAKE51 PDF (34 pages). This file is a draft for the authors; nothing has been submitted or sent.

## Overall change in scope

We have changed the central comparison from a fixed-window expense contrast to a retrospective comparison of explicitly defined search policies. The original contrast is now an accounting audit. No new architecture or universal ID guarantee is claimed. Original V1/V2 activity checks are absent from selection, and AU remains a diagnostic. Modified pruning implementations are separated from the main comparison; their limitations are not presented as completed source-method validation.

## Reviewer 1

| Item | Response and location |
|---|---|
| R1-1: Abstract variables | The revised Abstract states the study and results without relying on undefined mathematical variables. |
| R1-2: Typical choice of widths 32/64 | The unsupported general statement remains deleted; no replacement claim about a universally typical grid has been introduced. |
| R1-3: Forward references | Introduction contains only a short structural guide. |
| R1-4: Detailed conclusions in Introduction | Numerical time contrasts were removed from Introduction. They now appear in Section 6.1 (p. 9) as historical accounting and canonical replay results. |
| R1-5: Definitions | The target, population variance convention, sample SD convention, ID checks, local search, inference unit and cost accounting are stated explicitly in Sections 3–5. |
| R1-6: Simple/graph methods | PCA, Isomap, LLE and graph-based dimension estimation are retained. Gałka et al., DOI 10.1109/ACCESS.2022.3190505, is now cited as graph-based anomaly scoring. We distinguish that objective from VAE width selection; no unrequested anomaly-detector implementation is claimed. |
| R1-7: Proof notation | The earlier proof section and Q.E.D. issue were removed. The remaining theory is background; the new expense identity is tied to stated algorithmic conditions. |
| R1-8: CUDA | Section 5.3 (p. 8) reports the historical CUDA 11.8 environment. Section 5.4 (p. 8) separately identifies the CPU environment for new experiments. |
| R1-9: Splits, repetition, curves and test | Fixed split sizes and five-seed historical results remain. Figure 3 (p. 14) adds actual selected-candidate validation histories, including CIFAR-10. Figure A2 (p. 32) gives new MNIST train/validation histories. Historical natural-image train histories remain unavailable; no complete recovery is claimed. |
| R1-10: Bibliography | DOI formatting and line wrapping were retained and the requested graph citation was added. DOI-less proceedings entries were supplemented with primary source links where checked. |

## Reviewer 2

**R2-1 and R2-2: Increment beyond NOLTA and contribution.**
Table 1 (p. 3) distinguishes the earlier six architectures/four seeds from the present question. Equation 6 (p. 9) states why exhausting a window makes the old cost difference structurally nonnegative. The new contribution is the measured relationship among policy, information acquisition, quality, exact minimum recovery and candidate requests. Algorithm 1 (p. 5) and Table 3 (p. 9)–Table 4 (p. 10) compare six policies on 100 recorded conditions. ID-local matches the grid minimum in 100/100 at inherited defaults; the midpoint control matches in 98/100. The full 10,000-row factorial contains 155 nonminimal outputs. All analyses are explicitly retrospective and do not imply a universal policy ranking.

**R2-3: Existing selection/pruning methods.**
The FONDUE description-based implementation remains, with a source-pseudocode control-flow check on all 100 recorded paths. This check does not assert training-framework or published-benchmark parity. The inherited ARD and GECO rows were removed from the main comparison. Equation A2 (p. 29) corrects the finite-step normalization using saved scores and prior variances; Table A13 (p. 29) gives revised transferred outcomes. Figure A1 (p. 28) and Table A12 (p. 28) add a newly trained native HC diagnostic and distinguish it from the transferred ordinary VAE.

Source-ARD prior-update reproduction and original ARM benchmarking remain incomplete. The HC experiment is one MNIST seed; the manuscript does not state that this resolves the entire established-method comparison request.

**R2-4: Presentation.**
The final PDF was compiled, rendered and visually inspected. Tables use consistent uncertainty/denominator conventions. Legacy history is identified as audit material; source files, dependencies and actual results are included in the accompanying reproducibility ZIP.

## Reviewer 3

| Item | Response and location |
|---|---|
| R3-1: Factor two and V1/V2 | The factor and reliability cutoffs are internally specified heuristics, not theorem-derived guarantees. Section 6.2 (p. 10), Table A4 (p. 23) and Table A5 (p. 24) report their effects, including decision changes and nonminimal outputs. V1/V2 are not selection branches. |
| R3-2: Additional datasets | All four datasets remain represented, with CIFAR-10 identified as the natural colour-image condition. Rejections and fallback are retained. |
| R3-3: Reference-model dependence | Table 8 (p. 12) reports the three-width/three-seed bank, shared-reference limitation and estimator disagreement. Cold, warm and batch reuse costs are separated in Table 5 (p. 11). |
| R3-4: Linear versus nonlinear theory | The theory remains background. No claim equates AU counts with topological dimension or transfers a linear guarantee to nonlinear finite-sample reconstruction. |
| R3-5: MSE, ELBO and downstream baseline | The original MSE/proxy/probe results remain in Table 10 (p. 16). A genuine posterior-sampling ELBO evaluation was added after a new complete MNIST grid at seed 42: Table 11 (p. 16) and Table A18 (p. 31). K=32, evaluation seed and likelihood constant are stated. Capacity is selected among proxy-selected checkpoints, not among ELBO-selected epochs. Full four-dataset/five-seed ELBO replication remains open. |
| R3-6: Repeated conditional/history language | The main claims now concern specified policies and their measurable outputs; detailed prior implementation issues are concentrated in the audit appendices. |
| R3-7: Algorithm and thresholds | Algorithm 1 (p. 5) defines all revised branches. The old unreachable expansion branch is retained only in Algorithm A1 (p. 27) as a historical fixed-window audit. |

## Corrected AU analysis

The earlier +0.108 value is an archived result affected by probability saturation and ambiguous expanded-cache row extraction. The current score-ranking analysis reports AU-minus-baseline −0.024, with hierarchical-bootstrap 95% interval [−0.102, +0.027]. The paired random-feature comparisons and their limitations are retained. Baseline AUROC values 0.950, 0.980 and 1.000 are stated to make the prediction task's ceiling explicit. Neither equivalence nor absence of AU information is claimed.

## Reproducibility and unfinished work

Appendix B.1 (p. 24) and the 3159-row cost_lineage.csv connect anchor/final keys to the two timing snapshots. Canonical replay does not mix their prices. The accompanying MAKE51_reproducibility.zip contains analysis inputs, outputs, scripts, fixed dependency versions, split indices, new model checkpoints and a verified SHA-256 manifest. Historical checkpoints and missing timing components are not reconstructed. Exact pruning-source replication, independent reference-bank experiments, a broader ELBO study, natural-image train histories and leakage-corrected synthetic retraining remain outside the completed revision.
