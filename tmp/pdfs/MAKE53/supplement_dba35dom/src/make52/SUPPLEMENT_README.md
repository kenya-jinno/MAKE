# MAKE52 reviewer reproducibility supplement

Revision: 19 September 2026. This archive accompanies the MAKE52 manuscript.
It is a local reviewer supplement, not a public repository or DOI deposit.

## Rebuild the current manuscript

The final TeX embeds all table content. To compile it, keep Definitions/ and
results/figures/ in their relative locations and run from the archive root:

    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE52.tex
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE52.tex
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE52.tex

To regenerate the new numerical summaries and three figures from saved records:

    python src/make52/summarize_revision.py
    python src/make52/build_manuscript.py

These commands require no model retraining or raw images. The verified analysis
environment is Python 3.11.11, NumPy 2.3.2 and Matplotlib 3.10.5. The builder also
reads the supplied MAKE51 TeX, tables and paired-AU summaries. The source templates
are in src/make52/. Direct PDF compilation does not require Python.

## Evidence and provenance

- results/make52/start_width_ablation.json: 100 additional gated-midpoint replays;
  results/make51/search_replay.csv contains the original six policies (600 rows).
- results/make52/geco_arm_*.json: 12 new ARM training runs, three seeds per dataset.
- results/make52/ard_native_*.json: 24 new ARD training runs, three seeds per dataset
  and each of two variance-update centering settings.
- results/make52/manuscript_summary.json: independently aggregated counts and
  record-consistency checks used by the manuscript.
- results/make52/manuscript_source_manifest.json: input and analysis-source hashes.
- results/figures/MAKE52/: the new constraint and reconstruction plots.
- MAKE52_revision_notes.md: integration decisions and corrections to interpretations
  in the supplied MAKE52_実験検証結果.md. The latter is retained unchanged.
- src/make52/geco_arm.py, ard_native.py, start_width_ablation.py and
  test_arm_estimator.py: supplied training/control/test scripts, unchanged.
- src/make51/SUPPLEMENT_README.md: historical MAKE51 archive documentation. Its
  descriptions of remaining work refer to that revision, not all current evidence.
- results/make49/, results/make51/, src/make49/, src/make51/, src/metrics/ and config/:
  inherited evidence, implementations, analysis and settings.

The supplied new training provenance records Python 3.12.3, PyTorch 2.7.1+cu118,
CUDA 11.8, cuDNN 90100, NVIDIA driver 570.211.01 and GeForce RTX 3070. The current
reaggregation does not rerun those training experiments. The analytic ARM test
can be run separately with PyTorch:

    python src/make52/test_arm_estimator.py

Running a training script is a separate, optional operation, requires the raw
datasets and training dependencies, and can overwrite result files or extend
the candidate cache. Use a separate extracted copy. The start-width script imports
the historical replay module, which also writes historical replay outputs.
The current aggregation command above instead reads the delivered results only.

## Measurement limits

The new native elbo/val_elbo fields use posterior-mean reconstruction and a
standard-normal KL; they are not MC ELBOs for the native gated/ARD models.
The 32-sample option samples MSE only. Valid capacity-selection MC ELBO evidence
is still the separate, one-seed MAKE51 MNIST study, whose checkpoints are included.

New ARM native MSE uses binary gates; ARD native/full MSE evaluates an unpruned
model. Transfer metrics belong to cached ordinary VAEs at mapped grid widths.
Training-loop timings include periodic evaluation and are not complete selector
costs. New pruning checkpoints and historical MAKE49 checkpoints are unavailable.
Independent reference-bank replication and leakage-free synthetic reruns were
not performed. Original comments and summary claims are retained as provenance;
the MAKE52 manuscript and revision notes state the audited interpretation.

## Integrity

Every payload file is listed in MANIFEST.sha256. After extraction:

    shasum -a 256 -c MANIFEST.sha256

The hashes identify this delivered snapshot. PDF metadata and rendering-library
versions can change bytes in later rebuilds without changing numerical results.
