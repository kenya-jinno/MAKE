# MAKE53 reviewer reproducibility supplement

This fixed snapshot accompanies the English MAKE53 manuscript and its Japanese
NN53 translation (19 September 2026). It is not a public deposit or preregistration.
The supplied experiment report and earlier revision files are preserved as
provenance; MAKE53_revision_notes.md records the audited interpretation.

## Compile the delivered manuscripts

Run from the extracted archive root, retaining Definitions/ and results/figures/:

    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE53.tex
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE53.tex
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE53.tex
    lualatex -interaction=nonstopmode -halt-on-error main_paper_NN53.tex
    lualatex -interaction=nonstopmode -halt-on-error main_paper_NN53.tex

Japanese compilation requires LuaTeX, luatexja, Harano Aji fonts, TeX Gyre Pagella,
TeX Gyre Heros and Latin Modern Mono, available in the verified TeX Live 2026 setup.
If the default LuaTeX font cache is not writable, create a local directory and set
both TEXMFVAR and TEXMFCACHE to that directory's absolute path.
The delivered TeX files embed all table contents; Python is not needed to compile.

## Reaggregate saved numerical evidence (no training or raw images)

The analysis environment used here is Python 3.11.11, NumPy 2.3.2, Matplotlib
3.10.5, SciPy 1.16.1 and scikit-learn 1.7.1. Use a fresh extracted copy because
these commands rewrite derived tables, summaries and figures:

    python src/make51/rebuild_review.py
    python src/make51/replay_search.py
    python src/make51/summarize_validation.py
    python src/make52/summarize_revision.py
    python src/make53/summarize_native_pruning.py
    python src/make53/build_manuscript.py
    python src/nn53/assemble_blocks.py
    python src/nn53/render_figures.py
    python src/nn53/build_japanese.py

The first script rebuilds historical tables and the paired AU analysis; the second
replays search policies and all 10,000 coefficient/threshold evaluations.
The third summarizes the saved, one-seed MNIST MC ELBO records without retraining.
The fourth checks and summarizes the gated-start control and initial ARM/ARD runs.
The fifth independently aggregates all 12 native-masking records and recomputes
the four-condition quality-target counts. It does not use raw ELBO fields.
The builders check the Japanese translation's equations, numerical table cells,
labels, citation occurrences and bibliography against English.
Figure data did not change in MAKE53; the Japanese figure script copies the
previously localized NN52 PDFs, which are included.

## Source-to-output map

| Evidence | File(s) |
|---|---|
| Configuration | config/experiment_config_MAKE49.json |
| Historical ordinary method records | results/make49/stage4_methods.json |
| Historical pruning records | results/make49/stage7_b7b8.json |
| Candidate metrics and latent variances | results/make49/stage4_cache/*.json |
| Train/validation and readout trajectories | results/make49/stage3_e7a_e2.json |
| Representation diagnostics | results/make49/stage5_e3_reference.json |
| Original AU results | results/make49/stage5_e4_au_value.json |
| Synthetic diagnostics | results/make49/stage6_e6_synthetic.json |
| Audited method rows | results/make51/method_results.csv |
| Paired AU summaries and rows | results/make51/au_paired.json; results/make51/au_predictions.csv |
| Search and sensitivity | results/make51/search_replay.csv; results/make51/search_sensitivity.csv |
| Timing lineage | results/make51/cost_lineage.csv |
| Saved MNIST models and MC ELBO | results/make51/model_validation/ |
| Gated-start ablation | results/make52/start_width_ablation.json |
| Initial ARM and ARD records | results/make52/geco_arm_all.json; results/make52/ard_native_all.json |
| Native-masking specification and records | results/make52/native_pruned_spec.json; results/make52/ard_pruned_all.json; matching per-dataset JSON files |
| Audited native-pruning tables and summaries | results/make53/native_pruning_*.tex; results/make53/native_pruning_summary.json |
| Native-pruning source identities | results/make53/native_pruning_manifest.json |
| Supplied training implementation | src/make52/ard_native_pruned.py, ard_native.py, geco_arm.py |
| Current manuscript source | src/make53/; src/nn53/ |
| English and Japanese figures | results/figures/MAKE51/; results/figures/MAKE52/; results/figures/NN53/ |

## Interpretation and missing artifacts

- Native-pruning JSON now supplies posterior-mean MSE for full, masked and
  decoder-adapted models, and transfer metrics from cached ordinary VAEs.
  The preceding ard_native_* files evaluate full-width models only.
- The 12 masking evaluations repeat the same sample-centred configurations and
  seeds as earlier ARD runs. Their full-model validation/test MSE and counts
  match exactly. Do not count the repeated baseline as independent replication.
- The mask fixes nonselected coordinates to zero without shrinking the network.
  Decoder adaptation uses all 18,000 training-pool examples, including the 1800
  reserved during initial training. There is no unmasked adaptation control.
- The raw elbo/val_elbo fields in all additional pruning records are NOT native
  Monte Carlo ELBOs. They use posterior-mean likelihood and an unmasked
  standard-normal KL; K=32 changes MSE only. These fields must not enter ELBO
  comparisons. The native_pruned_spec.json request for true ELBO was not met.
- Valid MC ELBO capacity-selection evidence remains the separate MNIST seed 42
  study at proxy-selected checkpoints. Its weights are included.
- The supplied new scripts and specification define 300 training epochs and
  30 decoder-adaptation epochs. Native-masking JSON has no epoch trajectory,
  mask identities, relevance vectors or saved checkpoints. It is impossible
  to independently reevaluate these native models using summary JSON alone.
  The initial ARM/ARD curve records are included separately.
- Native-pruning timing covers the training interval and decoder loop only,
  not the Jacobian, mask, final evaluations, anchor or transferred model costs.
- The sample-centred variance update is a local sensitivity variant, not a
  necessary correction of the ARD source's zero-centre setting.
- Historical MAKE49 checkpoints, independent reference-bank replications,
  complete end-to-end timing and leakage-free synthetic reruns remain unavailable.

The training provenance is Python 3.12.3, PyTorch 2.7.1+cu118, CUDA 11.8, cuDNN
90100, NVIDIA driver 570.211.01 and GeForce RTX 3070 on Linux
6.8.0-124-generic x86-64 (glibc 2.39), matching the main historical GPU environment.
Reaggregation does not rerun those experiments. Training scripts are optional,
require raw datasets and their training environment, and can overwrite records
or extend caches; use a separate extracted copy if rerunning them.

## Integrity

Every delivered payload file is identified by MANIFEST.sha256. After extraction:

    shasum -a 256 -c MANIFEST.sha256

Use the manifest before regeneration. Later PDF/figure metadata can change bytes
without changing numerical results. Reaggregation verification for the delivered
snapshot is documented in MAKE53_revision_notes.md.
