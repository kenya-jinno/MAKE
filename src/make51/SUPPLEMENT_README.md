# MAKE51 reviewer reproducibility supplement

Fixed revision: 19 September 2026.

This archive accompanies the MAKE51 manuscript. It contains actual analysis
inputs and outputs, not only file names. It has not been uploaded to a public
repository and is not assigned a DOI. Paths below are relative to the extracted
archive root. Verify the payload with:

    shasum -a 256 -c MANIFEST.sha256

## Contents and provenance

- config/experiment_config_MAKE49.json: historical, internally dated configuration;
  its original preregistration wording is retained as source material, not endorsed
  as a public registration.
- results/make49/: archived method-level results, candidate metric/curve cache and
  diagnostic records. No historical MAKE49 network checkpoints are available.
- src/make49/ and src/metrics/: inspected historical code. Its old comments may
  contain claims corrected in the manuscript; these files are unchanged audit inputs.
- src/make51/: new reaggregation, replay, model-validation, assembly and packaging code.
- results/make51/: generated tables, 1,600 historical method rows, 600 policy replay
  rows, 10,000 sensitivity rows, full anchor/final timing lineage, corrected finite-step
  ARD scores, FONDUE control-flow checks, paired AU analysis, reconstructed split indices,
  specifications and source hashes.
- results/make51/model_validation/: new MNIST seed-42 records, saved ordinary-VAE and
  HC model weights, training/validation curves, Monte Carlo draw means, validation
  capacity decisions and final test summaries. These are new CPU experiments, not
  recovered CUDA runs.
- results/figures/MAKE51/: manuscript figure PDFs.
- main_paper_MAKE51.tex, main_paper_MAKE51.pdf, and Definitions/: manuscript and
  local MDPI class assets. main_paper_MAKE49.tex is the assembly input for inherited
  author information and bibliography.
- MAKE51_revision_notes.md and MAKE51_response_to_reviewers.md: change record and
  draft reviewer-response mapping. They identify work that remains incomplete.

## Regenerate the analysis and manuscript

Python 3.11.11 was used. Install src/make51/requirements-analysis.txt into an
appropriate environment. A TeX distribution with pdflatex, the packages required
by the MDPI class and xurl is needed for PDF compilation. Raw images, network
access and PyTorch are not needed for this reaggregation:

    python src/make51/rebuild_review.py
    python src/make51/replay_search.py
    python src/make51/summarize_validation.py
    python src/make51/build_manuscript.py
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE51.tex
    pdflatex -interaction=nonstopmode -halt-on-error main_paper_MAKE51.tex

Generated table content is embedded in the final TeX, so direct recompilation needs
only the TeX source, Definitions/ and the figure PDFs. Python is required only to
regenerate numerical results or assemble the source.

Figure PDFs and compiled PDFs may differ byte-for-byte due to metadata or rendering
versions. Numerical equality is the appropriate reaggregation check. The ZIP's
manifest identifies the delivered snapshot, not a promise of bitwise future builds.

## Optional new model validation

The completed records and checkpoints are included. Repeating training is optional,
requires the additional validation dependencies and takes materially longer.
The script uses CPU with four threads and does not require CUDA.

The official MNIST raw files must be available under data/MNIST/raw/, in the
torchvision MNIST layout. Images are not included in this supplement.
The script sets download=False; it never silently downloads data.
The fixed permutation and split rule are recorded in the script and specification.

    python src/make51/validate_models.py

Existing completed per-width records are reused. To conduct an independent training
repeat, extract a separate copy of the archive and preserve its delivered outputs
before running the script:

    mv results/make51/model_validation results/make51/model_validation_delivered
    python src/make51/validate_models.py

The new grid includes all twelve MNIST widths at one training
seed and weight one. Checkpoints are selected by the posterior-mean proxy, then
evaluated with 32 posterior samples for Monte Carlo ELBO. Capacity choices are saved
before test evaluation. The HC run is a local-implementation diagnostic at one seed;
it is not an ARM/source-method reproduction.

## Interpretive limits

The search replay uses stored validation outcomes and one canonical timing price
per logical key. It is retrospective, not a prospective training-time measurement.
Test summaries are attached only after replay decisions. The reference bank is shared
across candidate seeds and weights. Sensitivity rows are not independent trials.

The ARD correction divides each archived relevance score by its positive prior
variance, exactly undoing an extra finite-step factor. It does not recover an
infinitesimal Jacobian or change prior training. Missing historical model weights,
native pruning evaluation trajectories, original row-level AU probabilities and
complete ID/evaluation timings cannot be recovered from aggregate JSON.

New MC ELBO and native HC results cover one MNIST seed. No new CIFAR-10 training
curve or full four-dataset source-method benchmark is claimed. Historical synthetic
ablation records retain the disclosed preprocessing leakage and are descriptive only.
