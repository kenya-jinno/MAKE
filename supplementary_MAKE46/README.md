# Supplementary Materials

**Paper**: Intrinsic-Dimension-Guided Bottleneck Selection for Autoencoders and
Variational Autoencoders with Active-Unit Diagnostics
(submitted to MDPI *Machine Learning and Knowledge Extraction*)

This archive contains the code, configuration, and numerical logs supporting
all tables and figures of the paper. Upon acceptance, these materials will be
released in a public GitHub repository archived with a Zenodo DOI.

## Contents

```
code/src/
├── data/          # dataset loaders (MNIST, Fashion-MNIST, CIFAR-10, SVHN)
│                  #   and synthetic-manifold generators (Swiss Roll, Torus,
│                  #   Möbius band, S^1)
├── models/        # AE, VAE, CAE, DAE, IsometricAE implementations
├── metrics/       # TwoNN, MLE, AU, CKA, Trustworthiness, Jacobian analysis
└── experiments/   # one script per experiment (50 scripts), including the
                   #   figure-regeneration scripts (replot_*.py) that
                   #   reproduce the paper figures from logs/ without
                   #   retraining, and the V1/V2 threshold-sensitivity
                   #   analysis (Exp_AU_Rule.py) and subsampling-CI
                   #   analysis (Exp_Boot_CI.py)

code/analysis/     # table/figure regeneration from the logs:
                   #   regenerate_figures.py, update_tables_with_multiseed.py,
                   #   update_en_table.py, run helpers

config/            # experiment_config.json: shared fixed configuration
                   #   (seeds, data-subset indices, optimizer, architectures,
                   #   ID-estimation parameters, KL-annealing schedule)

logs/results_tables/   # 47 JSON/CSV files: seed-specific training logs and
                       #   the numerical results behind every table/figure
```

## Mapping: paper experiments → scripts → logs

| Paper name | Script (code/src/experiments/) | Log (logs/results_tables/) |
|---|---|---|
| Exp-Syn-1 / Exp-Syn-AU | E1_synthetic.py / EA_au_saturation.py | E1_*.json, EA_*.json |
| Exp-Syn-Verify / Exp-Syn-Top | EB_topology.py | EB_topology.json |
| MNIST Step 1 (reference-AE sweep) | E4_extended.py | E4_extended.json |
| Exp-Mnist-MS (3-seed verification) | E4_multiseed.py / EX_mnist_multiseed.py | E4_300ep_multiseed.json, EX_mnist_multiseed.json |
| Exp-Mnist-Dense | EL_dense_grid_mknee.py | EL_dense_grid_mknee.json |
| Exp-Boot-CI (subsampling CIs) | EP_twonn_stability.py | Exp_Boot_CI.json |
| Exp-Downstream (linear probe, run accounting for Table 10) | (analysis in) Exp_Downstream | Exp_Downstream.json |
| Exp-Real-Geom (AE/IsoAE/CAE/DAE) | EC_isometric_ae.py / ED_model_comparison.py / Exp_Iso_ID.py | EC_*.json, ED_*.json, Exp_Iso_ID.json |
| Exp-Fashion-MS | EF/Exp_Fashion_MS | Exp_Fashion_MS.json, EF_fashion_mnist.json |
| Exp-Conv-M256 / M512 | EN_conv_vae_extended_m.py / EN_extended_m512.py | EN_conv_vae_extended_m.json, EN_ext_m512.json |
| Exp-ConvV-HiBeta(-MS) | Exp_ConvV_HighBeta.py | Exp_ConvV_HiBeta*.json |
| Exp-Nat-Cifar / Exp-Nat-SVHN | EI_cifar10_conv_vae.py / EJ_svhn_conv_vae.py | EI_*.json, EJ_*.json |
| Exp-NatImg-Sweep | Exp_NatImg_AU.py | Exp_NatImg_AU.json |
| V1/V2 threshold-sensitivity ranges | (analysis) Exp_AU_Rule | Exp_AU_Rule.json |
| δ sensitivity (Appendix) | EX_au_delta_sensitivity.py | EX_au_delta_sensitivity.json |
| β sensitivity (Appendix) | EZ_beta_sensitivity_plot.py | EZ_beta_sensitivity.json |

Figures: `replot_E4_dualaxis_v43.py`, `replot_E4_multiseed_v43.py`,
`replot_EN_extended_v44.py` regenerate the current paper figures directly
from the JSON logs (no retraining required).

## Reproducibility settings

- Random seeds: 42 (single-seed sweeps); 42, 123, 777 (3-seed verification);
  synthetic-manifold experiments use 3–5 seeds as stated per experiment.
- Data subsets: the first n_train / n_test samples of the official splits in
  fixed order (MNIST / Fashion-MNIST: 20,000 / 3,000). The subset indices are
  therefore `range(0, n_train)` and `range(0, n_test)` of the official
  torchvision splits — deterministic given the official ordering.
- Fixed hyperparameters: SEED as above; TWONN_K = 2; MLE_K = 10
  (stability checked for k ∈ {5, 10, 20}); AU_THRESHOLD δ = 1e-2;
  CKA_SUBSAMPLE = 1000; Adam lr = 1e-3, batch size 128.
- KL annealing (Conv-VAE): cyclical, three cycles; within each cycle β rises
  linearly from 0 to β_max over the first half and is held at β_max over the
  second half (see paper Section 5.6.1 and Appendix C.1).

## Environment

See ENVIRONMENT.md.
