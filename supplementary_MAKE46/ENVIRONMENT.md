# Environment

## Estimator implementations

- TwoNN (k = 2) and MLE (k = 10): own implementations in
  `code/src/metrics/intrinsic_dim.py`.
- DANCo and ESS (auxiliary cross-checks in Exp-Boot-CI): the `skdim`
  (scikit-dimension) package, called from
  `code/src/experiments/Exp_Boot_CI.py`; record the installed `skdim`
  version together with the other library versions below.
- KL cyclic annealing schedule: implemented in the Conv-VAE experiment
  scripts (e.g., `code/src/experiments/EN_conv_vae_extended_m.py`,
  `Exp_ConvV_HighBeta.py`).

Training environment reported in the paper (Section 5.1):

- Framework: PyTorch 2.0
- Hardware: single NVIDIA GeForce RTX 3070 GPU (CPU fallback when CUDA is
  unavailable)
- Optimizer: Adam (lr = 1e-3, β1 = 0.9, β2 = 0.999, no weight decay)
- Batch size: 128; activations ReLU (Sigmoid output layer)
- Preprocessing: [0,1] pixel normalization (images); zero-mean unit-variance
  standardization (synthetic manifolds)

> **TODO before submission**: confirm and record the exact versions from the
> training machine (`python -V`, `torch.__version__`, `torch.version.cuda`,
> `nvidia-smi`, and `pip freeze > requirements_freeze.txt`), and place
> `requirements_freeze.txt` in this directory. The analysis machine used to
> regenerate figures ran Python 3.11.8 / PyTorch 2.5.1 / NumPy 1.23.5 /
> scikit-learn 1.4.2 / SciPy 1.13.1 / Matplotlib 3.8.4 (CPU); figure
> regeneration from the JSON logs does not depend on the GPU environment.
