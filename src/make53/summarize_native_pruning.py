"""Audit supplied native-pruning records and build MAKE53 tables without training.

Only MSE, counts and explicitly bounded duration fields are used. The raw
elbo/rec/kl fields are deliberately excluded from scientific summaries.
"""
from pathlib import Path
import hashlib
import json
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/make53"
OUT.mkdir(exist_ok=True)
DS = ["MNIST", "FashionMNIST", "dSprites", "CIFAR10"]
NAMES = dict(zip(DS, ["MNIST", "Fashion-MNIST", "dSprites", "CIFAR-10"]))
SEEDS = [42, 123, 777]
FIELDS = ["unpruned", "pruned_masked", "pruned_finetuned", "transfer"]
sources = []

def read(name):
    p = ROOT / name
    sources.append(p)
    return json.loads(p.read_text())

def pm(x, digits=2, scale=1000):
    a = [float(v) * scale for v in x]
    return "$" + f"{st.mean(a):.{digits}f} \\pm {st.stdev(a):.{digits}f}" + "$"

def table(name, caption, label, cols, header, rows, size="small"):
    (OUT / (name + ".tex")).write_text(
        "\\begin{table}[H]\n\\caption{" + caption + "}\\label{" + label + "}\n"
        "\\" + size + "\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + cols + "@{}}\n"
        "\\toprule\n" + header + " \\\\\n\\midrule\n" +
        "\n".join(row + " \\\\" for row in rows) +
        "\n\\bottomrule\\end{tabular*}\n\\end{table}\n")

data = read("results/make52/ard_pruned_all.json")
spec = read("results/make52/native_pruned_spec.json")
assert set(data) == set(DS)
summary = {"datasets": {}, "used_fields": ["mse", "selected_m", "m_on_grid",
           "train_seconds", "finetune_seconds"], "elbo_fields_used": False}
val_rows, test_rows, seed_rows, cost_rows = [], [], [], []
cfg = read("config/experiment_config_MAKE49.json")
for ds in DS:
    rr = data[ds]
    assert rr == read(f"results/make52/ard_pruned_{ds}.json")
    assert sorted(r["seed"] for r in rr) == SEEDS
    cache = read(f"results/make49/stage4_cache/{ds}.json")
    old = {r["seed"]: r for r in read(f"results/make52/ard_native_{ds}.json")
           if r["center_mu_alpha"]}
    grid = cfg["candidate_grids"][ds]
    thresholds = {}
    for r in rr:
        assert r["dataset"] == ds and r["m_start"] == max(grid)
        assert 0 < r["selected_m"] <= r["m_start"]
        assert r["selected_m"] == old[r["seed"]]["selected_m"]
        assert r["unpruned"]["mse"] == old[r["seed"]]["eval_full"]["mse"]
        assert r["unpruned_test"]["mse"] == old[r["seed"]]["eval_test_full"]["mse"]
        assert r["transfer"]["m_on_grid"] == min(grid, key=lambda g: abs(g-r["selected_m"]))
        def key(m):
            return f"vae|{ds}|m{m}|s{r['seed']}|e300|es1|b1"
        c = cache[key(r["transfer"]["m_on_grid"])]
        assert r["transfer"]["val"] == c["val"] and r["transfer"]["test"] == c["test"]
        thresholds[r["seed"]] = 1.1 * cache[key(r["m_start"])]["val"]["mse"]
        assert r["pruned_masked"]["elbo"] == r["pruned_masked_K32"]["elbo"]
        assert r["provenance"]["gpu"] == "NVIDIA GeForce RTX 3070"
        assert r["provenance"]["python"] == "3.12.3"
        assert r["provenance"]["torch"] == "2.7.1+cu118"
        assert r["provenance"]["cuda"] == "11.8"
        assert r["provenance"]["cudnn"] == 90100
        assert r["provenance"]["driver"] == "570.211.01"
        seed_rows.append(f"{NAMES[ds]} & {r['seed']} & {r['selected_m']} & "
                         f"{r['transfer']['m_on_grid']} & " +
                         " & ".join("$" + f"{1000*r[f]['mse']:.3f}" + "$" for f in FIELDS[:3]) +
                         " & $" + f"{1000*r['pruned_masked_K32']['mse']:.3f}" + "$")
    ds_sum = {"count": [r["selected_m"] for r in rr],
              "grid_width": [r["transfer"]["m_on_grid"] for r in rr],
              "threshold_by_seed": thresholds, "conditions": {}}
    vcells, tcells = [], []
    full_mean = st.mean(r["unpruned"]["mse"] for r in rr)
    for f in FIELDS:
        v = [(r[f]["val"] if f == "transfer" else r[f])["mse"] for r in rr]
        t = [(r[f]["test"] if f == "transfer" else r[f + "_test"])["mse"] for r in rr]
        q = sum(x <= thresholds[r["seed"]] for r, x in zip(rr, v))
        paired = [(x / r["unpruned"]["mse"] - 1) * 100 for r, x in zip(rr, v)]
        ds_sum["conditions"][f] = dict(
            validation_mean=st.mean(v), validation_sd=st.stdev(v), Q=q,
            test_mean=st.mean(t), test_sd=st.stdev(t),
            relative_change_ratio_of_means_percent=(st.mean(v)/full_mean-1)*100,
            paired_relative_change_mean_percent=st.mean(paired),
            paired_relative_change_sd_percent=st.stdev(paired))
        vcells.append(pm(v) + f" ({q}/3)")
        tcells.append(pm(t))
    val_rows.append(NAMES[ds] + " & " + pm(ds_sum["count"], 1, 1) + " & " + " & ".join(vcells))
    test_rows.append(NAMES[ds] + " & " + " & ".join(tcells))
    cost_rows.append(NAMES[ds] + " & " + pm([r["train_seconds"] for r in rr], 1, 1) +
                     " & " + pm([r["finetune_seconds"] for r in rr], 1, 1))
    summary["datasets"][ds] = ds_sum

table("native_pruning_validation",
      r"Native ARD masking and decoder adaptation using the sample-centred variance-update variant. "
      r"Three seeds per dataset; validation MSE $\times10^3$, mean $\pm$ sample SD. "
      r"Count is retained axes; Full uses all starting axes; Mask sets other axes to zero; "
      r"Adapt adds 30 decoder-only epochs; Transfer uses a separate cached ordinary VAE at the nearest grid width. "
      r"Parentheses give attainment of the same deterministic validation target $Q$ in each condition. "
      r"All four conditions fail $Q$ on dSprites, including Full.",
      "tab:ardpruned", "lccccc",
      r"Dataset & Count & Full ($Q$) & Mask ($Q$) & Adapt ($Q$) & Transfer ($Q$)",
      val_rows, "footnotesize")
table("native_pruning_test",
      r"Native-pruning test MSE $\times10^3$, mean $\pm$ sample SD across three seeds. "
      r"All columns use posterior-mean reconstruction. Conditions match Table~\ref{tab:ardpruned}; "
      r"test metrics do not choose the axes or the fixed adaptation budget.",
      "tab:ardprunedtest", "lcccc",
      r"Dataset & Full & Mask & Adapt & Transfer", test_rows)
table("native_pruning_seeds",
      r"Per-seed native-pruning validation MSE $\times10^3$. Count is the relevance-selected axis count; "
      r"Grid is the transferred ordinary-VAE width. Full, Mask and Adapt use posterior means. "
      r"Sampled mask averages MSE over 32 posterior draws before adaptation; it is neither an ELBO nor a comparison under a sampled-anchor target. "
      r"Exact masks and model checkpoints were not saved.",
      "tab:ardprunedseeds", "lrrrrrrr",
      r"Dataset & Seed & Count & Grid & Full & Mask & Adapt & Sampled mask",
      seed_rows, "footnotesize")
table("native_pruning_cost",
      r"Partial durations for native-pruning runs, seconds, mean $\pm$ sample SD across three seeds. "
      r"Training includes its final prior update but excludes the initial one. Adaptation times only the 30-epoch decoder loop. "
      r"Jacobian/mask construction, final evaluations, anchor and transfer training are excluded and are not separately timed in this run.",
      "tab:ardprunedcost", "lcc",
      r"Dataset & Training interval & Decoder adaptation", cost_rows)
summary["record_checks"] = dict(runs=12, per_dataset_files_match=True,
    seed_coverage=True, nearest_grid_and_cache_transfers_match=True,
    full_mse_and_counts_match_prior_sample_centred_records=True,
    anchor_Q_recomputed=True, native_masks_and_checkpoints_unavailable=True,
    new_elbo_fields_are_not_mc_elbo=True)
(OUT / "native_pruning_summary.json").write_text(json.dumps(summary, indent=2))
sources += [ROOT / "src/make52/ard_native_pruned.py",
            ROOT / "src/make52/ard_native.py", ROOT / "src/make52/geco_arm.py",
            ROOT / "MAKE52_revision_review.md", ROOT / "MAKE52_実験検証結果.md",
            Path(__file__)]
(OUT / "native_pruning_manifest.json").write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
     for p in sorted(set(sources))}, indent=2))
print(json.dumps(summary["record_checks"], indent=2))
