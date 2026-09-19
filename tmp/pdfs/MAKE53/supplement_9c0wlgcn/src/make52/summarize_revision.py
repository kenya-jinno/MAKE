"""Aggregate delivered MAKE52 records without importing or rerunning training."""
from pathlib import Path
import csv, hashlib, json, os
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/make52"
FIG = ROOT / "results/figures/MAKE52"
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/pdfs/MAKE52/mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp/pdfs/MAKE52/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DS = ["MNIST", "FashionMNIST", "dSprites", "CIFAR10"]
NAMES = dict(zip(DS, ["MNIST", "Fashion-MNIST", "dSprites", "CIFAR-10"]))
SEEDS = [42, 123, 777]
read = lambda p: json.loads((ROOT / p).read_text())
arm = read("results/make52/geco_arm_all.json")
ard = read("results/make52/ard_native_all.json")
gated = read("results/make52/start_width_ablation.json")["rows"]
replay = list(csv.DictReader((ROOT / "results/make51/search_replay.csv").open()))
prune = read("results/make49/stage7_b7b8.json")["results"]
key = lambda r: (r["dataset"], int(r["seed"]), float(r["beta"]))
idl = {key(r): r for r in replay if r["policy"] == "ID local"}
assert len(gated) == len(idl) == 100
assert all(r["selected_m"] == int(idl[key(r)]["selected_m"]) for r in gated)
assert all(r["exact"] and r["quality_met"] for r in gated)
for r in gated:
    old = idl[key(r)]
    assert r["id_accepted"] == (old["id_accepted"] == "True")
    assert r["fallback"] == (old["fallback"] == "True")
    assert r["id_accepted"] == (r["dataset"] == "MNIST")
    assert abs(r["reference_seconds"] - float(old["reference_seconds"])) < 1e-9
    if r["fallback"]:
        assert r["requested_count"] == int(old["requested_count"])
        assert abs(r["cold_seconds"] - float(old["cold_seconds"])) < 1e-9


def pm(values, digits=1, scale=1):
    x = [float(v) * scale for v in values]
    return "$" + f"{st.mean(x):.{digits}f} \\pm {st.stdev(x):.{digits}f}" + "$"

def table(name, caption, label, cols, header, rows, size="small"):
    text = (
        "\\begin{table}[H]\n\\caption{" + caption + "}\\label{" + label + "}\n"
        "\\" + size + "\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + cols + "@{}}\n"
        "\\toprule\n" + header + " \\\\\n\\midrule\n" +
        "\n".join(row + " \\\\" for row in rows) +
        "\n\\bottomrule\\end{tabular*}\n\\end{table}\n")
    (OUT / (name + ".tex")).write_text(text)

summary = {"strata": [], "arm": {}, "ard": {}, "record_checks": {}}
rows = []
for stratum, predicate in [
    ("ID accepted", lambda r: r["dataset"] == "MNIST"),
    ("Fallback", lambda r: r["dataset"] != "MNIST"),
    ("All", lambda r: True)]:
    for policy in ["ID local", "ID gated midpoint", "Midpoint local", "Ascending Q"]:
        rr = ([r for r in gated if predicate(r)] if policy == "ID gated midpoint"
              else [r for r in replay if predicate(r) and r["policy"] == policy])
        n = len(rr)
        exact = sum(str(r["exact"]).lower() == "true" for r in rr)
        calls = st.mean(float(r["requested_count"]) for r in rr)
        cold = st.mean(float(r["cold_seconds"]) for r in rr)
        rows.append(f"{stratum} & {policy} & {n} & {exact}/{n} & {calls:.2f} & {cold:.1f}")
        summary["strata"].append(dict(stratum=stratum, policy=policy, n=n,
                                      exact=exact, calls=calls, cold_seconds=cold))
table("gated_ablation",
      r"Start-width ablation across five weights and five candidate seeds per dataset. "
      r"ID accepted comprises MNIST (25 conditions); fallback comprises the other three datasets (75). "
      r"ID-local and ID-gated midpoint use identical reliability checks, fallback and reference-bank charges. "
      r"The bank-free midpoint is a separate policy. Exact denotes the full-grid minimum; costs are canonical cold-use seconds, not new wall-clock measurements.",
      "tab:gated", "llrrrr", r"Stratum & Policy & $n$ & Exact & Calls & Cold (s)", rows)

rows, gate_rows, ard_rows, count_rows, time_rows = [], [], [], [], []
for ds in DS:
    assert arm[ds] == read(f"results/make52/geco_arm_{ds}.json")
    assert ard[ds] == read(f"results/make52/ard_native_{ds}.json")
    assert sorted(r["seed"] for r in arm[ds]) == SEEDS
    assert len(ard[ds]) == 6
    cache = read(f"results/make49/stage4_cache/{ds}.json")
    thresholds = {}
    for r in arm[ds]:
        anchor_key = f"vae|{ds}|m{r['m_start']}|s{r['seed']}|e300|es1|b1"
        thresholds[r["seed"]] = 1.1 * cache[anchor_key]["val"]["mse"]
        assert abs(thresholds[r["seed"]] - r["tau_per_pixel"]) < 1e-12
        assert r["constraint_satisfied"] == (r["final_constraint_ma"] <= 0)
        assert r["selected_m"] == sum(p > .5 for p in r["gate_sigmoid"])
        assert r["eval_mean"]["elbo"] == r["eval_sampled_K32"]["elbo"]
        assert r["eval_mean"]["rec"] == r["eval_sampled_K32"]["rec"]
    for r in arm[ds] + ard[ds]:
        assert r["curves"]["epoch"] == list(range(5, 301, 5))
        assert all(len(v) == 60 for v in r["curves"].values())
        tr = r["transferred"]
        ck = f"vae|{ds}|m{tr['m_on_grid']}|s{r['seed']}|e300|es1|b1"
        assert tr["val"] == cache[ck]["val"] and tr["test"] == cache[ck]["test"]
    for center in [False, True]:
        rr = [r for r in ard[ds] if r["center_mu_alpha"] == center]
        assert sorted(r["seed"] for r in rr) == SEEDS
        for r in rr:
            score = np.array(r["sigma_hat"]) * np.array(r["jacobian_weight"])
            assert np.allclose(score, r["relevance_score"], rtol=1e-5)
            selected = np.searchsorted(np.cumsum(np.sort(score)[::-1]) / score.sum(), .99) + 1
            assert selected == r["selected_m"]
            assert "exact Jacobian" in r["weight_mode"]
    aa = arm[ds]
    q = lambda field: sum(r[field]["mse"] <= thresholds[r["seed"]] for r in aa)
    qt = sum(r["transferred"]["val"]["mse"] <= thresholds[r["seed"]] for r in aa)
    qs = sum(r["constraint_satisfied"] for r in aa)
    native = pm([r["eval_mean"]["mse"] for r in aa], 2, 1000)
    sampled = pm([r["eval_sampled_K32"]["mse"] for r in aa], 2, 1000)
    transfer = pm([r["transferred"]["val"]["mse"] for r in aa], 2, 1000)
    rows.append(f"{NAMES[ds]} & {pm([r['selected_m'] for r in aa])} & "
                f"{native} ({q('eval_mean')}/3) & {sampled} ({q('eval_sampled_K32')}/3) & "
                f"{transfer} ({qt}/3) & {qs}/3")
    summary["arm"][ds] = dict(raw=[r["selected_m"] for r in aa],
        constraint=qs, native_Q=q("eval_mean"), sampled_Q=q("eval_sampled_K32"), transfer_Q=qt)
    for r in aa:
        p = r["gate_sigmoid"]
        gate_rows.append(f"{NAMES[ds]} & {r['seed']} & "
                         f"{sum(x < .1 for x in p)} & {sum(.1 <= x <= .9 for x in p)} & "
                         f"{sum(x > .9 for x in p)} & {sum(p):.2f} & {r['selected_m']}")
    for center in [False, True]:
        rr = [r for r in ard[ds] if r["center_mu_alpha"] == center]
        nq = sum(r["eval_full"]["mse"] <= thresholds[r["seed"]] for r in rr)
        tq = sum(r["transferred"]["val"]["mse"] <= thresholds[r["seed"]] for r in rr)
        mode = "Zero" if not center else "Sample"
        ard_rows.append(f"{NAMES[ds]} & {mode} & {pm([r['selected_m'] for r in rr])} & "
                        f"{pm([r['eval_full']['mse'] for r in rr],2,1000)} ({nq}/3) & "
                        f"{pm([r['transferred']['m_on_grid'] for r in rr])} & "
                        f"{pm([r['transferred']['val']['mse'] for r in rr],2,1000)} ({tq}/3)")
        summary["ard"][ds + ("_centered" if center else "_zero")] = dict(
            raw=[r["selected_m"] for r in rr], full_Q=nq, transfer_Q=tq)
    time_rows.append(f"{NAMES[ds]} & ARM & {pm([r['train_seconds'] for r in aa],1)} & "
                     f"--- & {pm([r['eval_seconds'] for r in aa],3)}")
    for center in [False, True]:
        rr = [r for r in ard[ds] if r["center_mu_alpha"] == center]
        mode = "ARD, zero" if not center else "ARD, sample"
        time_rows.append(f"{NAMES[ds]} & {mode} & {pm([r['train_seconds'] for r in rr],1)} & "
                         f"{pm([r['jacobian_seconds'] for r in rr],3)} & NR")

table("arm_native",
      r"New ARM-based local implementation: three seeds (42, 123, 777), 300 epochs. "
      r"Widths and validation MSE ($\times10^3$) are mean $\pm$ sample SD; parentheses give external target attainment $Q$. "
      r"Mean and sampled reconstruction both use the learned binary gate mask; sampled uses 32 posterior draws. "
      r"Transfer denotes a separate ordinary VAE at the nearest grid width. Constraint is endpoint stochastic training-constraint attainment, distinct from $Q$.",
      "tab:armnew", "lccccc",
      r"Dataset & Gates & Native mean ($Q$) & Native sampled ($Q$) & Transfer ($Q$) & Constraint",
      rows, "footnotesize")
table("ard_native",
      r"New exact-Jacobian ARD implementations, three seeds per setting, 300 epochs. "
      r"Prior-update centre is zero or the sample latent mean. Count is a relevance readout; Full is the unpruned starting-width model, not a model reduced to that count. "
      r"Transfer is an ordinary VAE at the nearest grid width. MSE $\times10^3$, mean $\pm$ sample SD; parentheses give $Q$ counts.",
      "tab:ardnew", "llcccc",
      r"Dataset & Centre & Count & Full MSE ($Q$) & Transfer width & Transfer MSE ($Q$)",
      ard_rows, "footnotesize")
table("arm_gates",
      r"ARM gate probabilities after 300 epochs. Mid denotes $0.1\le p\le0.9$; the other bins use strict inequalities. "
      r"Expected is $\sum_j p_j$; count uses $p_j>0.5$. No dSprites gate reaches either extreme, so its threshold count should not be read as a stable binary architecture.",
      "tab:armgates", "lrrrrrr",
      r"Dataset & Seed & $p<0.1$ & Mid & $p>0.9$ & Expected & Count", gate_rows)
table("new_cost",
      r"Logged duration components for new GPU pruning runs, seconds, mean $\pm$ sample SD over three seeds. "
      r"Training loop includes periodic train/validation evaluations. ARM post-evaluation covers final native evaluations; ARD did not separately time them (NR). "
      r"Anchor, transferred-candidate and other missing costs are not included here. "
      r"These intervals are not complete end-to-end selector costs.",
      "tab:newcost", "llccc",
      r"Dataset & Implementation & Training loop & Jacobian & Post-evaluation", time_rows)

for ds in DS:
    historical_hc = [r["selected_m"] for r in prune if r["dataset"] == ds and r["method"] == "B7_geco_l0" and r.get("beta_context") == 1]
    historical_ard = [r["selected_m"] for r in prune if r["dataset"] == ds and r["method"] == "B8_ard_vae" and r.get("beta_context") == 1]
    if not historical_hc or not historical_ard:
        raise ValueError("Historical pruning method identifiers need inspection")
    count_rows.append(f"{NAMES[ds]} & {pm(historical_hc)} & {pm([r['selected_m'] for r in arm[ds]])} & "
                      f"{pm(historical_ard)} & "
                      f"{pm([r['selected_m'] for r in ard[ds] if not r['center_mu_alpha']])} & "
                      f"{pm([r['selected_m'] for r in ard[ds] if r['center_mu_alpha']])}")
table("pruning_counts",
      r"Raw dimension readouts: historical HC and unnormalized finite-step ARD (five seeds) versus new ARM and exact-Jacobian ARD (three seeds). "
      r"Mean $\pm$ sample SD. Different training and readout settings prevent attributing every difference to the estimator alone. "
      r"The corrected historical finite-step ARD results are retained in Table~\ref{tab:ard_normalized}.",
      "tab:newcounts", "lccccc",
      r"Dataset & HC ($n=5$) & ARM ($n=3$) & Old ARD ($n=5$) & Exact, zero & Exact, sample", count_rows, "footnotesize")

test_rows = []
for ds in DS:
    for mode in ["ARM", "ARD zero", "ARD sample"]:
        rr = (arm[ds] if mode == "ARM" else
              [r for r in ard[ds] if r["center_mu_alpha"] == (mode == "ARD sample")])
        field = "eval_test_mean" if mode == "ARM" else "eval_test_full"
        test_rows.append(f"{NAMES[ds]} & {mode} & {pm([r[field]['mse'] for r in rr],2,1000)} & "
                         f"{pm([r['transferred']['test']['mse'] for r in rr],2,1000)}")
table("new_test",
      r"Test MSE ($\times10^3$), mean $\pm$ sample SD over three seeds. "
      r"ARM native evaluation uses a binary gate mask and posterior mean. ARD native evaluation is the unpruned full-width model. "
      r"Transferred ordinary VAEs are reused from the archived candidate cache; their test summaries did not select the new raw counts.",
      "tab:newtest", "llcc", r"Dataset & Implementation & Native/full & Transferred", test_rows)

plt.rcParams.update({"font.size": 10, "legend.fontsize": 9, "pdf.fonttype": 42})
for method in ["arm", "ard"]:
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
    for ax, ds in zip(axes.flat, DS):
        rr = arm[ds] if method == "arm" else [r for r in ard[ds] if not r["center_mu_alpha"]]
        for field, label, color, ls in [
            ("train_mse", "Training-pool subset", "#2166ac", "-"),
            ("val_mse", "Validation", "#b2182b", "--")]:
            vals = np.array([r["curves"][field] for r in rr])
            avg, sd = vals.mean(0), vals.std(0, ddof=1)
            x = rr[0]["curves"]["epoch"]
            ax.plot(x, avg, color=color, ls=ls, label=label)
            ax.fill_between(x, np.maximum(avg-sd, 1e-8), avg+sd, color=color, alpha=.14)
        ax.set(title=NAMES[ds], xlabel="Epoch", ylabel="MSE", yscale="log")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.savefig(FIG / (method + "_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
for ax, ds in zip(axes.flat, DS):
    for r in arm[ds]:
        ax.plot(r["curves"]["epoch"], r["curves"]["constraint_ma"], label=f"seed {r['seed']}")
    ax.axhline(0, c=".4", ls="--")
    ax.set(title=NAMES[ds], xlabel="Epoch", ylabel="Training constraint (SSE)", yscale="symlog")
    ax.grid(alpha=.2)
axes[0, 0].legend()
fig.savefig(FIG / "arm_constraints.pdf", bbox_inches="tight")
plt.close(fig)
summary["record_checks"] = dict(gated_rows=100, arm_runs=12, ard_runs=24,
    per_dataset_seed_files_match=True, candidate_transfers_match=True,
    relevance_count_recomputed=True, expected_curve_epochs=True,
    new_elbo_fields_are_not_mc_elbo=True)
(OUT / "manuscript_summary.json").write_text(json.dumps(summary, indent=2))
sources = [ROOT / "MAKE52_実験検証結果.md"] + sorted((ROOT/"src/make52").glob("*.py"))
sources += [OUT / f"{n}.json" for n in ["geco_arm_all","ard_native_all","start_width_ablation","stratified_replay"]]
(OUT / "manuscript_source_manifest.json").write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
print(json.dumps(summary, indent=2))
