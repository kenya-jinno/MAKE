"""
複数シード実験結果を読み込み，main_paper.tex の表を更新するスクリプト
"""
import json
import sys

def load_json(path):
    with open(path) as f:
        return json.load(f)

def fmt(mean, std, decimals=3):
    """mean±std のフォーマット"""
    if decimals == 3:
        return f"{mean:.3f}$\\pm${std:.3f}"
    elif decimals == 4:
        return f"{mean:.4f}$\\pm${std:.4f}"
    elif decimals == 1:
        return f"{mean:.1f}$\\pm${std:.1f}"
    else:
        return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"

def print_e1_tables(e1):
    """E1の表をLaTeX形式で出力"""
    print("\n=== E1: Swiss Roll ===")
    print("% Swiss Roll (d_true=2)")
    print(r"\begin{tabular}{rccc}")
    print(r"\toprule")
    print(r"$m$ & MSE & AU & TwoNN ID \\")
    print(r"\midrule")
    sr = e1['SwissRoll']
    for m in [1, 2, 3, 4]:
        s = sr[str(m)]
        mse_str = fmt(s['mse_mean'], s['mse_std'], 3)
        au_str = f"{s['au_mean']:.1f}$\\pm${s['au_std']:.2f}"
        id_str = fmt(s['id_mean'], s['id_std'], 2)
        prefix = r"\textbf{" + str(m) + r"} $= \dtrue$" if m == 2 else str(m)
        print(f"{prefix} & {mse_str} & {au_str} & {id_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n=== E1: Torus ===")
    print("% Torus (d_true=2)")
    print(r"\begin{tabular}{rccc}")
    print(r"\toprule")
    print(r"$m$ & MSE & AU & TwoNN ID \\")
    print(r"\midrule")
    torus = e1['Torus']
    for m in [1, 2, 3, 4]:
        s = torus[str(m)]
        mse_str = fmt(s['mse_mean'], s['mse_std'], 3)
        au_str = f"{s['au_mean']:.1f}$\\pm${s['au_std']:.2f}"
        id_str = fmt(s['id_mean'], s['id_std'], 2)
        prefix = r"\textbf{3} $\leftarrow$ 実質的肘点" if m == 3 else str(m)
        print(f"{prefix} & {mse_str} & {au_str} & {id_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_ea_tables(ea):
    """EA の表をLaTeX形式で出力"""
    m_values = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    print("\n=== EA: Swiss Roll ===")
    print(r"\begin{tabular}{rccc}")
    print(r"\toprule")
    print(r"$m$ & VAE（AU） & VAE MSE \\")
    print(r"\midrule")
    sr = ea['SwissRoll']
    for m in m_values:
        s = sr[str(m)]
        au_str = f"{s['au_mean']:.1f}$\\pm${s['au_std']:.1f}"
        mse_str = fmt(s['mse_mean'], s['mse_std'], 3)
        sat_mark = r" $\leftarrow$ 飽和" if m == 3 else ""
        bold_start = r"\textbf{" if m >= 3 else ""
        bold_end = "}" if m >= 3 else ""
        print(f"{m} & {bold_start}{au_str}{bold_end}{sat_mark} & {mse_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n=== EA: Torus ===")
    print(r"\begin{tabular}{rccc}")
    print(r"\toprule")
    print(r"$m$ & VAE（AU） & VAE MSE \\")
    print(r"\midrule")
    torus = ea['Torus']
    for m in m_values:
        s = torus[str(m)]
        au_str = f"{s['au_mean']:.1f}$\\pm${s['au_std']:.1f}"
        mse_str = fmt(s['mse_mean'], s['mse_std'], 3)
        sat_mark = r" $\leftarrow$ 飽和" if m == 6 else ""
        bold_start = r"\textbf{" if m >= 6 else ""
        bold_end = "}" if m >= 6 else ""
        print(f"{m} & {bold_start}{au_str}{bold_end}{sat_mark} & {mse_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_ec_table(ec):
    """EC の表をLaTeX形式で出力"""
    print("\n=== EC: IsometricAE 比較 ===")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"モデル & MSE & 条件数$\kap$ & Trustworthiness \\")
    print(r"\midrule")
    for model in ['AE', 'CAE', 'IsometricAE', 'DAE']:
        s = ec[model]
        mse_str = fmt(s['mse_mean'], s['mse_std'], 4)
        kap_str = fmt(s['kappa_mean'], s['kappa_std'], 3)
        trust_str = fmt(s['trust_mean'], s['trust_std'], 4)
        prefix = r"\textbf{IsometricAE}" if model == 'IsometricAE' else model
        print(f"{prefix} & {mse_str} & {kap_str} & {trust_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    try:
        e1 = load_json('results/tables/E1_multiseed.json')
        print_e1_tables(e1)
    except FileNotFoundError:
        print("E1 results not yet available")

    try:
        ea = load_json('results/tables/EA_multiseed.json')
        print_ea_tables(ea)
    except FileNotFoundError:
        print("EA results not yet available")

    try:
        ec = load_json('results/tables/EC_multiseed.json')
        print_ec_table(ec)
    except FileNotFoundError:
        print("EC results not yet available")
