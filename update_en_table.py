"""EN 実験結果をmain_paper_NN12.texのテーブルに反映するスクリプト"""

import json, numpy as np, re

JSON_PATH = 'results/tables/EN_conv_vae_extended_m.json'
TEX_PATH  = 'main_paper_NN12.tex'

def load_en_results():
    with open(JSON_PATH) as f:
        data = json.load(f)
    seeds = [42, 123, 777]
    m_list = [4, 8, 16, 32, 64, 96, 128, 192, 256]
    rows = {}
    for m_idx, m in enumerate(m_list):
        aus = [data[str(s)]['results'][m_idx]['au'] for s in seeds]
        twonns = [data[str(s)]['results'][m_idx]['twonn'] for s in seeds]
        rows[m] = {
            'au_mean': np.mean(aus), 'au_std': np.std(aus),
            'twonn_mean': np.mean(twonns), 'twonn_std': np.std(twonns),
            'aus': aus,
        }
    mknees = [data[str(s)]['mknee'] for s in seeds]
    return rows, mknees, m_list

def build_table(rows, mknees, m_list):
    notes = {
        4: '', 8: '', 16: '', 32: '',
        64: 'EK と同一範囲（上限）',
        96: 'AU 増加鈍化開始域',
        128: '', 192: '', 256: '',
    }
    lines = []
    for m in m_list:
        r = rows[m]
        au_str = f"${r['au_mean']:.1f}\\pm{r['au_std']:.1f}$"
        tw_str = f"${r['twonn_mean']:.2f}\\pm{r['twonn_std']:.2f}$"
        note = notes.get(m, '')
        lines.append(f"{m:4d} & {au_str} & {tw_str} & {note} \\\\")
    mknee_mean = np.mean(mknees)
    mknee_std  = np.std(mknees)
    footer = (f"\\multicolumn{{4}}{{l}}{{$\\mknee$（$\\tau=0.25$）："
              f"seed42={mknees[0]}, seed123={mknees[1]}, seed777={mknees[2]}；"
              f"平均$={mknee_mean:.0f}\\pm{mknee_std:.0f}$}} \\\\")
    return lines, footer

def update_tex(table_lines, footer):
    with open(TEX_PATH, encoding='utf-8') as f:
        tex = f.read()

    # Build new tabular body
    new_body = '\n'.join(table_lines) + '\n\\midrule\n' + footer

    # Replace placeholder rows in tab:en
    pattern = (r'(\\label\{tab:en\}.*?\\midrule\n)'
               r'(.*?)'
               r'(\\midrule\n\\multicolumn\{4\}\{l\}\{（結果は実験完了後に更新予定）\}.*?\\bottomrule)')
    replacement = r'\g<1>' + new_body + r'\n\\bottomrule'

    new_tex = re.sub(pattern, replacement, tex, flags=re.DOTALL)
    if new_tex == tex:
        print("Warning: pattern not matched. Dumping table lines instead:")
        print('\n'.join(table_lines))
        print(footer)
        return

    with open(TEX_PATH, 'w', encoding='utf-8') as f:
        f.write(new_tex)
    print(f"Updated {TEX_PATH} with EN results.")

if __name__ == '__main__':
    rows, mknees, m_list = load_en_results()
    print("EN results loaded:")
    for m in m_list:
        r = rows[m]
        print(f"  m={m:3d}: AU={r['au_mean']:.1f}±{r['au_std']:.1f}, TwoNN={r['twonn_mean']:.2f}±{r['twonn_std']:.2f}")
    print(f"mknees: {mknees}, mean={np.mean(mknees):.1f}±{np.std(mknees):.1f}")
    table_lines, footer = build_table(rows, mknees, m_list)
    update_tex(table_lines, footer)
