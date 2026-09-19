"""Japanese captions and cells; retain original numerical table data."""
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
inherited = runpy.run_path(str(ROOT / "src/nn51/localization.py"))
CAPTIONS = dict(inherited["CAPTIONS"])
CELLS = dict(inherited["CELLS"])

def environments(s):
    return {re.search(r"\\label\{([^}]+)\}", m[0])[1]: m[0]
            for m in re.finditer(r"\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}", s, re.S)}

def caption(s):
    start = s.index(r"\caption{") + len(r"\caption{")
    end, depth = start, 1
    while depth:
        c = s[end]
        escaped = (len(s[:end]) - len(s[:end].rstrip("\\"))) % 2
        if c in "{}" and not escaped:
            depth += 1 if c == "{" else -1
        end += 1
    return s[start:end-1]

def cell_parts(cell):
    core = cell.strip()
    m = re.match(r"(.*?)(\s*\\\\(?:\\(?:midrule|bottomrule|toprule))?\s*)$", core, re.S)
    return (m[1].strip(), m[2]) if m else (core, "")

en = environments((ROOT / "main_paper_MAKE51.tex").read_text())
ja = environments((ROOT / "main_paper_NN51.tex").read_text())
for key, translated in ja.items():
    if key not in en:
        continue
    CAPTIONS[key] = caption(translated)
    a = [x for x in en[key].splitlines() if "&" in x and not x.startswith(r"\caption")]
    b = [x for x in translated.splitlines() if "&" in x and not x.startswith(r"\caption")]
    if len(a) == len(b):
        for x, y in zip(a, b):
            if len(x.split("&")) == len(y.split("&")):
                for c, d in zip(x.split("&"), y.split("&")):
                    CELLS[cell_parts(c)[0]] = cell_parts(d)[0]

CAPTIONS.update({
    "tab:gated": r"各データセットで5重み・5候補シードにわたる開始次元数のアブレーション。ID採用はMNISTの25条件，fallbackは他3データセットの75条件である。ID-localとID-gated midpointは同じ信頼性判定，fallback，参照群の費用計上を用いる。参照群を使わない中点方策は別方策である。一致は全グリッド最小値との一致を表す。費用は初回利用の統一秒数であり，新しい実時間測定ではない。",
    "tab:newcounts": r"未変換の次元読み取り値。過去のHC・未正規化有限刻み幅ARD（5シード）と，新しいARM・厳密ヤコビアンARD（3シード）の比較。平均$\pm$標本SD。学習・読み取り設定が異なるため，すべての差を推定器だけに帰属できない。過去の有限刻み幅ARDの修正後結果は表\ref{tab:ard_normalized}に保持する。",
    "tab:armnew": r"新しいローカルARM実装。3シード（42，123，777），300 epochs。次元数と検証MSE（$10^3$倍）は平均$\pm$標本SD，括弧内は外部目標$Q$の達成数。平均再構成・標本再構成はともに学習済み二値ゲートマスクを用い，標本再構成は事後32標本による。転用は最近傍のグリッド次元における別の通常VAEを指す。制約は最終時点の確率的学習制約の達成であり，$Q$とは異なる。",
    "tab:ardnew": r"新しい厳密ヤコビアンARD実装。各設定3シード，300 epochs。事前分布更新の中心はゼロまたは潜在標本平均。個数は関連度の読み取り値であり，全次元は未剪定の初期次元モデルを表す。その個数へ削減したモデルではない。転用は最近傍のグリッド次元の通常VAEである。MSEは$10^3$倍，平均$\pm$標本SD，括弧内は$Q$達成数。",
    "tab:armgates": r"300 epochs後のARMゲート確率。中間は$0.1\le p\le0.9$，他の区分は厳密不等号を用いる。期待値は$\sum_j p_j$，個数は$p_j>0.5$で数える。dSpritesのゲートはどちらの極端な区分にも達しないため，閾値個数を安定した二値構造と解釈すべきではない。",
    "tab:newtest": r"test MSE（$10^3$倍）。3シードの平均$\pm$標本SD。ARMのnative評価は二値ゲートマスクと事後平均を用いる。ARDのnative評価は未剪定の全次元モデルを用いる。転用先通常VAEは過去の候補キャッシュから再利用したもので，そのtest集計は新しい未変換個数の選択に用いていない。",
    "tab:newcost": r"新しいGPU剪定実験の記録時間内訳。単位は秒，3シードの平均$\pm$標本SD。学習ループには定期的な学習・検証評価を含む。ARMの事後評価は最終native評価を含むが，ARDではその時間を個別に記録していない（未記録）。アンカー，転用候補，その他の欠落費用は含まない。これらは完全なend-to-end選択器費用ではない。",
    "fig:armconstraints": r"新しいARM各シードの確率的学習制約の移動平均。5 epochsごとに記録した。ゼロが制約境界で，負値は制約充足を表す。大きな正の違反量と負値を表示するため，縦軸は対称対数尺度とする。これらの推移は，決定論的な検証目標の達成とは異なる。",
    "fig:armcurves": r"その時点の二値ゲートマスクを用いたnative ARMの事後平均MSE。学習2000例の部分集合と検証分割で5 epochsごとに評価した。線と陰影は3シードの平均と標本SD。縦軸は対数尺度で，表示のため陰影の下限を正値に制限する。欠けている過去の通常VAE学習記録を再構成した曲線ではない。",
    "fig:ardcurves": r"ゼロ中心の分散更新を用いた，全次元ARDの事後平均MSE。5 epochsごとに評価し，3シードの平均と標本SDを示す。縦軸は対数尺度。学習用プールの2000例の部分集合には，事前分布更新専用の1800例とパラメータ勾配学習に用いた200例が含まれるため，純粋な学習適合度の推定ではない。これらの評価ではnativeモデルを剪定していない。",
})

CELLS.update({
    "FONDUE variants, grid criteria and search policies; separate pruning validation":
        "FONDUEの変種，グリッド基準，探索方策；別途の剪定検証",
    "Stratum": "層",
    "ID accepted": "ID採用",
    "Fallback": "fallback",
    "All": "全体",
    "ID gated midpoint": "ID-gated midpoint",
    "Gates": "ゲート数",
    "Native mean ($Q$)": "native平均（$Q$）",
    "Native sampled ($Q$)": "native標本（$Q$）",
    "Transfer ($Q$)": "転用（$Q$）",
    "Centre": "中心",
    "Count": "個数",
    "Full MSE ($Q$)": "全次元MSE（$Q$）",
    "Transfer width": "転用次元",
    "Transfer MSE ($Q$)": "転用MSE（$Q$）",
    "Zero": "ゼロ",
    "Sample": "標本平均",
    "Old ARD ($n=5$)": "旧ARD（$n=5$）",
    "Exact, zero": "厳密・ゼロ",
    "Exact, sample": "厳密・標本",
    "Seed": "シード",
    "Mid": "中間",
    "Expected": "期待値",
    "Implementation": "実装",
    "Training loop": "学習ループ",
    "Jacobian": "ヤコビアン",
    "Post-evaluation": "事後評価",
    "NR": "未記録",
    "ARD, zero": "ARD・ゼロ",
    "ARD, sample": "ARD・標本",
    "ARD zero": "ARD・ゼロ",
    "ARD sample": "ARD・標本",
    "Native/full": "native／全次元",
    "Transferred": "転用",
    "Fourteen historical ordinary selectors/controls": "過去の通常選択器・対照14手法",
    "Historical HC adaptation": "過去のHC変更実装",
    "Historical ARD adaptation": "過去のARD変更実装",
    "Search replay and start-width ablation": "探索再評価と開始次元のアブレーション",
    "Seven policies, 700 rows; 10,000 coefficient/threshold settings":
        "7方策，700行；係数・閾値の10,000設定",
    "Separate MC ELBO capacity study": "独立したMC ELBO容量実験",
    "One seed, 12 ordinary VAE widths": "1シード，通常VAEの12次元設定",
    "Additional ARM validation": "追加ARM検証",
    r"Three seeds, 12 runs; $\beta=1$ anchor": r"3シード，12実行；$\beta=1$アンカー",
    "Additional exact-Jacobian ARD": "追加の厳密ヤコビアンARD",
    "Three seeds, two variance-update centres; 24 runs": "3シード，分散更新の2中心設定；24実行",
    "Separate MNIST model records": "独立したMNISTモデル記録",
    "Additional start-width control": "追加開始次元対照",
    "Additional ARM/ARD records": "追加ARM・ARD記録",
    "Additional-run aggregation": "追加実行の集計",
    "Additional source identities": "追加ソースの識別",
})
