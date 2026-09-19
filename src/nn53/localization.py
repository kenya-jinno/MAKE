"""Japanese captions and cells; retain original numerical table data."""
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
inherited = runpy.run_path(str(ROOT / "src/nn52/localization.py"))
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

en = environments((ROOT / "main_paper_MAKE52.tex").read_text())
ja = environments((ROOT / "main_paper_NN52.tex").read_text())
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
    "tab:ardpruned": r"標本中心の分散更新を用いたARD変種のnativeマスクと復号器微調整。各データセット3シード。検証MSEは$10^3$倍，平均$\pm$標本SD。個数は保持軸数。全次元は初期の全軸を用い，マスクは他の軸をゼロに固定する。微調整は復号器のみを30 epochs追加更新する。転用は最近傍グリッド次元の別の保存済み通常VAEを用いる。括弧内は，各条件で同じ決定論的検証目標$Q$を達成した数。dSpritesでは，全次元を含む4条件すべてが$Q$に届かない。",
    "tab:ardprunedtest": r"Native剪定のtest MSE（$10^3$倍）。3シードの平均$\pm$標本SD。全列で事後平均による再構成を用いる。条件は表\ref{tab:ardpruned}と同じであり，test指標は軸や固定微調整予算の選択に用いない。",
    "tab:ardprunedseeds": r"シードごとのnative剪定の検証MSE（$10^3$倍）。個数は関連度で選択した軸数，グリッドは転用先通常VAEの次元数。全次元，マスク，微調整は事後平均を用いる。標本マスクは微調整前の32事後標本による平均MSEであり，ELBOでも標本アンカー目標による比較でもない。マスクそのものとモデルcheckpointは保存されていない。",
    "tab:ardprunedcost": r"Native剪定実行の部分的な時間計測。単位は秒，3シードの平均$\pm$標本SD。学習時間は最後の事前分布更新を含むが，最初の更新を含まない。微調整時間は復号器の30 epochsのループのみを計測する。ヤコビアン・マスクの生成，最終評価，アンカーおよび転用先の学習は含まず，この実行では個別に計測していない。",
    "tab:inventory": r"本改訂のソースと出力の対応。ディレクトリはプロジェクト直下からの相対表記。補足資料のREADMEに個別ファイル名と再集計コマンドを示す。",
})
CELLS.update({
    "Full ($Q$)": r"全次元（$Q$）", "Mask ($Q$)": r"マスク（$Q$）",
    "Adapt ($Q$)": r"微調整（$Q$）", "Full": "全次元", "Mask": "マスク",
    "Adapt": "微調整", "Grid": "グリッド", "Sampled mask": "標本マスク", "Transfer": "転用",
    "Training interval": "学習区間", "Decoder adaptation": "復号器微調整",
    "Native ARD masking and adaptation": "Native ARDのマスク・微調整",
    "Three seeds; paired extension of sample-centred settings, 12 evaluations":
        "3シード；標本中心設定の対応のある拡張，12評価",
    "Historical method, trajectory and diagnostic records": "過去の手法・推移・診断記録",
    "Search, AU and historical audit summaries": "探索・AU・過去の監査の集計",
    "Separate MNIST models and MC ELBO records": "別途のMNISTモデル・MC ELBO記録",
    "Gated-start, ARM, ARD and native-masking raw records": "判定付き開始点・ARM・ARD・nativeマスクのraw記録",
    "Native-masking summaries and source hashes": "Nativeマスク集計・ソースハッシュ",
    "Historical and additional training code": "過去と追加の学習コード",
    "Historical and additional reaggregation": "過去と追加の再集計",
    "English manuscript templates": "英語原稿テンプレート",
    "Japanese translation and verification": "日本語訳と検証",
    "English figures": "英語図", "Japanese figures": "日本語図",
})
