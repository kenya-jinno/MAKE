"""Assemble the complete Japanese MAKE53 translation without changing results."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
from localization import CAPTIONS,CELLS,cell_parts

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src/nn53';OUT=ROOT/'results/nn53';OUT.mkdir(exist_ok=True)
english=(ROOT/'main_paper_MAKE53.tex').read_text()
template=(ROOT/'src/make53/manuscript_body.tex').read_text().strip()
template=re.sub(r'@@section:(\w+)@@',lambda m:(ROOT/'src/make53'/(m[1]+'.tex')).read_text().strip(),template)
original=template.split('\n\n')
parts=re.split(r'^@@block:(\d+)@@\n',(SRC/'blocks_ja.txt').read_text(),flags=re.M)
blocks={int(parts[i]):parts[i+1].strip() for i in range(1,len(parts),2)}
assert len(original)==166 and set(blocks)==set(range(166))
body='\n\n'.join(original[k] if blocks[k]=='@@keep@@' else blocks[k] for k in range(166))

def environments(text,kind):
    return re.findall(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}',text,re.S)
def by_label(items):
    return {re.search(r'\\label\{([^}]+)\}',s)[1]:s for s in items if r'\label{' in s}
maths=by_label(environments(english,'equation')+environments(english,'align'))
body=re.sub(r'@@math:([^@]+)@@',lambda m:maths[m[1]],body)
tables=by_label(environments(english,'table'))
body=re.sub(r'@@table:(\w+)@@',lambda m:tables['tab:'+m[1]],body)

au=json.loads((ROOT/'results/make51/au_paired.json').read_text())
mapping={'AU_C':f"{au['C']:g}",'AU_DIFF':f"{au['paired_difference']:+.3f}",
 'AU_BASE_DIFF':f"{au['mean_delta_au']:+.3f}",'AU_BASE_LO':f"{au['delta_au_hierarchical_ci'][0]:+.3f}",
 'AU_BASE_HI':f"{au['delta_au_hierarchical_ci'][1]:+.3f}",'AU_LO':f"{au['paired_hierarchical_ci'][0]:+.3f}",
 'AU_HI':f"{au['paired_hierarchical_ci'][1]:+.3f}",'AU100_DIFF':f"{au['noise100_difference']:+.3f}",
 'AU100_LO':f"{au['noise100_hierarchical_ci'][0]:+.3f}",'AU100_HI':f"{au['noise100_hierarchical_ci'][1]:+.3f}",
 'PYTHON':au['environment']['python'],'NUMPY':au['environment']['numpy'],'SKLEARN':au['environment']['sklearn']}
choice=json.loads((ROOT/'results/make51/model_validation/selection_test.json').read_text())['choices']
assert set(choice.values())=={12}
mapping['MC_INTERPRETATION']='検証MSEは12次元，proxyは12次元，Monte Carlo ELBOも12次元を選ぶ。この実行での一致は，proxyがELBO推定量であることを意味しない。'
hc=json.loads((ROOT/'results/make51/model_validation/native_hc_s42.json').read_text())
last=hc['curves'][-1]
mapping['HC_INTERPRETATION']=(f"最終的な学習移動平均制約は二乗誤差和の単位で{last['constraint_ma_sse']:+.3f}，"
 f"計数ゲートは{hc['raw_count']}個，乗数は{last['lambda_value']:.3f}である。転用先のグリッド次元は{hc['grid_width']}である。"
 "最終時点の学習制約はなお違反している。したがってこの診断により，HC実装を妥当性確認済みの剪定基準と位置づけることはしない。")
for k,v in mapping.items():body=body.replace('@@'+k+'@@',v)
assert '@@' not in body

def replace_caption(s,new):
    start=s.index(r'\caption{')+len(r'\caption{')
    depth=1;end=start
    while depth:
        char=s[end]
        if char in '{}':
            escaped=(len(s[:end])-len(s[:end].rstrip('\\')))%2
            if not escaped:depth+=1 if char=='{' else -1
        end+=1
    return s[:start]+new+s[end-1:]

def localize_environment(match):
    s=match[0];label=re.search(r'\\label\{([^}]+)\}',s)[1]
    if label in CAPTIONS:s=replace_caption(s,CAPTIONS[label])
    if s.startswith(r'\begin{table}'):
        lines=[]
        for line in s.splitlines():
            if '&' in line and not line.startswith(r'\caption'):
                parts=line.split('&');new=[]
                for cell in parts:
                    prefix=cell[:len(cell)-len(cell.lstrip())]
                    suffix=cell[len(cell.rstrip()):]
                    core,ending=cell_parts(cell)
                    new.append(prefix+CELLS.get(core,core)+ending+suffix)
                line='&'.join(new)
            lines.append(line)
        s='\n'.join(lines)
    # Flush the tall learning-curve float before the following fixed table.
    if label == 'tab:selection':
        s = '\\clearpage\n' + s
    return s
body=re.sub(r'\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}',localize_environment,body,flags=re.S)
body=re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{)results/figures/MAKE(?:51|52|53)/',
            lambda m:m[1]+'results/figures/NN53/',body)
def labels(s):return Counter(re.findall(r'\\label\{([^}]+)\}',s))
def citations(s):return Counter(x for m in re.findall(r'\\cite\{([^}]+)\}',s) for x in m.split(','))
eng_body=english[english.index(r'\section{Introduction}'):english.index(r'\reftitle')]
assert labels(body)==labels(eng_body),(labels(body)-labels(eng_body),labels(eng_body)-labels(body))
assert citations(body)==citations(eng_body),(citations(body)-citations(eng_body),citations(eng_body)-citations(body))
assert by_label(environments(body,'equation')+environments(body,'align'))==maths
# Every displayed numeric table cell is copied, including signs, decimals and SDs.
table_checks=[]
for label,s in by_label(environments(body,'table')).items():
    def nums(t):
        rows=[line for line in t.splitlines() if '&' in line and not line.startswith(r'\caption')]
        joined='\n'.join(rows)
        words={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','twelve':'12','fourteen':'14'}
        joined=re.sub(r'\b(one|two|three|four|five|six|seven|twelve|fourteen)\b',
                      lambda m:words[m[0].lower()],joined,flags=re.I)
        return re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',joined)
    a,b=nums(tables[label]),nums(s)
    assert a==b,(label,a,b)
    table_checks.append(label)

preamble=r'''% !TeX program = lualatex
% MAKE53の全訳。数式・数値・引用・図表番号は英語版に対応する。
% 再コンパイル：lualatex main_paper_NN53.tex（参照確定まで2回）
\documentclass[11pt,a4paper]{ltjsarticle}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{fix-cm}
\usepackage{amsmath,amsthm,amssymb,mathtools}
\usepackage{graphicx,booktabs,array,multirow,tabularx,here}
\usepackage{enumitem}
\usepackage[margin=23mm]{geometry}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{caption}
\usepackage{etoolbox}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}[Scale=MatchLowercase]
\captionsetup{font=small,labelfont=bf}
\setlist[enumerate]{leftmargin=2em,itemsep=2pt,topsep=4pt}
\emergencystretch=2em
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.12}
\newtheorem{Algorithm}{アルゴリズム}
\appto{\appendix}{%
 \renewcommand{\thesection}{\Alph{section}}%
 \setcounter{table}{0}\renewcommand{\thetable}{A\arabic{table}}%
 \renewcommand{\theHtable}{A\arabic{table}}%
 \setcounter{figure}{0}\renewcommand{\thefigure}{A\arabic{figure}}%
 \renewcommand{\theHfigure}{A\arabic{figure}}%
 \setcounter{equation}{0}\renewcommand{\theequation}{A\arabic{equation}}%
 \renewcommand{\theHequation}{A\arabic{equation}}%
 \setcounter{Algorithm}{0}\renewcommand{\theAlgorithm}{A\arabic{Algorithm}}%
 \renewcommand{\theHAlgorithm}{A\arabic{Algorithm}}}
\renewcommand{\refname}{参考文献}
\AtBeginEnvironment{thebibliography}{\interlinepenalty=10000}
\hypersetup{pdftitle={変分自己符号化器の内在次元誘導によるボトルネック選択と活性ユニット診断},
 pdfauthor={小端 千佳，代 美月，神野 健哉},pdflang={ja-JP}}
\title{変分自己符号化器の内在次元誘導による\\ボトルネック選択と活性ユニット診断}
\author{小端 千佳，代 美月，神野 健哉\\
東京都市大学 情報工学部 知能情報工学科\\
{\small 〒158-8557 東京都世田谷区玉堤1-28-1}\\
{\small g2332016@tcu.ac.jp（C.O.），g2591402@tcu.ac.jp（M.D.）}\\
{\small 連絡著者：kjinno@tcu.ac.jp，電話：+81-3-5707-0104（K.J.）}}
\date{}
\begin{document}
\maketitle
\begin{abstract}
内在次元推定と活性ユニット診断は，変分自己符号化器のボトルネック選択の異なる側面を扱う。
4種類の画像データセット，5学習シード，5正則化重みについて，検証品質目標と共通の保存候補費用を用いて探索方策を評価する。
開始次元数のアブレーションでは，信頼性判定，fallback，参照群の費用計上を固定する。
IDが採用されるのはMNISTの25条件だけである。ID由来の開始点と判定付き中点は同じ次元数を選び，
平均要求候補数はそれぞれ8.84件，5.84件である。残る75条件は同じ昇順fallbackを用いる。
100種類の係数・閾値組合せと100種類のデータセット・重み・シード条件からなる10,000件のreplay評価で，
局所探索の155出力が品質を満たす最小次元を逃す。
各データセット3シードの剪定実験では，選択されたnativeモデルの再構成と，通常の変分自己符号化器への次元転用を区別する。
ローカルな関連度事前分布の変種では，非選択軸のマスクによりdSpritesの検証誤差が126.3\%増え，
30 epochsの復号器微調整により，全次元モデルより37.1\%高い水準まで低下する。
全次元モデルを含む4評価条件すべてが，このデータセットの品質目標に届かない。
活性ユニットの予測は，天井に近い課題で追加の利益を確立しない。
これらの知見は選択規則と診断の統制された実証評価を支持するが，一般的な効率改善や剪定の保証を与えない。
\end{abstract}
\noindent\textbf{キーワード：}変分自己符号化器，ボトルネック次元，内在次元，活性ユニット，モデル選択，再構成品質，再現性
\par\smallskip
{\small 本稿は英語版MAKE53の日本語版であり，節・図表・式番号は英語版に対応する。参考文献の書誌情報は原文表記を保持する。}
\par\medskip
'''
bib=english[english.index(r'\begin{thebibliography}'):english.index(r'\end{thebibliography}')+len(r'\end{thebibliography}')]
translation_note=r'''

\paragraph{日本語版の生成}
日本語本文の組立ては\path{src/nn53/build_japanese.py}，
図の表示文字の日本語化は\path{src/nn53/render_figures.py}による。
日本語TeXのコンパイルには，プロジェクト直下で
\texttt{lualatex main\_paper\_NN53.tex}を参照が確定するまで繰り返す。
日本語図は\path{results/figures/NN53/}に保存する。
'''
final=preamble+body+translation_note+'\n\n'+bib+'\n\\end{document}\n'
(ROOT/'main_paper_NN53.tex').write_text(final)
(OUT/'translation_verification.json').write_text(json.dumps({
 'source':'main_paper_MAKE53.tex','source_sha256':hashlib.sha256(english.encode()).hexdigest(),
 'translated_blocks':len(blocks),'labels_match':True,'citation_occurrences_match':True,
 'displayed_equations_identical':len(maths),'numeric_table_cells_identical':table_checks,
 'bibliography_identical':True,'output_sha256':hashlib.sha256(final.encode()).hexdigest()
},indent=2,ensure_ascii=False))
print(f'Wrote main_paper_NN53.tex; {len(blocks)} blocks, {len(maths)} equations and {len(table_checks)} tables checked.')
