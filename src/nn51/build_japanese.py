"""Assemble the complete Japanese MAKE51 translation without changing results."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
from localization import CAPTIONS,CELLS

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src/nn51';OUT=ROOT/'results/nn51';OUT.mkdir(exist_ok=True)
english=(ROOT/'main_paper_MAKE51.tex').read_text()
template=(ROOT/'src/make51/manuscript_body.tex').read_text().strip()
original=template.split('\n\n')
parts=re.split(r'^@@block:(\d+)@@\n',(SRC/'blocks_ja.txt').read_text(),flags=re.M)
blocks={int(parts[i]):parts[i+1].strip() for i in range(1,len(parts),2)}
assert len(original)==128 and set(blocks)==set(range(128))
body='\n\n'.join(original[k] if blocks[k]=='@@keep@@' else blocks[k] for k in range(128))

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
                    core=cell.strip();ending=''
                    if core.endswith(r'\\'):core=core[:-2].rstrip();ending=r' \\'
                    new.append(prefix+CELLS.get(core,core)+ending+suffix)
                line='&'.join(new)
            lines.append(line)
        s='\n'.join(lines)
    return s
body=re.sub(r'\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}',localize_environment,body,flags=re.S)
body=body.replace('{results/figures/MAKE51/','{results/figures/NN51/')
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
        words={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','twelve':'12','fourteen':'14'}
        joined=re.sub(r'\b(one|two|three|four|five|six|twelve|fourteen)\b',
                      lambda m:words[m[0].lower()],joined,flags=re.I)
        return re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',joined)
    a,b=nums(tables[label]),nums(s)
    assert a==b,(label,a,b)
    table_checks.append(label)

preamble=r'''% !TeX program = lualatex
% MAKE51の全訳。数式・数値・引用・図表番号は英語版に対応する。
% 再コンパイル：lualatex main_paper_NN51.tex（参照確定まで2回）
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
\hypersetup{pdftitle={内在次元誘導によるボトルネック次元選定の再検証：探索方策，品質目標，診断の限界},
 pdfauthor={Chika Obata, Mizuki Dai, Kenya Jin'no},pdflang={ja-JP}}
\title{内在次元誘導によるボトルネック次元選定の再検証：\\探索方策，品質目標，診断の限界}
\author{小幡 千佳，大井 瑞生，神野 健哉\\
東京都市大学 情報工学部 知能情報工学科\\
{\small 〒158-8557 東京都世田谷区玉堤1-28-1}\\
{\small g2332016@tcu.ac.jp（C.O.），g2591402@tcu.ac.jp（M.D.）}\\
{\small 連絡著者：kjinno@tcu.ac.jp，電話：+81-3-5707-0104（K.J.）}}
\date{}
\begin{document}
\maketitle
\begin{abstract}
潜在次元の選択は，診断情報とそれを使う探索方策の双方に依存する。
4種類の画像データセット，5学習シード，5正則化重みの保存済みVAE候補に対して，6方策を事後的に比較する。
検証アンカーで品質目標を定め，単一の統一時間スナップショットで各要求に費用を付ける。
従来の内在次元探索窓には構造的に冗長な処理が含まれるため，費用差だけでは次元情報の価値を識別できない。
改訂した局所方策と対応する中点対照は，要求数と最小適格次元のトレードオフを示す。
引き継いだ既定値では，IDを初期値とする方策は全100条件で全グリッド最小値に一致し，中点対照は98条件で一致する。
ただし前者は3データセットで昇順fallbackを使う。
係数・閾値を変えた事後的な10,000設定では，局所探索の155出力が品質を満たしながらグリッド最小値を逃す。
参照モデル群の取得と再利用によって費用の解釈は変わる。
新しいMNISTの1シード実験ではMonte Carlo ELBOによる選択を評価し，剪定変更実装は別の実装監査として扱う。
活性ユニットによる予測は，天井に近い課題で追加の利得を確立しない。
これらの結果は，次元選択法の普遍的順位ではなく，探索・品質・費用の仕様を明示する必要性を支持する。
\end{abstract}
\noindent\textbf{キーワード：}変分自己符号化器，潜在次元，内在次元，活性ユニット，モデル選択，再構成品質，再現性
\par\smallskip
{\small 本稿は英語版MAKE51の日本語版であり，節・図表・式番号は英語版に対応する。参考文献の書誌情報は原文表記を保持する。}
\par\medskip
'''
bib=english[english.index(r'\begin{thebibliography}'):english.index(r'\end{thebibliography}')+len(r'\end{thebibliography}')]
translation_note=r'''

\paragraph{日本語版の生成}
日本語本文の組立ては\path{src/nn51/build_japanese.py}，
図の表示文字の日本語化は\path{src/nn51/render_figures.py}による。
日本語TeXのコンパイルには，プロジェクト直下で
\texttt{lualatex main\_paper\_NN51.tex}を参照が確定するまで繰り返す。
日本語図は\path{results/figures/NN51/}に保存する。
'''
final=preamble+body+translation_note+'\n\n'+bib+'\n\\end{document}\n'
(ROOT/'main_paper_NN51.tex').write_text(final)
(OUT/'translation_verification.json').write_text(json.dumps({
 'source':'main_paper_MAKE51.tex','source_sha256':hashlib.sha256(english.encode()).hexdigest(),
 'translated_blocks':len(blocks),'labels_match':True,'citation_occurrences_match':True,
 'displayed_equations_identical':len(maths),'numeric_table_cells_identical':table_checks,
 'bibliography_identical':True,'output_sha256':hashlib.sha256(final.encode()).hexdigest()
},indent=2,ensure_ascii=False))
print(f'Wrote main_paper_NN51.tex; {len(blocks)} blocks, {len(maths)} equations and {len(table_checks)} tables checked.')
