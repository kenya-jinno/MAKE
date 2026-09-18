"""Assemble a self-contained-table MAKE51 TeX source from reviewed text/log tables."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]
old=(ROOT/'main_paper_MAKE49.tex').read_text()
pre=old[old.index(r'\documentclass'):old.index(r'\abstract')]
pre=pre.replace('% Notation shorthands (kept in sync with main_paper_NN35.tex)',
                '% Notation shorthands')
pre=pre.replace('A Common-Protocol Comparison with Existing Criteria',
                'Search Policies, Quality Targets and Diagnostic Limits')
pre += r'''
% MAKE51: record-audited revision dated 2026-09-19.
\usepackage{xurl}
\emergencystretch=1.5em
\setlength{\headheight}{20pt}
\abstract{%
Latent-width selection depends on both diagnostic information and the
search policy using it. We retrospectively compare six policies on stored
variational-autoencoder candidates from four image datasets, five training
seeds and five regularisation weights. A validation anchor defines the
quality target, and one canonical timing snapshot prices every request.
The historical intrinsic-dimension window incurs structurally redundant
work, so its expense contrast alone cannot identify the value of dimension
information. A revised local policy and a matched midpoint control expose
the trade-off between requests and the smallest qualifying width.
At the inherited defaults, the dimension-initialised policy matches the
full-grid minimum in all 100 conditions, while the midpoint control
matches in 98; the former uses ascending fallback on three datasets.
Across 10,000 retrospective coefficient/threshold settings, 155 local
outputs miss the grid minimum despite meeting quality. Reference-bank
acquisition and reuse change the cost interpretation. A new one-seed
MNIST experiment evaluates Monte Carlo ELBO selection, while pruning
adaptations remain separate implementation audits. Active-unit prediction
does not establish an additional benefit on its near-ceiling task.
These results support explicit search, quality and cost specifications
rather than a universal ranking of dimension-selection methods.}
\keyword{variational autoencoder; latent dimension; intrinsic dimension;
active units; model selection; reconstruction quality; reproducibility}
\begin{document}
'''
body=(ROOT/'src/make51/manuscript_body.tex').read_text()
def table(match):
    return (ROOT/'results/make51'/f'{match[1]}.tex').read_text()
body=re.sub(r'@@table:(\w+)@@',table,body)
a=json.loads((ROOT/'results/make51/au_paired.json').read_text())
mapping={'AU_C':f"{a['C']:g}",'AU_DIFF':f"{a['paired_difference']:+.3f}",
         'AU_BASE_DIFF':f"{a['mean_delta_au']:+.3f}",
         'AU_BASE_LO':f"{a['delta_au_hierarchical_ci'][0]:+.3f}",
         'AU_BASE_HI':f"{a['delta_au_hierarchical_ci'][1]:+.3f}",
         'PYTHON':a['environment']['python'],'NUMPY':a['environment']['numpy'],'SKLEARN':a['environment']['sklearn'],
         'AU_LO':f"{a['paired_hierarchical_ci'][0]:+.3f}",'AU_HI':f"{a['paired_hierarchical_ci'][1]:+.3f}",
         'AU100_DIFF':f"{a['noise100_difference']:+.3f}",
         'AU100_LO':f"{a['noise100_hierarchical_ci'][0]:+.3f}",'AU100_HI':f"{a['noise100_hierarchical_ci'][1]:+.3f}"}
mapping.update(json.loads((ROOT/'results/make51/validation_text.json').read_text()))
for key,value in mapping.items():body=body.replace('@@'+key+'@@',value)
assert '@@' not in body
refs=old[old.index(r'\begin{thebibliography}'):old.index(r'\end{thebibliography}')]
entries={m[1]:m[2].strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)',refs,re.S)}
entries['fu2019cyclical']=r'''Fu, H.; Li, C.; Liu, X.; Gao, J.; Celikyilmaz, A.; Carin, L.
Cyclical annealing schedule: A simple approach to mitigating KL vanishing.
In \textit{Proceedings of NAACL-HLT 2019}, Minneapolis, MN, USA, 2019;
pp.~240--250. \url{https://doi.org/10.18653/v1/N19-1021}.'''
entries['galka2022isolation']=r'''Ga{\l}ka, {\L}.; Karczmarek, P.; Tokovarov, M.
Isolation Forest Based on Minimal Spanning Tree.
\textit{IEEE Access} \textbf{2022}, \textit{10}, 74175--74186.
\url{https://doi.org/10.1109/ACCESS.2022.3190505}.'''
entries['levina2005mle']=r'''Levina, E.; Bickel, P.J.
Maximum likelihood estimation of intrinsic dimension.
In \textit{Advances in Neural Information Processing Systems},
\textbf{2004}, \textit{17}.
\url{https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html}.'''
entries['ansuini2019intrinsic']+=r''' \url{https://proceedings.neurips.cc/paper_files/paper/2019/file/cfcce0621b49c983991ead4c3d4d3b6b-Paper.pdf}.'''
entries['lucas2019elbo']+=r''' \url{https://proceedings.neurips.cc/paper/2019/file/7e3315fe390974fcf25e44a9445bd821-Paper.pdf}.'''
entries['pope2021intrinsic']+=r''' arXiv:2104.08894.'''
entries['whitney1944']=r'''Whitney, H.
The self-intersections of a smooth $n$-manifold in $2n$-space.
\textit{Ann. Math.} \textbf{1944}, \textit{45}, 220--246.
\url{https://doi.org/10.2307/1969265}.'''
entries['bonheme2023active']=r'''Bonheme, L.; Grzes, M.
Be more active! Understanding the differences between mean and sampled
representations of variational autoencoders.
\textit{J. Mach. Learn. Res.} \textbf{2023}, \textit{24}(324), 1--30.
\url{https://jmlr.org/papers/v24/21-1145.html}.'''
# Conference identity was checked against the authors' institutional list.
entries['obata2025manifold']=entries['obata2025manifold']+r'''
Publication record: \url{https://www.comm.tcu.ac.jp/nel/list_j_2025.html}
(accessed on 18 September 2026).'''
doi_add={'kingma2014vae':'10.48550/arXiv.1312.6114',
         'louizos2018l0':'10.48550/arXiv.1712.01312',
         'burda2016iwae':'10.48550/arXiv.1509.00519',
         'xiao2017fashion':'10.48550/arXiv.1708.07747',
         'lucas2019elbo':'10.48550/arXiv.1911.02469',
         'ansuini2019intrinsic':'10.48550/arXiv.1905.12784',
         'pope2021intrinsic':'10.48550/arXiv.2104.08894'}
# Only add arXiv identifiers already explicitly present in the inherited entry.
for key,doi in doi_add.items():
    if doi.split('arXiv.')[-1] in entries[key]:
        entries[key]+=f' \\url{{https://doi.org/{doi}}}.'
for key,ent in entries.items():
    ent=re.sub(r'(?<!\{)(https?://[^\s{}]+)',lambda m:r'\url{'+m[1].rstrip('.')+'}'+('.' if m[1].endswith('.') else ''),ent)
    entries[key]=ent
order=[]
for match in re.finditer(r'\\cite\{([^}]+)\}',body):
    for key in match[1].split(','):
        if key not in order:order.append(key)
bib='\n\\reftitle{References}\n\\begin{thebibliography}{999}\n'
for key in order:
    assert key in entries,key
    bib+='\n\\bibitem{'+key+'}\n'+entries[key]+'\n'
bib+='\n\\end{thebibliography}\n\\end{document}\n'
tex='% MDPI MAKE51: revised against MAKE50_revision_review.md\n'+pre+body+bib
(ROOT/'main_paper_MAKE51.tex').write_text(tex)
print(f'Wrote main_paper_MAKE51.tex ({len(tex.splitlines())} lines, {len(order)} cited references).')
