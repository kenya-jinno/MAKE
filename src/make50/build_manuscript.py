"""Assemble a self-contained-table MAKE50 TeX source from reviewed text/log tables."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]
old=(ROOT/'main_paper_MAKE49.tex').read_text()
pre=old[old.index(r'\documentclass'):old.index(r'\abstract')]
pre=pre.replace('% Notation shorthands (kept in sync with main_paper_NN35.tex)',
                '% Notation shorthands')
pre += r'''
% MAKE50: record-audited revision dated 2026-09-18.
\usepackage{xurl}
\emergencystretch=1.5em
\setlength{\headheight}{20pt}
\abstract{%
Selecting a variational autoencoder's latent width requires distinguishing
representation diagnostics from reconstruction requirements. We compare
intrinsic-dimension guidance, ascending and grid searches, FONDUE variants,
and local pruning adaptations on four image datasets with five candidate
training seeds. A validation anchor defines an explicit reconstruction
target; final ordinary VAEs provide a common dimension-transfer comparison.
We report test error, target attainment, termination outcomes and recorded
training costs, with missing overhead identified. At the primary MNIST
setting, ID guidance and ascending search select the same six-dimensional
model, with training accounts of 715 and 182 seconds. Shared reference-bank
checks reject ID guidance on Fashion-MNIST, dSprites and CIFAR-10, which
therefore use a coarse-search fallback. Selected dimensions vary with
regularisation context, but method orderings reverse across datasets.
The ARD adaptation did not vary its selection-stage weight and cannot
support a zero-sensitivity claim. Active-unit counts depend on their
threshold, and a retrospective paired random-feature comparison does not
establish an AU-specific predictive benefit. These results delimit what
ID and activity diagnostics alone establish about capacity under the
tested architectures, quality targets and implementation choices.}
\keyword{variational autoencoder; latent dimension; intrinsic dimension;
active units; model selection; reconstruction quality; reproducibility}
\begin{document}
'''
body=(ROOT/'src/make50/manuscript_body.tex').read_text()
def table(match):
    return (ROOT/'results/make50'/f'{match[1]}.tex').read_text()
body=re.sub(r'@@table:(\w+)@@',table,body)
a=json.loads((ROOT/'results/make50/au_paired.json').read_text())
mapping={'AU_C':f"{a['C']:g}",'AU_DIFF':f"{a['paired_difference']:+.3f}",
         'AU_BASE_DIFF':f"{a['mean_delta_au']:+.3f}",
         'AU_BASE_LO':f"{a['delta_au_hierarchical_ci'][0]:+.3f}",
         'AU_BASE_HI':f"{a['delta_au_hierarchical_ci'][1]:+.3f}",
         'PYTHON':a['environment']['python'],'NUMPY':a['environment']['numpy'],'SKLEARN':a['environment']['sklearn'],
         'AU_LO':f"{a['paired_hierarchical_ci'][0]:+.3f}",'AU_HI':f"{a['paired_hierarchical_ci'][1]:+.3f}",
         'AU100_DIFF':f"{a['noise100_difference']:+.3f}",
         'AU100_LO':f"{a['noise100_hierarchical_ci'][0]:+.3f}",'AU100_HI':f"{a['noise100_hierarchical_ci'][1]:+.3f}"}
for key,value in mapping.items():body=body.replace('@@'+key+'@@',value)
assert '@@' not in body
refs=old[old.index(r'\begin{thebibliography}'):old.index(r'\end{thebibliography}')]
entries={m[1]:m[2].strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)',refs,re.S)}
entries['fu2019cyclical']=r'''Fu, H.; Li, C.; Liu, X.; Gao, J.; Celikyilmaz, A.; Carin, L.
Cyclical annealing schedule: A simple approach to mitigating KL vanishing.
In \textit{Proceedings of NAACL-HLT 2019}, Minneapolis, MN, USA, 2019;
pp.~240--250. \url{https://doi.org/10.18653/v1/N19-1021}.'''
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
tex='% MDPI MAKE50: revised against MAKE49_manuscript_review.md\n'+pre+body+bib
(ROOT/'main_paper_MAKE50.tex').write_text(tex)
print(f'Wrote main_paper_MAKE50.tex ({len(tex.splitlines())} lines, {len(order)} cited references).')
