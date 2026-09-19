"""段階 5 / E4: AU の増分価値が「AU 固有」か「特徴量が 3 本増えただけ」かを切り分ける対照。

事前登録（設定表 §1.1 の方式 2）には**含まれていなかった**頑健性確認である。
主判定を置き換えるものではなく、主判定の解釈可能性を検証するために追加した。

雑音特徴量 3 本（標準正規乱数、AU 特徴量と同数）を S_base に足した場合の ΔAUROC を、
AU 特徴量 3 本を足した場合と比較する。

実行: プロジェクト直下で python3 src/make49/stage5_e4_au_noise_control.py
"""
import sys, json, os
sys.path.insert(0,'/home/kjinno/claude/manifold/manifold'); os.chdir('/home/kjinno/claude/manifold/manifold')
import numpy as np
import src.make49.stage5_e4_au_value as A

iid = json.load(open('results/make49/input_space_id.json'))
tables = {ds: A.build_table(ds, iid[ds]['twonn']) for ds in A.DEV + A.HELD_OUT}
dev = [r for ds in A.DEV for r in tables[ds]]
C = 10.0
rng = np.random.default_rng(7)

# 雑音特徴量 3 本を全行に付与（AU と同数）
for ds, t in tables.items():
    for r in t:
        for j in range(3):
            r[f'noise{j}'] = float(rng.standard_normal())
NOISE = ['noise0','noise1','noise2']

print(f"{'未使用条件':13s} {'base':>7s} {'+AU':>7s} {'+雑音3本':>9s} {'ΔAU':>7s} {'Δ雑音':>7s} {'Y=1率':>6s}")
dAU, dNZ = [], []
for ds in A.HELD_OUT:
    te = tables[ds]
    b = A.fit_eval(dev, te, A.BASE_FEATS, C)
    a = A.fit_eval(dev, te, A.BASE_FEATS + A.AU_FEATS, C)
    n = A.fit_eval(dev, te, A.BASE_FEATS + NOISE, C)
    y = np.mean([r['Y'] for r in te])
    print(f"{ds:13s} {b['auroc']:7.3f} {a['auroc']:7.3f} {n['auroc']:9.3f} "
          f"{a['auroc']-b['auroc']:+7.3f} {n['auroc']-b['auroc']:+7.3f} {y:6.2f}")
    for s in sorted({r['seed'] for r in te}):
        sub=[r for r in te if r['seed']==s]
        bb=A.fit_eval(dev,sub,A.BASE_FEATS,C); aa=A.fit_eval(dev,sub,A.BASE_FEATS+A.AU_FEATS,C)
        nn=A.fit_eval(dev,sub,A.BASE_FEATS+NOISE,C)
        if bb and aa and nn:
            dAU.append(aa['auroc']-bb['auroc']); dNZ.append(nn['auroc']-bb['auroc'])

def ci(v):
    v=np.array(v); bs=[np.mean(v[rng.integers(0,len(v),len(v))]) for _ in range(2000)]
    return np.percentile(bs,[2.5,97.5]), v.mean()

(cl,ch),m1 = ci(dAU); (nl,nh),m2 = ci(dNZ)
print(f"\nΔAUROC(AU)   平均 {m1:+.3f}  95%CI [{cl:+.3f}, {ch:+.3f}]")
print(f"ΔAUROC(雑音) 平均 {m2:+.3f}  95%CI [{nl:+.3f}, {nh:+.3f}]")
print(f"\n→ AU の改善が雑音 3 本の改善を上回るか: "
      f"{'はい（AU 固有の寄与がある）' if cl > nh else 'いいえ（特徴量数の効果と区別できない）'}")
print(f"\n注意: 基底予測器の AUROC は dSprites 0.500（偶然水準）、CIFAR10 0.583 と弱い。")
print(f"      開発条件(MNIST)の Y=1 率 0.82 に対し dSprites 0.29 と偏りが大きく、転移が難しい。")
