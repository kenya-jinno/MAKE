"""E2 の代表試行: 学習曲線と過学習・未収束の切り分け（方針 §5.5 E2、R1-9 直結）。

E7a と同じランから作る。守る事項:
  - train MSE も**評価モード**で計算し、validation と同じ再構成方式（決定論的 g(mu(x))）を使う。
  - checkpoint は validation のみで選択し、test は選択済み checkpoint でのみ評価する。
  - 図には train/validation 曲線に**最終 test 値を点で加える**（Reviewer 1 の 3 分割要求への第一候補）。
    全 epoch の test 曲線は描かない。
  - VAE の KL と通常の ELBO は別パネルにする。
  - 白黒で読めるよう線種とマーカーで区別する。

実行: プロジェクト直下で python3 src/make49/stage3_e2_curves.py
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join('results', 'make49')
FIG = os.path.join('results', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

STYLES = {'small': ('-', 'o'), 'initial': ('--', 's'), 'ample': (':', '^')}
LABELS = {'small': 'small', 'initial': 'initial candidate', 'ample': 'ample'}


def agg(runs, key):
    """seed 方向に平均と SD を取る。"""
    a = np.array([r['curves'][key] for r in runs], float)
    return a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])


def main():
    d = json.load(open(os.path.join(OUT, 'stage3_e7a_e2.json')))
    runs = d['runs']
    datasets = sorted({r['dataset'] for r in runs})
    summary = []

    for ds in datasets:
        sub = [r for r in runs if r['dataset'] == ds]
        caps = sorted({(r['capacity_label'], r['m']) for r in sub},
                      key=lambda t: ['small', 'initial', 'ample'].index(t[0]))
        lik = sub[0]['likelihood']
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

        for lab, m in caps:
            rs = [r for r in sub if r['capacity_label'] == lab]
            if not rs:
                continue
            ep = np.array(rs[0]['curves']['epoch'])
            ls, mk = STYLES[lab]
            for ax, key, title in [(axes[0], 'val_mse', None), (axes[1], 'val_kl', None),
                                   (axes[2], 'val_elbo', None)]:
                mean, sdv = agg(rs, key)
                ax.plot(ep, mean, ls, marker=mk, markevery=40, markersize=5,
                        color='k' if lab == 'small' else ('0.45' if lab == 'initial' else '0.7'),
                        label=f'{LABELS[lab]} m={m}')
                ax.fill_between(ep, mean - sdv, mean + sdv, color='0.8', alpha=0.35, linewidth=0)
            # train（評価モード）は同じパネルに細線で重ねる
            tm, _ = agg(rs, 'train_mse')
            axes[0].plot(ep, tm, ls, linewidth=1.0, alpha=0.8,
                         color='k' if lab == 'small' else ('0.45' if lab == 'initial' else '0.7'))
            # 選択済み checkpoint の test 値を点で置く
            te = np.mean([r['test_eval_at_best_checkpoint']['mse_per_pixel_mean'] for r in rs])
            be = int(np.mean([r['best_epoch'] for r in rs]))
            axes[0].plot([be], [te], marker='*', markersize=13, linestyle='none',
                         color='k' if lab == 'small' else ('0.45' if lab == 'initial' else '0.7'))
            summary.append({'dataset': ds, 'capacity': lab, 'm': m,
                            'best_epoch_mean': be,
                            'val_mse_final': float(np.mean([r['curves']['val_mse'][-1] for r in rs])),
                            'train_mse_final': float(np.mean([r['curves']['train_mse'][-1] for r in rs])),
                            'test_mse_at_best': float(te),
                            'plateau_fraction': float(np.mean([r['plateau'] for r in rs]))})

        axes[0].set_xlabel('epoch'); axes[0].set_ylabel('MSE (per-pixel mean)')
        axes[0].set_title(f'{ds}: reconstruction MSE\nthick = validation, thin = train (eval mode), '
                          '* = test at selected checkpoint')
        axes[0].set_yscale('log'); axes[0].legend(fontsize=9)
        axes[1].set_xlabel('epoch'); axes[1].set_ylabel('KL (per sample)')
        axes[1].set_title(f'{ds}: KL term'); axes[1].legend(fontsize=9)
        axes[2].set_xlabel('epoch'); axes[2].set_ylabel('plain ELBO (beta=1)')
        axes[2].set_title(f'{ds}: plain ELBO\n(distinct from the beta-weighted objective)')
        axes[2].legend(fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f'E2 representative trial - {ds} '
                     f'({lik} likelihood, beta=1, 5 seeds, band = +/-SD)', y=1.02)
        fig.tight_layout()
        p = os.path.join(FIG, f'MAKE49_E2_{ds}.pdf')
        fig.savefig(p, bbox_inches='tight'); plt.close(fig)
        print(f'図: {p}')

    print(f"\n{'データ':9s} {'容量':8s} {'m':>4s} {'best_ep':>8s} {'train MSE':>10s} "
          f"{'val MSE':>9s} {'test MSE':>9s} {'plateau率':>9s}")
    for s in summary:
        print(f"{s['dataset']:9s} {s['capacity']:8s} {s['m']:4d} {s['best_epoch_mean']:8d} "
              f"{s['train_mse_final']:10.5f} {s['val_mse_final']:9.5f} "
              f"{s['test_mse_at_best']:9.5f} {s['plateau_fraction']:9.2f}")
    print("\n過学習の判定: train MSE と val MSE の乖離を見る。")
    print("未収束の判定: plateau 率が低い条件は 300 epochs で plateau していない。")
    with open(os.path.join(OUT, 'stage3_e2_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"書き出し: {OUT}/stage3_e2_summary.json")


if __name__ == '__main__':
    main()
