"""
実験 Exp-AU-Rule: Step 3 の再現可能な合否判定規則の後検証
Review_codex35 Major 2 / Review_codex36 Major 2 対応

規則（検証点 m_ver >= 2 * dhat で評価）:
  V1（遮断の発生）: AU(m_ver) / m_ver <= 0.9
  V2（オーダー整合）: dhat <= AU(m_ver) <= 3 * dhat
両方成立で PASS．V1 不成立 → beta 不足（表: failure 第1行）．
V2 上側不成立 → Step 1 過小推定または遮断不十分．
V2 下側不成立 → 過正則化（beta 過大）．

v2 追加（codex36 Major 2）:
  - 多シードデータはシード毎にも判定し，全シード一致を確認
  - 判定が不変となる閾値範囲（V1 の 0.9，V2 の係数 3）を計算
v3 追加（codex37 Minor 4）:
  - 閾値不変域を「多シード平均に対する範囲」と「シード毎判定に対する範囲」に
    分けて計算・保存する（per-seed の下端は MNIST m=20 の AU=14 → 0.70）
v4 追加（Peer_Review_Report41 Major 2 対応）:
  - Fashion-MNIST FC-VAE の 3 シード検証（Exp-Fashion-MS）を判定対象に追加
  - Conv-VAE HiBeta を 3 シード（Exp-ConvV-HiBeta-MS）の平均判定に更新
既存データ（新規学習なし）に規則を適用して各設定の判定を確認する．
"""

import json

RULE_R = 0.9
RULE_C = 3.0


def apply_rule(au, m, dhat):
    v1 = au / m <= RULE_R
    v2 = dhat <= au <= RULE_C * dhat
    return v1, v2, (v1 and v2)


def show(name, au, m, dhat, note='', per_seed_au=None):
    v1, v2, ok = apply_rule(au, m, dhat)
    seed_str = ''
    if per_seed_au is not None:
        verdicts = [apply_rule(a, m, dhat)[2] for a in per_seed_au]
        agree = all(v == ok for v in verdicts)
        seed_str = f" per-seed AU={per_seed_au} verdicts={'unanimous' if agree else 'MIXED'}"
    print(f"{name:42s} m={m:4d} AU={au:6.1f} dhat={dhat:5.1f} "
          f"r={au/m:.2f} V1={'o' if v1 else 'x'} V2={'o' if v2 else 'x'} "
          f"=> {'PASS' if ok else 'FAIL'} {note}{seed_str}")
    return {'name': name, 'm': m, 'au': au, 'dhat': dhat,
            'ratio': au / m, 'au_over_dhat': au / dhat,
            'v1': v1, 'v2': v2, 'pass': ok, 'note': note,
            'per_seed_au': per_seed_au}


def main():
    out = []

    # 合成多様体（Exp-Syn-AU, 3シード, sigma=0）
    out.append(show('SwissRoll VAE (dhat=2, m=4=2dhat)', 2.0, 4, 2.0,
                    per_seed_au=[2, 2, 2]))
    out.append(show('Torus VAE (dhat=2, m=4=2dhat)', 2.0, 4, 2.0,
                    note='(AU<demb: over-pruning)', per_seed_au=[2, 2, 2]))
    out.append(show('Torus VAE (dhat=2, m=6=3dhat)', 3.0, 6, 2.0,
                    per_seed_au=[3, 3, 3]))

    # MNIST FC-VAE beta=4（Exp-Mnist-Multi-300, 3シード）
    d = json.load(open('results/tables/E4_300ep_multiseed.json'))
    dhat = 10.0
    for m_ver in [20, 32, 64]:
        aus = [r['au'] for s in ['42', '123', '777'] for r in d[s] if r['m'] == m_ver]
        au = sum(aus) / len(aus)
        out.append(show('MNIST FC-VAE beta=4 (3-seed mean)', au, m_ver, dhat,
                        per_seed_au=aus))

    # Conv-VAE MNIST beta_max=4（Exp-Conv-M256, 3シード）
    en = json.load(open('results/tables/EN_conv_vae_extended_m.json'))
    for m_ver in [64, 256]:
        aus = [r['au'] for s in ['42', '123', '777']
               for r in en[s]['results'] if r['m'] == m_ver]
        au = sum(aus) / len(aus)
        out.append(show('MNIST Conv-VAE beta_max=4 (3-seed mean)', au, m_ver,
                        dhat, per_seed_au=aus))

    # Fashion-MNIST FC-VAE beta=4（Exp-Fashion-MS, 3シード, 300 ep）
    try:
        fm = json.load(open('results/tables/Exp_Fashion_MS.json'))
        m_ver = 20
        aus = [r['au'] for s in ['42', '123', '777'] for r in fm[s] if r['m'] == m_ver]
        au = sum(aus) / len(aus)
        out.append(show('Fashion FC-VAE beta=4 (3-seed mean)', au, m_ver, dhat,
                        per_seed_au=aus))
    except FileNotFoundError:
        print('  (Exp_Fashion_MS.json not found; skipping Fashion entry)')

    # Conv-VAE HiBeta（Exp-ConvV-HiBeta-MS, 3シード）
    try:
        hb = json.load(open('results/tables/Exp_ConvV_HiBeta_MS.json'))
        for beta_max in ['10', '20']:
            aus = [r['au'] for s in ['42', '123', '777']
                   for r in hb[s][beta_max] if r['m'] == 256]
            au = sum(aus) / len(aus)
            out.append(show(f'MNIST Conv-VAE beta_max={beta_max} (3-seed mean)',
                            au, 256, dhat, per_seed_au=aus))
    except FileNotFoundError:
        print('  (Exp_ConvV_HiBeta_MS.json not found; using single-seed values)')
        out.append(show('MNIST Conv-VAE beta_max=10', 93.0, 256, dhat, '(single seed)'))
        out.append(show('MNIST Conv-VAE beta_max=20', 66.0, 256, dhat, '(single seed)'))

    # 自然画像（Exp-Nat-Cifar / SVHN, 3シード）
    out.append(show('CIFAR-10 Conv-VAE beta_max=4', 256.0, 256, 25.2,
                    per_seed_au=[256, 256, 256]))
    out.append(show('SVHN Conv-VAE beta_max=4', 206.7, 256, 20.0,
                    per_seed_au=[206, 207, 207]))

    # ---- 閾値不変範囲（codex36 Major 2）：多シード平均に対する範囲 ----
    passing = [r for r in out if r['pass']]
    failing = [r for r in out if not r['pass']]
    max_pass_ratio = max(r['ratio'] for r in passing)
    min_fail_ratio = min(r['ratio'] for r in failing if not r['v1'])
    max_pass_c = max(r['au_over_dhat'] for r in passing)
    # V2 の係数 c は上側境界のみを動かすため，上側不成立（AU > c*dhat）の行のみ対象
    min_fail_c = min(r['au_over_dhat'] for r in failing
                     if r['v1'] and not r['v2'] and r['au'] > RULE_C * r['dhat'])
    print('\n=== Threshold invariance (multi-seed mean verdicts) ===')
    print(f'V1 threshold: passing max ratio = {max_pass_ratio:.2f}, '
          f'V1-failing min ratio = {min_fail_ratio:.2f} '
          f'-> verdicts invariant for threshold in ({max_pass_ratio:.2f}, {min_fail_ratio:.2f})')
    print(f'V2 factor: passing max AU/dhat = {max_pass_c:.2f}, '
          f'V2-failing min AU/dhat = {min_fail_c:.2f} '
          f'-> verdicts invariant for factor in ({max_pass_c:.2f}, {min_fail_c:.2f})')

    # V2 下側係数（既定 1.0）の不変域：下側不成立の最大 AU/dhat と合格の最小 AU/dhat
    low_fails = [r for r in failing if r['v1'] and not r['v2'] and r['au'] < r['dhat']]
    if low_fails:
        max_faillow_c = max(r['au_over_dhat'] for r in low_fails)
        min_pass_c = min(r['au_over_dhat'] for r in passing)
        print(f'V2 lower coefficient: fail-low max AU/dhat = {max_faillow_c:.2f}, '
              f'passing min AU/dhat = {min_pass_c:.2f} '
              f'-> verdicts invariant for lower coefficient in ({max_faillow_c:.2f}, {min_pass_c:.2f}]')

    # ---- 閾値不変範囲（codex37 Minor 4）：シード毎判定に対する範囲 ----
    def seed_vals(r, kind):
        aus = r['per_seed_au'] if r['per_seed_au'] is not None else [r['au']]
        if kind == 'ratio':
            return [a / r['m'] for a in aus]
        return [a / r['dhat'] for a in aus]

    ps_max_pass_ratio = max(v for r in passing for v in seed_vals(r, 'ratio'))
    ps_min_fail_ratio = min(v for r in failing if not r['v1']
                            for v in seed_vals(r, 'ratio'))
    ps_max_pass_c = max(v for r in passing for v in seed_vals(r, 'c'))
    ps_min_fail_c = min(v for r in failing
                        if r['v1'] and not r['v2'] and r['au'] > RULE_C * r['dhat']
                        for v in seed_vals(r, 'c'))
    print('\n=== Threshold invariance (per-seed verdicts) ===')
    print(f'V1 threshold: per-seed passing max ratio = {ps_max_pass_ratio:.2f}, '
          f'per-seed V1-failing min ratio = {ps_min_fail_ratio:.2f} '
          f'-> per-seed verdicts invariant for threshold in '
          f'({ps_max_pass_ratio:.2f}, {ps_min_fail_ratio:.2f})')
    print(f'V2 factor: per-seed passing max AU/dhat = {ps_max_pass_c:.2f}, '
          f'per-seed V2-failing min AU/dhat = {ps_min_fail_c:.2f} '
          f'-> per-seed verdicts invariant for factor in '
          f'({ps_max_pass_c:.2f}, {ps_min_fail_c:.2f})')

    with open('results/tables/Exp_AU_Rule.json', 'w') as f:
        json.dump({'rule': {'r_max': RULE_R, 'c': RULE_C},
                   'threshold_invariance': {
                       'v1_pass_max': max_pass_ratio,
                       'v1_fail_min': min_fail_ratio,
                       'v2_pass_max': max_pass_c,
                       'v2_fail_min': min_fail_c},
                   'threshold_invariance_per_seed': {
                       'v1_pass_max': ps_max_pass_ratio,
                       'v1_fail_min': ps_min_fail_ratio,
                       'v2_pass_max': ps_max_pass_c,
                       'v2_fail_min': ps_min_fail_c},
                   'results': out}, f, indent=2)
    print('\nSaved to results/tables/Exp_AU_Rule.json')


if __name__ == '__main__':
    main()
