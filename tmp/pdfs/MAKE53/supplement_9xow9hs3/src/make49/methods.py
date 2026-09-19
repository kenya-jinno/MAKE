"""段階 4 の比較法。設定表 §5.3 と FONDUE 原著（arXiv:2209.12806v1）に従う。

各方法は「参照モデルから選択・最終学習まで」を 1 回実行し、
選択 m・品質・**総費用**・終了状態を返す。

費用の公平性（方針 §5.3）:
  どの方法も「最終評価できる選択済みモデルが得られるまで」の費用に揃える。
  参照 AE 学習・ID 推定・候補学習・anchor 学習・再探索・probe 学習・評価をすべて含める。
  同じ方法の実行中に同じ (m, epochs) を再要求した場合は memoisation で無料にする
  （FONDUE Algorithm 2 の GET-MEM に相当）。
"""
import numpy as np

from .common import CFG

GRIDS = CFG['candidate_grids']
TERM = CFG['budget_and_termination']
MAX_RUNS = TERM['max_training_runs_per_dataset_method_seed']
EPS = CFG['quality_criterion_Q']['epsilon_D']


class Ledger:
    """1 方法・1 seed 分の費用台帳。同じ要求は 1 回だけ課金する（memoisation）。"""

    def __init__(self, cache, seed, wallclock_limit):
        self.cache, self.seed = cache, seed
        self.limit = wallclock_limit
        self.seconds = 0.0
        self.runs = 0
        self.aux_runs = 0
        self.charged = set()
        self.trace = []
        self.exhausted = False

    def _charge(self, key, seconds, kind):
        if key in self.charged:
            return False                      # memoisation: 再課金しない
        self.charged.add(key)
        self.seconds += seconds
        if kind == 'vae':
            self.runs += 1          # ラン上限は「選択対象のモデル族の学習」のみを数える
        else:
            self.aux_runs += 1      # 参照 AE・probe は実時間にのみ計上する
        self.trace.append({'key': key, 'kind': kind, 'seconds': seconds})
        if self.runs > MAX_RUNS or self.seconds > self.limit:
            self.exhausted = True
        return True

    def vae(self, m, epochs=None, early_stopping=True):
        ep = epochs or CFG['budget_and_termination']['max_epochs']
        r = self.cache.get_vae(m, self.seed, epochs=ep, early_stopping=early_stopping)
        self._charge(f'vae|m{m}|e{ep}|es{int(early_stopping)}|b{self.cache.beta:g}',
                     r['elapsed_seconds'], 'vae')
        return r

    def ae(self, m_ref, seed=None, epochs=100):
        s = self.seed if seed is None else seed
        r = self.cache.get_ae(m_ref, s, epochs=epochs)
        self._charge(f'ae|m{m_ref}|s{s}|e{epochs}', r['elapsed_seconds'], 'ae')
        return r

    def probe(self, run, m):
        p = run.get('probe')
        if p:
            self._charge(f'probe|m{m}', p['probe_seconds'], 'probe')
        return p

    def result(self, selected_m, state, extra=None):
        out = {'selected_m': selected_m,
               'termination_state': 'BUDGET_EXHAUSTED' if self.exhausted else state,
               'total_seconds': self.seconds, 'training_runs': self.runs,
               'auxiliary_runs': self.aux_runs,
               'budget_exhausted': self.exhausted, 'trace': self.trace}
        if extra:
            out.update(extra)
        return out


# ── 品質条件 Q ──────────────────────────────────────────────────────────
def quality_target(anchor_runs, X_val, rel=None):
    """T_D = D_anchor + max(rel*D_anchor, 3*SD_seed, abs_floor*Var(X_val))。

    SD_seed は anchor を複数 seed で学習した場合のみ使える。
    1 seed しかない場合は第 2 項を落とし、その旨を返す。
    """
    rel = EPS['rel_primary'] if rel is None else rel
    ds = [r['val']['mse'] for r in anchor_runs]
    Da = float(np.mean(ds))
    terms = {'relative': rel * Da,
             'abs_floor': EPS['abs_floor'] * float(np.var(X_val))}
    if len(ds) > 1:
        terms['three_sd'] = 3 * float(np.std(ds, ddof=1))
    eps = max(terms.values())
    return {'D_anchor': Da, 'epsilon_D': eps, 'T_D': Da + eps, 'terms': terms,
             'sd_available': len(ds) > 1}


def smallest_satisfying(evaluated, T_D):
    """検証済み候補のうち Q を満たす最小の m。探索していない次元は含めない。"""
    ok = sorted([m for m, r in evaluated.items() if r['val']['mse'] <= T_D])
    return ok[0] if ok else None


# ── B1/B2/B3: 全探索 ────────────────────────────────────────────────────
def B1_full_grid_mse(led, grid, **_):
    ev = {m: led.vae(m) for m in grid}
    best = min(ev, key=lambda m: ev[m]['val']['mse'])
    return led.result(best, 'SUCCESS', {'evaluated': sorted(ev), 'criterion': 'val MSE 最良'})


def B2_full_grid_elbo(led, grid, **_):
    ev = {m: led.vae(m) for m in grid}
    best = max(ev, key=lambda m: ev[m]['val']['elbo'])
    return led.result(best, 'SUCCESS', {'evaluated': sorted(ev), 'criterion': 'val ELBO 最良'})


def B3_full_grid_downstream(led, grid, **_):
    ev = {m: led.vae(m) for m in grid}
    accs = {}
    for m, r in ev.items():
        p = led.probe(r, m)
        if p:
            accs[m] = p['val_acc']
    if not accs:
        return led.result(None, 'ID_UNRELIABLE', {'note': 'ラベルなし'})
    best = max(accs, key=lambda m: accs[m])
    return led.result(best, 'SUCCESS', {'evaluated': sorted(ev), 'val_acc': accs,
                                        'criterion': 'validation accuracy 最良'})


# ── B4: ID を使わない粗い探索 ───────────────────────────────────────────
def coarse_subset(grid, k=4):
    """対数間隔で k 点を選ぶ（同じ有限グリッドから）。"""
    idx = np.unique(np.round(np.geomspace(1, len(grid), k)).astype(int) - 1)
    return [grid[i] for i in idx]


def B4_coarse(led, grid, **_):
    cand = coarse_subset(grid)
    ev = {m: led.vae(m) for m in cand}
    best = min(ev, key=lambda m: ev[m]['val']['mse'])
    return led.result(best, 'SUCCESS', {'evaluated': cand, 'criterion': '粗いグリッドの val MSE 最良'})


# ── B5: 固定次元 ────────────────────────────────────────────────────────
def B5_fixed(led, grid, fixed=32, **_):
    m = fixed if fixed in grid else min(grid, key=lambda g: abs(g - fixed))
    led.vae(m)
    return led.result(m, 'SUCCESS', {'evaluated': [m], 'criterion': f'固定 m={m}'})


# ── B6a: FONDUE Algorithm 1（原著の擬似コードどおり）────────────────────
def B6a_fondue(led, grid, d0=None, fondue_epochs=1, **_):
    """l<-0, u<-inf, p<-IDE_data, threshold<-0.2*IDE_data（一度だけ計算）。

    閾値は探索中の p に応じて変えない（U13、原著 Algorithm 1 行 6）。
    """
    p = max(1, int(round(d0)))
    threshold = 0.2 * d0
    l, u = 0, float('inf')
    path = []
    while p != l:
        r = led.vae(p, epochs=fondue_epochs, early_stopping=False)
        gap = r['readouts']['C']['gap']
        path.append({'p': p, 'gap': gap, 'threshold': threshold, 'ok': gap <= threshold})
        if gap <= threshold:
            l = p
            p = int(min(p * 2, u)) if np.isfinite(u) else p * 2
        else:
            u = p
            p = int(np.floor((l + u) / 2))
        if led.exhausted or len(path) > 40:
            break
    return led.result(p, 'SUCCESS', {'search_path': path, 'threshold': threshold,
                                     'fondue_epochs': fondue_epochs,
                                     'criterion': 'FONDUE Algorithm 1 (20% 規則)'})


# ── B6b: FONDUE-VAR Algorithm 3（付録 G）────────────────────────────────
def B6b_fondue_var(led, grid, d0=None, fondue_epochs=1, keep_mixed=True, **_):
    """l <- 2*data_ide、passive/mixed が出るまで容量を倍増する。"""
    l = max(2, int(round(2 * d0)))
    n, path = -1, []
    while n < 0:
        r = led.vae(l, epochs=fondue_epochs, early_stopping=False)
        A = r['readouts']['A']
        av, mv, pv = A['active'], A['mixed'], A['passive']
        path.append({'l': l, 'active': av, 'mixed': mv, 'passive': pv})
        if pv > 0 and keep_mixed:
            n = av + mv
        elif (mv > 0 or pv > 0) and not keep_mixed:
            n = av
        else:
            l *= 2
        if led.exhausted or len(path) > 20:
            break
    return led.result(n if n >= 0 else None,
                      'SUCCESS' if n >= 0 else 'BUDGET_EXHAUSTED',
                      {'search_path': path, 'keep_mixed': keep_mixed,
                       'fondue_epochs': fondue_epochs,
                       'criterion': 'FONDUE-VAR Algorithm 3 (付録 G)'})


# ── 提案法: ID 誘導 + 品質条件 Q ────────────────────────────────────────
def proposed(led, grid, d0=None, X_val=None, record_au=True, m_refs=None,
             ref_seeds=(42, 123, 777), **_):
    """設定表 §5.2/§5.3 と Algorithm 1 の骨子に従う。

    AU は選択の分岐に使わない（設定表 §1.1 の仕様）。record_au=False は
    「本手法 −AU」対照で、選択 m と品質は変わらず AU 算出分の時間だけが減る。
    """
    # 1. 参照 AE と ID 推定
    refs = []
    for mr in m_refs:
        for s in ref_seeds:
            refs.append(led.ae(mr, seed=s))
    by_ref = {}
    for mr in m_refs:
        vals = [r['twonn'] for r in refs if r['m_ref'] == mr]
        by_ref[mr] = float(np.mean(vals))
    d_hats = list(by_ref.values())
    rel_range = (max(d_hats) - min(d_hats)) / max(np.median(d_hats), 1e-12)
    tw = float(np.mean(d_hats))
    ml = float(np.mean([r['mle'] for r in refs]))
    agree = abs(tw - ml) / max(tw, 1e-12)
    crit = CFG['reference_and_id']
    stable = rel_range <= crit['id_stability_criterion']['max_relative_range']
    agreed = agree <= crit['estimator_agreement_criterion']['max_relative_diff']
    id_info = {'d_hat_by_m_ref': by_ref, 'relative_range': rel_range,
               'estimator_rel_diff': agree, 'stable': bool(stable), 'agreed': bool(agreed),
               'd_hat': tw, 'mle': ml}

    if not (stable and agreed):
        # 事前宣言した通常の validation 探索（B4）へ切り替え、費用は全額計上する
        fb = B4_coarse(led, grid)
        fb['termination_state'] = 'ID_UNRELIABLE'
        fb['fallback'] = True
        fb['id_info'] = id_info
        return fb

    # 2. 初期窓と anchor
    c = CFG['initial_window']['c_primary']
    init = next((g for g in grid if g >= c * tw), grid[-1])
    anchor = grid[-1]
    ev = {}
    anchor_run = led.vae(anchor); ev[anchor] = anchor_run
    q = quality_target([anchor_run], X_val)

    # 3. 初期窓 W の候補を**全点**評価する（Algorithm 1 手順 4 に忠実。短絡しない）
    window = [g for g in grid if g <= init]
    for m in sorted(window):
        if m in ev:
            continue
        ev[m] = led.vae(m)
        if led.exhausted:
            break

    # 4. W 内に Q を満たす候補が無ければ、固定した拡張規則で窓の外へ広げる（手順 6）
    if smallest_satisfying(ev, q['T_D']) is None:
        for m in [g for g in grid if g > init]:
            if m in ev or led.exhausted:
                continue
            ev[m] = led.vae(m)
            if ev[m]['val']['mse'] <= q['T_D']:
                break

    sel = smallest_satisfying(ev, q['T_D'])
    if sel is None:
        state = 'NO_COMPACT_ALTERNATIVE_FOUND'
        sel = anchor
    elif sel == anchor:
        state = 'NO_COMPACT_ALTERNATIVE_FOUND'
    else:
        state = 'SUCCESS'
    au = {m: r['readouts']['B']['au'] for m, r in ev.items()} if record_au else None
    return led.result(sel, state, {'evaluated': sorted(ev), 'id_info': id_info,
                                   'quality': q, 'initial_candidate': init,
                                   'anchor': anchor, 'au_diagnostics': au,
                                   'record_au': record_au,
                                   'criterion': 'ID 誘導 + 品質条件 Q'})


# ── +Q 対照 ─────────────────────────────────────────────────────────────
def with_Q(base_fn, name):
    """既存法に提案法と同じ Q と追加探索規則を足す（設定表 §5.3）。"""
    def wrapped(led, grid, X_val=None, **kw):
        base = base_fn(led, grid, X_val=X_val, **kw)
        anchor = grid[-1]
        anchor_run = led.vae(anchor)
        q = quality_target([anchor_run], X_val)
        ev = {anchor: anchor_run}
        start = base['selected_m']
        if start is None:
            return led.result(None, 'BUDGET_EXHAUSTED', {'base': name})
        # 選択点をグリッドに合わせ、Q を満たすまで拡張する
        cands = [g for g in grid if g >= min(start, anchor)]
        for m in cands:
            ev[m] = led.vae(m)
            if ev[m]['val']['mse'] <= q['T_D'] or led.exhausted:
                break
        sel = smallest_satisfying(ev, q['T_D'])
        state = 'SUCCESS' if sel is not None and sel != anchor else 'NO_COMPACT_ALTERNATIVE_FOUND'
        return led.result(sel if sel is not None else anchor, state,
                          {'base_method': name, 'base_selected_m': start,
                           'quality': q, 'evaluated': sorted(ev),
                           'criterion': f'{name} + Q'})
    return wrapped


# ── B9: PCA / Minka（補助的対照、設定表 §5.3）──────────────────────────
def B9_pca(led, grid, X_val=None, X_est=None, variance_ratio=0.95, **_):
    """分散比基準と Minka の MLE で有効次元を得る。

    **PCA 次元を真の ID としない**（設定表 §5.3）。目的関数が VAE と異なるため、
    学習損失の値を VAE 系と直接順位づけしない。共通の品質・費用でのみ比較する。
    PCA 自体は CPU で安価だが、選択後の最終学習は他方法と同様に課金される。
    """
    import time as _t
    from sklearn.decomposition import PCA
    X = X_est.reshape(len(X_est), -1)
    t0 = _t.time()
    p_full = PCA().fit(X)
    cum = np.cumsum(p_full.explained_variance_ratio_)
    d_var = int(np.searchsorted(cum, variance_ratio) + 1)
    try:
        d_mle = int(PCA(n_components='mle').fit(X).n_components_)
    except Exception:
        d_mle = None
    led._charge('pca|fit', _t.time() - t0, 'aux')
    # 候補グリッド上の最も近い点に丸めて最終学習へ渡す
    sel = min(grid, key=lambda g: abs(g - d_var))
    led.vae(sel)
    return led.result(sel, 'SUCCESS',
                      {'pca_variance_dim': d_var, 'pca_minka_dim': d_mle,
                       'variance_ratio': variance_ratio, 'evaluated': [sel],
                       'criterion': f'PCA 累積寄与率 {variance_ratio:.0%}（Minka も併記）',
                       'caveat': 'PCA 次元を真の ID とはみなさない'})


# ── 昇順走査 + Q（ID なし）: ID 誘導の増分価値を直接測る対照 ──────────────
def ascending_scan_Q(led, grid, X_val=None, **_):
    """候補グリッドを**小さい順に**走査し、$Q$ を満たした最初の点で止める。

    提案法と**同じ目的関数**（検証済み候補内で Q を満たす最小の m）を、
    **ID 推定・参照 AE を一切使わずに**達成する対照。
    提案法との差は「ID 誘導があるか」だけになるので、ID の増分価値を直接測れる。
    """
    anchor = grid[-1]
    anchor_run = led.vae(anchor)
    q = quality_target([anchor_run], X_val)
    ev = {anchor: anchor_run}
    for m in sorted(grid):
        if m in ev:
            continue
        ev[m] = led.vae(m)
        if ev[m]['val']['mse'] <= q['T_D'] or led.exhausted:
            break
    sel = smallest_satisfying(ev, q['T_D'])
    if sel is None:
        sel, state = anchor, 'NO_COMPACT_ALTERNATIVE_FOUND'
    elif sel == anchor:
        state = 'NO_COMPACT_ALTERNATIVE_FOUND'
    else:
        state = 'SUCCESS'
    return led.result(sel, state, {'evaluated': sorted(ev), 'quality': q,
                                   'anchor': anchor,
                                   'criterion': '昇順走査 + Q（ID 推定なし）'})
