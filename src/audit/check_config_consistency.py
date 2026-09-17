"""MAKE49_実験設定表.md と config/experiment_config_MAKE49.json の整合を検証する。
段階 2 の完了条件「設定表と Algorithm・設定ファイルが一致」の機械的チェック。
プロジェクト直下で python3 src/audit/check_config_consistency.py を実行する。
"""
import json, sys

cfg = json.load(open('config/experiment_config_MAKE49.json'))
md  = open('MAKE49_実験設定表.md').read()

checks = [
 ("AU 評価の主方式 = 方式2",
  cfg['au']['evaluation_mode'] == 'primary_method_2_predictive'
  and cfg['au']['method_1_intervention']['used'] is False,
  "方式 2" in md and "方式 1 は実施しない" in md),
 ("AU 判定規則 (ΔAUROC CI 下限 > 0)",
  'lower bound exceeds 0' in cfg['au']['method_2_predictive']['decision_rule'],
  "下限が $0$ を上回る場合に限り" in md),
 ("E7a 主要比較時点 = 5, 300",
  cfg['E7a']['primary_comparison_epochs'] == [5, 300],
  "**(5 epochs, 300 epochs)**" in md),
 ("E7a 全軌跡 = 1,2,5,50,300",
  cfg['E7a']['recorded_epochs'] == [1, 2, 5, 50, 300], "1, 2, 5, 50, 300 epochs" in md),
 ("H2 主比較 = active + mixed",
  cfg['E7a']['readouts']['A_variable_type']['primary_count_for_H2'] == 'active_plus_mixed',
  "主比較は active + mixed" in md),
 ("keep_mixed 主比較 True / 感度 False",
  cfg['fondue_reimplementation']['B6b_algorithm3_fondue_var']['keep_mixed'] is True
  and cfg['fondue_reimplementation']['B6b_algorithm3_fondue_var']['keep_mixed_sensitivity'] is False,
  "`keep_mixed = True`" in md and "`keep_mixed = False`" in md),
 ("epsilon_D 主水準 = 10%",
  cfg['quality_criterion_Q']['epsilon_D']['rel_primary'] == 0.10,
  "0.10 \\cdot D_{\\mathrm{anchor}}" in md),
 ("epsilon_D 感度水準",
  cfg['quality_criterion_Q']['epsilon_D']['rel_sensitivity'] == [0.01, 0.05, 0.25, 0.50],
  "相対 1%, 5%, 25%, 50%" in md),
 ("開発条件の T_D = 0.009831",
  abs(cfg['quality_criterion_Q']['dev_condition_precomputation_pre_A1_fix']['T_D'] - 0.009831) < 1e-9,
  "0.009831" in md),
 ("予算 = 12 ラン",
  cfg['budget_and_termination']['max_training_runs_per_dataset_method_seed'] == 12, "**12 ラン**" in md),
 ("early stopping = 30 epochs",
  cfg['budget_and_termination']['early_stopping_patience_epochs'] == 30, "**30 epochs**" in md),
 ("plateau = 直近 50 epochs / 1% 未満",
  'last 50 epochs' in cfg['budget_and_termination']['plateau_criterion'],
  "**50 epochs**" in md and "**1% 未満**" in md),
 ("終了状態 5 種（段階4で DEGENERATE_SELECTION を追加）",
  len(cfg['budget_and_termination']['termination_states']) == 5,
  all(s in md for s in ['SUCCESS', 'NO_COMPACT_ALTERNATIVE_FOUND', 'ID_UNRELIABLE',
                        'BUDGET_EXHAUSTED', 'DEGENERATE_SELECTION'])),
 ("予算超過を平均から除外しない",
  'NOT be silently dropped' in cfg['budget_and_termination']['aggregation_rule'], "黙って除外しない" in md),
 ("beta 主設定 = 1",
  cfg['beta']['primary'] == 1.0, "$\\beta = 1$" in md),
 ("損失規約 = sum_recon_sum_kl_v1",
  cfg['loss_convention']['id'] == 'sum_recon_sum_kl_v1', "torch.sum(1 + logvar" in md),
 ("GPU 実時間上限が確定済み（段階3校正）",
  isinstance(cfg['budget_and_termination']['max_wallclock_seconds_per_dataset_method_seed'], dict)
  and set(cfg['budget_and_termination']['max_wallclock_seconds_per_dataset_method_seed'])
      >= {'MNIST', 'dSprites', 'CIFAR10'},
  "**これで本設定表に未確定の項目はなくなった。**" in md),
 ("校正実測値が設定表に記載されている",
  'measured 2026-09-12' in cfg['budget_and_termination']['max_wallclock_basis'],
  "0.327" in md and "1.147" in md and "0.650" in md),
 ("E7a は early stopping を適用しない",
  cfg['budget_and_termination']['early_stopping_scope']['does_not_apply_to'] == ['E7a_trajectory_runs'],
  "E7a の軌跡ランは early stopping を適用せず" in md),
 ("ID 推定標本は train から",
  cfg['reference_and_id']['estimation_sample']['source'] == 'train',
  "**train** から固定乱数で 10000 点" in md),
]

ng = 0
for name, in_cfg, in_md in checks:
    ok = in_cfg and in_md
    ng += not ok
    print(f"{'OK ' if ok else 'NG '}{name}   config={in_cfg} 設定表={in_md}")

print("\n整合: " + ("全項目一致" if ng == 0 else f"{ng} 件が不一致"))
sys.exit(1 if ng else 0)
