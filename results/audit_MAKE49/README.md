# MAKE49 段階 1 監査の生成物

`MAKE49_段階1_監査結果.md` の根拠データ。再生成はプロジェクト直下から実行する。

| ファイル | 生成元 | 内容 |
|---|---|---|
| `A2_loss_convention_ablation.json` | `src/audit/A2_loss_convention_ablation.py` | 現行損失 vs 修正損失の AU・MSE 比較（MNIST FC-VAE, β=4, 60ep, seed 42） |
| `A2_beta_scan_corrected_loss.json` / `.log` | `src/audit/A2_beta_scan_corrected_loss.py` | 修正損失のもとでの β スキャン（β∈{0.05,0.1,0.25,0.5,1.0} × m∈{16,32,64}） |

`src/audit/A1_kl_convention_check.py` は依存なしで実行でき、
KL の実装値が標準規約の 1/m であることを数値的に示す。

`src/audit/check_config_consistency.py` は
`MAKE49_実験設定表.md` と `config/experiment_config_MAKE49.json` の整合を検証する（段階 2 の完了条件）。

**注意**: これらは監査用の短い実行（60 epochs・単一 seed）である。
本実験の設定は `MAKE49_実験設定表.md` に従い、300 epochs・5 seeds で行う。
