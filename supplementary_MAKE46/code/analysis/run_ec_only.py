"""EC のみを実行するスクリプト"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.experiments.multi_seed_robustness import run_ec_multiseed

print("EC実験を開始します...")
ec_results = run_ec_multiseed()
print("完了！")
