# 实验三续（top-K pricing 规模的列池富化）结果摘要

## 1. 配置诊断（Step 0 结果）

- 三个规模的 baseline_pool_complete 全部为 False：YES
- num_iters 全部 ≥ 5：YES
- is_restricted 全部为 False：YES
- 所有硬性门禁通过：YES

## 2. 主表

| N | mode | Baseline pool | Enriched pool (1%) | Cols added | Z_baseline | Z_enriched | Improvement (%) | n_plans_changed | n_mode_switches | equivalence_type | cols_used_from_enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | top-K diag | 9422.5 ± 447.7 | 9915.4 ± 437.6 | 492.9 ± 162.2 | 183337.84 ± 4234.79 | 183337.84 ± 4234.79 | 0.0000 ± 0.0000 | 81.3 ± 6.1 | 0.4 ± 0.8 | alternative_optima | 4.8 ± 1.7 |
| 200 | top-K | 17033.3 ± 657.7 | 17895.6 ± 916.9 | 862.3 ± 411.5 | 535891.97 ± 7703.35 | 535891.97 ± 7703.35 | 0.0000 ± 0.0000 | 82.5 ± 4.8 | 5.8 ± 2.4 | alternative_optima | 4.2 ± 1.8 |
| 500 | top-K | 39933.1 ± 788.8 | 42016.6 ± 2185.4 | 2083.5 ± 1644.5 | 1690831.88 ± 13429.77 | 1690831.88 ± 13429.77 | 0.0000 ± 0.0000 | 79.4 ± 7.1 | 0.2 ± 0.6 | alternative_optima | 3.0 ± 1.9 |

## 3. Equivalence type 分布

| N | ε_loose | # identical | # alternative_optima | # improved |
|---|---|---|---|---|
| 100 | 0.5% | 0/10 | 10/10 | 0/10 |
| 100 | 1.0% | 0/10 | 10/10 | 0/10 |
| 100 | 2.0% | 0/10 | 10/10 | 0/10 |
| 200 | 0.5% | 0/10 | 10/10 | 0/10 |
| 200 | 1.0% | 0/10 | 10/10 | 0/10 |
| 200 | 2.0% | 0/10 | 10/10 | 0/10 |
| 500 | 0.5% | 0/10 | 10/10 | 0/10 |
| 500 | 1.0% | 0/10 | 10/10 | 0/10 |
| 500 | 2.0% | 0/10 | 10/10 | 0/10 |

## 4. N=100 两模式交叉验证

| Metric | N100 full-pool | N100 top-K diagnostic | Delta |
|---|---|---|---|
| Baseline pool size | 37389.7 | 9422.5 ± 447.7 | -27967.2 |
| Objective value | 183337.84 ± 4234.79 | 183337.84 ± 4234.79 | 0.0000% |
| Plan distribution (SP/BS/AE) | see main/full-pool reference | 0.149 / 0.851 / 0.000 | descriptive |

## 5. 三个核心问题的回答

### Q1：top-K 配置下 IRMP 是否仍然最优？
- `identical / alternative_optima / improved` 的总体计数为 0 / 90 / 0。
- 如果没有 improved，说明即便列池不完整，baseline 已经足以触达最优目标值；alternative_optima 只是在目标值层面给出等价替代。

### Q2：富化是否真实工作？
- `columns_used_from_enrichment > 0` 的实例比例为 96.7%。
- 这个指标直接区分了“富化列被实际采用但目标值不变”和“富化列完全没被用到”两种情况。

### Q3：N=100 两模式的解质量是否一致？
- N=100 top-K diagnostic 与 full-pool 主实验的平均目标值差异为 0.0000%。
- 若该差异远低于 0.1%，就能把它作为 top-K 在更大规模上无明显解质量损失的间接证据。

## 6. 对论文的修改建议

### §4.5.3 增补段落建议（核心）
- 明确说明 follow-up 覆盖的是 `N=100 (diagnostic), 200, 500` 的 iterative top-K pricing with a 60-iteration budget，而不是 full-pool shortcut。
- 用 `identical / alternative_optima / improved` 三分法报告结果，避免把“目标值不变但 plan 变化”误写成“富化无效”。
- 如果 `columns_used_from_enrichment > 0` 的比例可观，应明确写出“富化列被 IRMP 实际采用，但目标值保持不变”。

### §2.6 contribution 措辞建议
- 不要写成 novel CG techniques；更稳妥的是强调在论文采用的 top-K iterative pricing 配置下，列池富化没有暴露出可观的目标值改进。

### §5.1.3 实现细节说明的增补
- 建议固定表述为：for N>100, the implementation uses iterative top-K pricing with K=3 and a 60-iteration budget。

### 附录建议
- 把 follow-up 的详细表、equivalence 分布和 N=100 cross-check 放进附录；正文只保留 1 段核心总结。

## 7. 情景判定

- 判定结果：情景 α-prime
- 解释：目标值完全稳定，且显著比例实例真实使用了富化列。
