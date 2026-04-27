# 实验三（列池富化）结果摘要

## 1. 实验参数确认

- 规模点：N ∈ {20, 50, 100}
- 场景：U；方法：CG+IRMP；operation_mode：simops
- ε_loose 测试值：0.5%、1%、2% of Z_baseline
- 富化模式：enumeration（基于收敛时 duals 对单船全 plan 枚举）
- 实际复现实例种子沿用主实验配置，为 1..10，而不是重新改成 0..9

## 2. 主表（1% 富化）

| N | Baseline pool | Enriched pool (1%) | Cols added | Z_baseline | Z_enriched | Improvement | Identical solutions | vs MILP |
|---|---|---|---|---|---|---|---|---|
| 20 | 6580.1 | 6580.1 | 0.0 | 24666.41 | 24666.41 | 0.0000% | True | 0.0000% |
| 50 | 17930.2 | 17930.2 | 0.0 | 82695.01 | 82695.01 | 0.0000% | True | 0.0000% |
| 100 | 37389.7 | 37389.7 | 0.0 | 183338.23 | 183338.23 | 0.0000% | True | n/a |

## 3. 三个核心问题

### Q1：原 IRMP 是否已经最优？
- 判定：情景 α。所有富化结果与原 IRMP 完全一致。
- 所有富化记录的平均 improvement 为 0.000000%，最大单实例 improvement 为 0.000000%。
- solutions_identical 的总体成立比例为 100.0%。

### Q2：ε_loose 扩大后的边际效益如何？
- ε=0.5%：平均新增列 0.00，平均 improvement 0.000000%，最大 improvement 0.000000%。
- ε=1.0%：平均新增列 0.00，平均 improvement 0.000000%，最大 improvement 0.000000%。
- ε=2.0%：平均新增列 0.00，平均 improvement 0.000000%，最大 improvement 0.000000%。

### Q3：改变发生在哪里？
- 有 100.0% 的富化记录新增列数为 0。
- baseline_pool_complete 的比例为 100.0%，used_full_pool_small 的比例为 100.0%。
- 这意味着在当前论文主配置下，N≤100 的 baseline pool 实际上已经等于完整单船列枚举池，因此 enrichment 无法再向池中加入新列。

## 4. 与 §5.2 外部 MILP 验证的一致性

- N=20 和 N=50 的 enriched-vs-MILP 平均 gap 分别为 0.000000% 和 0.000000%。
- 在本次复现中，baseline IRMP、enriched IRMP 与 existing exact MILP 在 N=20/50 上保持一致，没有出现 enriched 优于 MILP 的异常。

## 5. 对论文的修改建议

### §4.5.3 增补建议
- 可以报告富化实验结果为零改进，但必须同时说明这是在当前 `use_full_pool_small=true, full_pool_n=100` 的主实验配置下得到的，因此基线列池本身已经是完整枚举池。
- 这条证据能够支撑“当前实现没有遗漏列”的实现完整性，但对 Remark 3 的独立经验支撑力度弱于真正的受限列池情形。

### §2.6 contribution 措辞建议
- 不建议因为本实验去强化“CG 定价天然使列池完整”的表述。
- 更稳妥的写法是：在论文报告的 N≤100 主实验设置下，CG+IRMP 的最终求解并未受到列池截断误差影响；对更一般的受限列池情形，完整 branch-and-price 仍留作未来工作。

### 附录建议
- 把本实验放入附录，重点展示 baseline/enriched pool size、zero-improvement 结果以及 full-pool 配置说明。

## 6. 情景判定

- 判定结果：情景 α
- 解释：所有富化结果与原 IRMP 完全一致。
