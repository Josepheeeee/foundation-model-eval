# 正式实验登记表

状态：**草稿，未填写实际模型/数据指纹，尚未冻结。**这是操作模板，不是实验完成证明。协议定义见 `EXPERIMENTS.md`；正式训练前填完并提交 Git。

## 协议身份

| 项目 | 实际登记值 |
|---|---|
| 登记人、日期、Git commit | 待填写 |
| `configs/experiment.yaml` 哈希 | 待填写 |
| `configs/tasks.yaml` / `configs/models.yaml` 哈希 | 待填写 |
| 数据 fingerprint（GSM8K/SVAMP/MMLU-Pro/BBH/IFEval/HumanEval+） | 待填写 |
| 模型 revision（HF commit）与 tokenizer revision | 待填写 |
| 主终点 | B2−B1 GSM8K test 最终答案准确率差值及 95%CI |
| 种子/预算 | 训练 42/43/44；选样 2,048；Token 差 ≤1%；Bootstrap 10000/2026 |
| 协议 | completion 主比较；chat 另表 |
| 变更记录 | 无正式运行；变更需列日期、旧/新值、重跑范围 |

## 模型与端点

| 角色 | 模型/权重 ID | 端点/服务版本 | 选择理由 | 状态 |
|---|---|---|---|---|
| B0 8B | 待填 | 待填 | 主基线 | 未运行 |
| B1 8B ×3 | 待填 | 待填 | 等预算随机 | 未训练 |
| B2 8B ×3 | 待填 | 待填 | 错误加权 | 未训练 |
| B0/B1/B2 4B ×3 | 待填 | 待填 | 主要结论复核 | 未训练 |
| 辅助 Judge | 待填 | 待填 | 仅错误分析 | 未运行 |

商业 API 的价格、币种、最大调用/Token 预算：待填；未填时远程端点必须停止。

## 数据与选样记录

- GSM8K 官方 train/test 数量与 fingerprint：待填（期望 7473/1319）。
- dev 500 / pool 6973 的清单哈希：待填。
- 训练池诊断 run、诊断覆盖率、答对/答错数：待填。
- B1/B2 每种子样本数、Token、层分布、重叠、诊断覆盖：待填。
- SVAMP/MMLU-Pro/BBH/IFEval/HumanEval+ fingerprint：待填。
- 污染检查：精确/近重复/同源计数与阈值：待填。

## 训练记录

- 底座 revision、数据 hash、配置 hash、代码 commit：待填。
- rank/alpha/dropout/lr/global batch/len/epoch/seed：待填（协议固定值）。
- 小步检查日志、完整训练日志、checkpoint 路径、合并命令：待填。
- 权重登记 `weights.jsonl` 行：待填。

## 正式评测与结果索引

每模型/种子/任务一行；冻结前不填 test 结果。

| 模型 | 种子 | 任务 | 协议 | freeze | run | 完整性 | 状态 |
|---|---:|---|---|---|---|---|---|
| B0 | — | GSM8K | completion | — | — | — | 未执行 |
| B1 | 42/43/44 | GSM8K | completion | — | — | — | 未执行 |
| B2 | 42/43/44 | GSM8K | completion | — | — | — | 未执行 |
| B1/B2 | 42/43/44 | SVAMP | completion | — | — | — | 未执行 |
| 全部 | — | 辅助/RULER | completion/chat | — | — | — | 未执行 |

主比较结果、每种子差值、CI、辅助能力变化、成本、失败案例、局限：待填写。
