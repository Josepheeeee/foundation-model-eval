# 完整实验设计与预注册 v1

登记日期：2026-10-22。状态：**协议已实现，尚未开始真实数据下载、训练与评测**。本文件与 `configs/experiment.yaml`、`configs/tasks.yaml`、`configs/models.yaml` 定义协议；实际模型 revision、数据指纹和冻结信息在 `PREREGISTRATION.md` 与运行清单中填写。预期方向是研究假设，不是已取得结果。

## 1. 研究问题与业务决策

| 问题 | 零假设 / 检验方向 | 证据与决策 |
|---|---|---|
| RQ1 能力测量是否可靠 | 不同协议/模板会制造虚假“能力变化” | 固定 few-shot、提取、生成参数；逐样本记录；上游样本核对 |
| RQ2 错误诊断选样是否优于随机 | B2−B1 的 GSM8K test 准确率差值为 0；可能正、负或退化 | 三种子配对差值与 95%CI；SVAMP 迁移；失败案例 |
| RQ3 收益是否以其他能力为代价 | 辅助任务无变化或退化 | MMLU-Pro/BBH/IFEval/HumanEval+/RULER 分任务变化 |

主要终点只有 RQ2 的 B2−B1 GSM8K 最终答案准确率差值。不根据 test 结果改主终点，不挑最好种子，不用辅助任务反向选择主结论。

## 2. 数据、划分与污染控制

| 任务 | 角色 | 数据 | 用途 | 禁止 |
|---|---|---|---|---|
| GSM8K | 主能力 | 官方 train 7473 / test 1319 | 训练池诊断、B1/B2 选择、最终 test | test 不进入诊断/选择/调参 |
| SVAMP | 迁移 | 官方 test | 冻结后的跨数据集检验 | 不进入诊断/选样 |
| MMLU-Pro | 辅助 | 官方 test | 知识/推理变化 | 不与 GSM8K 合成总分 |
| BBH | 辅助 | 23 子任务 test | 推理变化；任务内 exact match，再宏平均 | 不按 test 挑任务 |
| IFEval | 辅助 | 官方 541 | 指令遵循规则验证 | unsupported 不算 pass |
| HumanEval+ | 辅助 | EvalPlus test 164 | 功能正确性 | 执行失败不算 pass |
| RULER | 诊断 | 本地固定长度生成 | 4K/8K/16K 长上下文诊断 | 不称官方 RULER 榜单 |

GSM8K dev 固定：按 `SHA256("gsm8k-dev-v1:" + sample_id)` 升序取前 500；其余 6973 为训练池。`prepare` 写出 `train/dev/test` 三个文件和样本级哈希；B1/B2、诊断、阈值都只用 train/dev，test 只在协议冻结后采集。

污染检查覆盖 GSM8K train/dev/test、few-shot 示例与可选的 SVAMP test：
- 精确：规范化 prompt 的 SHA-256 相同；
- 近重复：5-word shingle 的 Jaccard ≥0.8；
- 同源：`group_id` 跨划分重叠。

检查结果写入运行清单。**局限**：只能检查本项目可见的数据，不能证明原始预训练语料没有见过 benchmark；不把“未发现重复”写成“无污染”。

## 3. 固定测量协议

- 主协议：base-model **completion**。每个任务使用同一系统提示、同一固定 few-shot 示例、同一答案提取和生成参数。
- SFT 模型的 native chat 协议单独报告；chat 与 completion 不合并、不互相替代。
- 生成：temperature 0、top_p 1、max_new_tokens 见任务配置、seed 2026；服务端若不保证确定性，记录限制。
- 每个样本保存 prompt hash、response 全文、usage、耗时、finish_reason、score、status。
- 评分：GSM8K/SVAMP 最终数值；MMLU-Pro 选项；BBH 任务 exact match；IFEval 规则；HumanEval+ 受限执行；RULER 数值/字符串。
- LLM Judge 只用于辅助错误分析，输出单独保存，不进入任何主指标。

状态语义：`ok` 才进入分母；`extract_error`/`unsupported`/`execution_error`/`judge_error`/`error` 分别计数，不能当 pass，也不能悄悄丢掉。

## 4. B0/B1/B2 与选样

| 组别 | 设置 |
|---|---|
| B0 | 原始 Qwen3-8B-Base，不训练 |
| B1 | 从 6973 训练池按长度分层随机选 2,048 条 |
| B2 | 同一池、同一长度层，基座答错样本权重 3、其余 1，选 2,048 条 |

- 长度层：对训练池的 prompt+answer token 数做 10 分位分层；B1/B2 使用同一层映射。
- 权重只在层内生效；B2 不是全局过采样，也不改变层配额。
- 两组的长度分布和有效训练 Token 预算匹配，差异 ≤1%；通过同层交换实现，记录交换次数和最终 Token。
- 每条样本只能出现一次；B1/B2 可重叠，重叠数记录。主比较在 test 上配对，训练集重叠不构成评测泄漏。
- B2 要求训练池每个样本都有基座诊断；缺失即报错，不把缺失默认为答对。
- 诊断采集用 B0、completion、train split，输出 `diagnosis.jsonl`；诊断不读取 test。

选样清单记录：种子、层配额、样本 ID、Token、答错数/率、重叠数、诊断覆盖、输入哈希。选样完成后不再改数据；若要改规则，必须先登记新协议并重跑 B1/B2。

## 5. SFT 训练

默认 Qwen3-8B-Base；Qwen3-4B-Base 复核。BF16 LoRA：

| 参数 | 值 |
|---|---|
| rank / alpha / dropout | 32 / 64 / 0.05 |
| learning rate | 1e-4 |
| global batch | 16（4 GPU × 1 × grad-accum 4） |
| cutoff length | 4096 |
| epochs | 1 |
| seeds | 42 / 43 / 44 |
| template | qwen3（非思考模式；训练脚本固定） |

B1/B2 使用完全相同训练设置，仅训练数据不同。保存配置、日志、checkpoint、合并/接回命令、数据与代码哈希。4B 只复跑 GSM8K、SVAMP 和主要稳健性结果。

先做数据加载与小步训练检查（例如 10 step），确认 loss、显存、模板与保存路径，再跑完整矩阵；不能用一次失败训练的结果补成成功。

## 6. 评测矩阵与终点

| 模型 | GSM8K | SVAMP | MMLU-Pro | BBH | IFEval | HumanEval+ | RULER |
|---|---|---|---|---|---|---|---|
| B0 8B | test | test | test | test | test | test | 4K/8K/16K |
| B1 8B ×3 | test | test | test | test | test | test | 4K/8K/16K |
| B2 8B ×3 | test | test | test | test | test | test | 4K/8K/16K |
| B0/B1/B2 4B ×3 | test | test | — | — | — | — | 主要稳健性 |

- 主终点：B2−B1 的 GSM8K test 最终答案准确率，按训练种子配对，样本级配对 Bootstrap 10000 次，随机种子 2026；报告每种子差值和 95%CI。
- 迁移：SVAMP 同样配对报告，但只作支持证据。
- 辅助：各任务原生指标与状态计数；不做合成总分，不按辅助结果改主结论。
- 稳健性：长度分层（短/中/长）准确率、答案格式错误率、extract_error 率、few-shot 顺序固定检查；RULER 按长度分列。
- 成本：每任务调用数、已知 Token、未知 usage 数、平均耗时；不把未知 usage 当 0。

## 7. 统计与失败处理

- 任务/样本配对；同一 test 样本在 B1/B2 中必须都 status=ok 才进入配对差；缺失导致比较拒绝，不做“可用样本”替换。
- 三种子是训练重复，不是三个独立数据集；报告种子间差异，不把 3 个种子当大样本。
- CI 跨 0 不支持稳定改进；CI 不跨 0 也需结合任务覆盖、数据规模和多重比较限制。
- 运行异常：期望/实际/ok 记录数逐项对账；传输失败、解析失败、执行失败分别保留。中断 run 标记 `interrupted`，修复后新 run，不覆盖原始输出。
- 正式 test 需要冻结协议；协议签名包含任务配置、生成参数、数据哈希、prompt/few-shot 哈希和评分代码哈希。代码/数据/提示改变后旧冻结失效。

## 8. 预算与真实模型登记

商业 API 的模型 ID、端点、价格、预算在运行前写入 `.env` 与 `configs/experiment.yaml` 的 `budget`。远程端点未设置 `max_calls` 时 CLI 停止。实际调用写入 run manifest；价格只用于成本报告，不写进公开密钥。

| 角色 | 模型/权重 | 用途 | 状态 |
|---|---|---|---|
| 被测基座 | Qwen/Qwen3-8B-Base | B0 与训练底座 | 未运行 |
| 复核基座 | Qwen/Qwen3-4B-Base | 4B 复核 | 未运行 |
| B1/B2 权重 | 自行训练 LoRA | 训练前后对照 | 未训练 |
| 辅助 Judge | 未配置 | 错误分析，非主评分 | 未运行 |

## 9. 产物与验收

| 产物 | 生成方式 | 验收 |
|---|---|---|
| 数据指纹/划分 | prepare/download | 7473 train、500 dev、6973 pool、1319 test 无重叠 |
| 污染报告 | check/run manifest | 精确/近重复/同源；记录阈值与局限 |
| B1/B2 选样 | select | 2048×2、层一致、Token 差 ≤1%、诊断完整 |
| 训练 | train-export/train_lora | 配置/数据/种子/hash/日志/checkpoint 齐备 |
| 评测 run | collect | 样本级响应、状态对账、冻结签名 |
| 主比较 | compare | 3 种子、配对、CI、每种子差值 |
| 报告 | report/matrix | 原生指标、失败案例、成本、限制；无预填收益 |

实施顺序：离线检查 → 数据下载与指纹 → B0 诊断 → B1/B2 选样 → 小步训练 → 三次种子训练 → 冻结 test → 8B 矩阵 → 4B 复核 → 统计与报告。

## 10. 明确不做

- 不训练/继续预训练基座，不把 LoRA SFT 说成预训练研究。
- 不把 MMLU-Pro/BBH/IFEval/HumanEval+ 合成一个“综合能力分”。
- 不用 LLM Judge 替代确定性评分，不用 RULER 本地诊断冒充官方榜单。
- 不声称已证明数据选择普遍有效；结论只针对本模型、数据 revision、预算和协议。
