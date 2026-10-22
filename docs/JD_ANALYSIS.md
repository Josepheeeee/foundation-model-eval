# 基座侧能力映射：70 条 JD 证据与本项目落点

复核日期：2026-10-22。原文与来源清单与 `retail-agent-eval` 保持同一版本：[JD_RAW.md](JD_RAW.md)、[jd_sources.json](../data/manifests/jd_sources.json)、[jd_review.json](../data/manifests/jd_review.json)。两份 JSON 由零售项目的 `scripts/build_jd_artifacts.py` 从同一冻结研究产物生成后按字节复制，避免两边口径漂移。本文件只写**基座能力测量与训练数据选择**这一侧；Agent 轨迹侧映射见另一项目的 JD_ANALYSIS。

## 1. 口径与计数

70 条中核心评测 51 条、相邻 18 条、待核验 1 条；实习 28、校招 14、社招 28。能力统计（全部/实习校招）来自 [jd_capability_stats.json](../data/manifests/jd_capability_stats.json)：Benchmark 56/33、指标 39/17、LLM Judge 27/15、Agent 轨迹 49/27、校准统计 32/17、污染 8/4、代码 45/28、数据质量/选择 29/17、训练-评测闭环 42/25、长上下文 7/5、多模态 37/21、评测平台 58/35、错误诊断 27/11。

基座项目优先承接的是“Benchmark/指标/预训练与后训练评测/污染/数据选择/训练闭环/长上下文/代码”，不承接多模态、语音、安全专项和生产平台。

## 2. 岗位要求 → 原文证据 → 项目实现 → 验证 → 缺口

| 岗位要求 | 代表性原文证据 | 基座项目实现 | 验证结果 | 剩余缺口 |
|---|---|---|---|---|
| 预训练/中训练阶段的多维能力评估 | JD-12 Dots-预训练评估策略；JD-27 预训练评测；JD-29 Pre/Mid-training 评测；JD-55 基模评测 | GSM8K 主能力 + MMLU-Pro/BBH/IFEval/HumanEval+/RULER 分任务原生指标；固定 completion 协议 | 任务清单、提取/评分、状态对账已离线测试；真实数据未下载 | 预训练内部指标/checkpoint 扫描未实现；只做后训练 LoRA 对照 |
| 固定 Benchmark、few-shot、答案提取与可比报告 | JD-13 多维度评测体系；JD-19 基座测评；JD-39 通用 Benchmark；JD-62 LLM Evaluation | 任务级 lock：revision/fingerprint、few-shot、提取、生成参数、协议签名；分任务报告，不合成总分 | 冻结/签名/数据哈希测试；demo 全链路 | 真实 fingerprint 与上游逐样本核对待 H20 |
| 评测集质量控制、污染与泄漏检查 | JD-13 抗污染；JD-28 Agentic 评测数据；JD-38 数据评测 | 精确 prompt hash、5-word shingle 近重复、同源分组；覆盖 train/dev/test/few-shot | 三类污染可检出；明确“只检查可见数据” | 不能证明预训练语料无污染 |
| 基座错误诊断与评测驱动数据闭环 | JD-01 缺陷归因；JD-16 自动化评测演进；JD-45 自动化评测+benchmark；JD-51 动态评测与诊断 | 训练池逐样本诊断 → B1/B2 等预算选样；记录答错率/层分布/Token | 选样/预算匹配/诊断门禁测试 | 真实 6973 诊断与三次种子训练待执行 |
| 等预算数据选择与训练前后对照 | JD-14 后训练泛化；JD-29 评测数据配比；JD-38 数据评测；JD-58 评测体系；JD-67 MiMo 后训练 | B0/B1/B2；GSM8K dev500；10 长度层；B2 答错权重 3；Token≤1%；LoRA 三种子 | 选样/配置/训练导出离线测试 | 无真实 B2−B1 差值/CI |
| 统计校准、置信区间与种子方差 | JD-41 评测专家；JD-46 Benchmark 体系；JD-51 动态评测；JD-54 评测理论；JD-62 LLM Evaluation | 样本级配对 Bootstrap、每种子差值、Wilson 区间；明确 3 种子不是大样本 | 配对/完整性/协议不匹配拒绝 | 真实重复评测与人工复核未执行 |
| 代码能力功能正确性 | JD-13 代码维度；JD-20 Code Agent；JD-34 AI Coding | HumanEval+ pass@1；Docker 无网络/限额/非 root；执行失败不算 pass | 解析/执行契约与 local 通过/失败样例 | Docker 未在 H20 实跑；Code Agent 多文件任务不是本项目 |
| 长上下文诊断 | JD-13 长上下文；JD-27 长文本；JD-29 长文本；JD-38 长文本 | RULER 4K/8K/16K × niah_single_1/niah_multikey_1/vt，固定种子 2026 | 生成一致性与长度/任务校验 | 本地诊断不是官方 RULER 榜单；需目标 tokenizer 与真实模型 |
| 指令遵循/规则验证 | JD-13 指令遵循；JD-26 可扩展评测；JD-30 应用评测 | IFEval 规则验证；unsupported 单独计，不默认 pass | 规则/unsupported 测试 | 只覆盖已实现指令类型；未覆盖全部 25 类 |
| 训练与评测协议分离、可复现 | JD-29 训练-评测一致性；JD-45 闭环；JD-58 体系 | 训练/推理依赖分文件；配置/数据/代码/权重 hash；运行清单；test 冻结 | 离线检查/配置导出/冻结测试 | H20 四卡与 vLLM 未验证 |
| 多模态/语音/视觉评测 | JD-05、JD-09、JD-36、JD-42、JD-50、JD-53、JD-57 | 明确非目标；只保留为岗位背景 | 未实现 | 不用文本项目冒充多模态能力 |
| 安全/对抗评测 | JD-24、JD-32、JD-33、JD-35、JD-37、JD-40、JD-60 | 仅保留为背景；不把安全评测混入主终点 | 未实现 | 非目标；需单独安全 benchmark |
| 评测平台/调度/可视化产品 | JD-02、JD-03、JD-13、JD-28、JD-39、JD-46 | 只做薄 Harness：清单、采集、报告、矩阵；不建 Web | 离线闭环 | 不声称平台化/调度产品能力 |

## 3. 与零售项目的分工

| 维度 | retail-agent-eval | foundation-model-eval |
|---|---|---|
| 被测对象 | DeepAgents + τ 零售环境 | Qwen3 Base + LoRA SFT |
| 核心测量 | 业务轨迹、官方 reward、分层 Judge、人工校准 | 数学/知识/推理/指令/代码/长上下文原生指标 |
| 反馈/改进 | 分层失败反馈 → GEPA 提示词 | 基座错误诊断 → B1/B2 数据选择 |
| 主要终点 | A3−A2 官方任务奖励 | B2−B1 GSM8K 准确率 |
| 不声称 | 六层最优、已完成因果根因 | 预训练研究、合成分数、普遍有效 |

## 4. 结论

- 70 条 JD 中与基座项目直接相关的是预训练/后训练评测、Benchmark/指标、污染、代码/长上下文、数据选择与训练闭环；本项目逐项有薄实现或明确非目标。
- 实习/校招样本占比高（42/70），能力统计优先使用这部分；社招用于补充方法边界。
- 所有“验证”只指离线工程检查；真实模型、训练、人工复核结果必须在 H20 运行后回填，当前不预填收益。
