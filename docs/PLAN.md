# 批准计划：基座能力测量与评测驱动的 LightSFT

归档日期：2026-10-22。目标是把公开基座评测、训练集错误诊断、等预算数据选择、LoRA 训练和独立验证接成可复现工程，用于实习/校招展示。

## 1. 项目边界

- 保留 `retail-agent-eval` 作为 Agent 项目；在同级新建 `foundation-model-eval`，独立 Git、依赖、数据清单和产物。
- 复制白名单工程骨架，不复制旧 Git、虚拟环境、密钥、运行结果或“已验证”声明。
- 使用 Qwen3-8B-Base 为主、Qwen3-4B-Base 复核；只做 LoRA SFT，不做预训练。
- 不建 Web 平台，不做在线服务，不做多模态专项。

## 2. 评测任务

| 任务 | 角色 | 指标 |
|---|---|---|
| GSM8K | 主能力 | 最终答案 exact match |
| SVAMP | 迁移 | 最终答案 exact match；冻结后 |
| MMLU-Pro | 辅助 | 选项 exact match |
| BBH | 辅助 | 每任务 exact match，宏平均 |
| IFEval | 辅助 | 指令级 strict accuracy；unsupported 不算 pass |
| HumanEval+ | 辅助 | 受限容器功能正确性 pass@1 |
| RULER | 诊断 | 4K/8K/16K 固定种子，数值/字符串 |

所有任务分别报告原生指标；不合成总分。completion 为主协议，SFT 模型 chat 另表。

## 3. 数据与选样

- GSM8K 官方 train 7473：固定 SHA-256 留出 500 dev，余 6973 训练池；官方 test 1319 不进入诊断/选样。
- B0 在训练池诊断；B1 按 10 个长度层随机选 2,048；B2 同层内答错权重 3、其余 1。
- 两组样本数、长度分布、Token 预算差 ≤1%；同种子、同训练配置。
- 污染：精确/近重复/同源分组，覆盖 train/dev/test/few-shot；明确只能检查可见数据。

## 4. 训练

- BF16 LoRA：rank32/alpha64/dropout0.05/lr1e-4/global batch16/len4096/1 epoch。
- 4 卡：per-device 1 × grad-accum 4；种子 42/43/44。
- 8B 全套；4B 复核 GSM8K、SVAMP 和主要稳健性。
- 保存数据/配置/日志/checkpoint/合并/权重 hash。

## 5. 统计与交付

- 主终点 B2−B1 GSM8K test 准确率；三种子配对，样本级 Bootstrap 10000，95%CI。
- SVAMP 迁移；辅助任务检查退化；RULER 分长度。
- 交付：两项目 README、JD 原文/来源清单/能力映射、离线检查、Linux 脚本、结果模板。
- 真实结果前保持待执行；不预填收益。

## 6. 验收

1. 离线测试全绿；数据隔离/标签/选样/评分/报告/冻结有测试。
2. 数学提取与上游逐样本核对；IFEval 规则、代码执行、RULER 生成可复现。
3. 拒绝跨划分泄漏、不匹配协议、未冻结 test、不完整比较。
4. 先小步训练，再完整矩阵；保存配置/日志/checkpoint/选择依据。
5. 分别生成主结果、迁移、辅助、成本、失败案例、局限；负结果保留。
