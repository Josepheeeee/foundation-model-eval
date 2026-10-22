# 项目约定

本项目研究“基座模型能力测量与评测驱动的 LightSFT 数据选择”。开始工作先读 `README.md`、`docs/EXPERIMENTS.md`、`docs/STATE.md`；结束工作更新 `STATE.md` 的已验证事实、限制和下一步。

`docs/PLAN.md` 保存批准计划，`docs/PLAN_CHECKLIST.md` 是逐项验收，`docs/NEXT_STEPS.md`指导 H20 操作。模板、入口、fixture、脚本模型检查不能替代真实训练/评测。

## 实验纪律

- 主模型 Qwen3-8B-Base，Qwen3-4B-Base 复核；开放基座 + 自行训练 LoRA，不声称完成预训练。
- 每个任务锁定数据集 revision/fingerprint、few-shot、答案提取、生成参数；completion 是主比较协议，chat 另表。
- GSM8K 官方 train 固定哈希留出 500 dev，余 6973 训练池；官方 test 不进入诊断/选样/调参。
- B1/B2 各 2,048 条，长度分层一致，Token 预算差 ≤1%，同种子/同训练配置；B2 权重只在层内生效。
- 主终点 B2−B1 的 GSM8K test 最终答案准确率；SVAMP 只做冻结后的迁移；不合成总分。
- 确定性答案、功能测试、规则验证是主评分；LLM Judge 只做辅助错误分析，不进入指标。
- IFEval 未支持指令记为 unsupported，不能算 pass；代码在无网络受限容器执行，执行失败不能算 pass。
- 原始输出不可覆盖；失败 run 保留；权重/数据/配置/代码哈希进入运行清单。
- 密钥只放 `.env`；不提交原始数据、权重、checkpoint、运行结果或含密钥配置。
- 真实结果前保持“待执行”，不预填收益；负结果和不确定性必须报告。

## 工程要求

Python 3.12；`pytest` 离线通过；`foundation-eval check --offline` 与 `demo` 可运行。
新增行为需检查数据隔离、评分状态、选样预算和可复现性。入口为 `foundation-eval`。
上游源码在 `external/`，按 `references.lock.yaml` 固定 commit，不提交。
训练/推理依赖分文件：`requirements-train.txt`、`requirements-infer.txt`。
