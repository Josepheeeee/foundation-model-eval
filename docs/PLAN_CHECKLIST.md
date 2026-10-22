# 计划逐项验收表

核对日期：2026-10-22。依据批准计划、当前源码、离线测试与本机 demo。

**结论：工程实现与离线协议已交付；真实数据、训练、模型评测和结果分析尚未完成。**不能勾选“真实实验完成”。

状态定义：**已验证**=代码和离线检查有证据；**待执行**=入口/规程存在但无真实产物；**非目标**=明确不做。

## 1. 工程基座

| 要求 | 状态 | 证据 |
|---|---|---|
| 独立目录/Git/依赖 | 已实现；Git 待初始化/推送 | 根目录、pyproject、.gitignore |
| 独立包名与入口 | 已验证 | `foundation_eval`、`foundation-eval` |
| 八模块职责 | 已验证 | schema/data/agent/judge/selection/report/run/__init__ |
| agent.py 改为被测模型适配 | 已验证 | 无工具/记忆；completion/chat、重试、usage；README 说明 |
| 配置/提示/Rubric 分离 | 已验证 | configs、prompts、rubrics |
| 运行清单/数据/代码哈希 | 已验证 | manifest、data_hashes、source_hashes、protocol_signature |
| 原始输出不可覆盖 | 已验证 | run raw/records、失败保留、新 run |
| 开发记忆 | 已完成 | AGENTS.md、STATE.md；不传给被测模型 |

## 2. 任务与评分

| 要求 | 状态 | 证据/缺口 |
|---|---|---|
| GSM8K 主任务 | 接口已验证；真实数据待下载 | 提取/评分测试；无真实准确率 |
| 数学提取与上游逐样本核对 | 脚本已交付；真实数据待执行 | `scripts/check_upstream_gsm8k.py` 读固定 lm-eval 任务 regex |
| SVAMP 迁移 | 接口+隔离已验证 | 不进入诊断/选样 |
| MMLU-Pro | 接口+选项评分已验证 | 真实数据待下载 |
| BBH | 接口+exact/choice 已验证 | 23 子任务真实数据待下载 |
| IFEval | 规则+unsupported 已验证 | 只覆盖已实现指令类型；未支持类型不判 pass |
| HumanEval+ | Docker/local 契约+评分已验证 | Docker 未在 H20 实跑 |
| RULER | 4K/8K/16K 生成/种子已验证 | 本地诊断，非官方榜单 |
| 不合成总分 | 已验证 | report/rubrics 明确禁止 |
| LLM Judge 仅辅助 | 已验证 | 不进入 metrics；单独记录 |

## 3. 数据与选样

| 要求 | 状态 | 证据/缺口 |
|---|---|---|
| 7473/500/6973/1319 | 逻辑已验证；真实数据待执行 | 1200 合成数据测试 500/700；H20 待跑 |
| 固定哈希 dev | 已验证 | salt、排序、重复一致 |
| 污染检查 | 已验证 | 精确/近重复/同源；明确局限 |
| B1 长度分层随机 | 已验证 | 10 层、2048、可复现 |
| B2 层内错误权重 3 | 已验证 | 全池诊断门禁、错题率记录 |
| Token 预算 ≤1% | 已验证 | 同层交换、gap 记录、失败拒绝 |
| 诊断/选择/训练记录 | 已验证 | selection.json/training_matrix/weights |

## 4. 训练

| 要求 | 状态 | 证据/缺口 |
|---|---|---|
| LoRA 参数固定 | 已验证配置 | rank32/alpha64/dropout0.05/lr1e-4/len4096/1 epoch |
| 4 卡全局 batch16 | 配置已验证 | 1×4×4=16；脚本检查 4 GPU |
| 三种子 | 已验证配置 | 42/43/44 |
| 训练/推理依赖隔离 | 脚本已交付 | 未在 H20 安装验证 |
| 小步→完整 | 规程已写 | 未执行 |
| checkpoint/权重登记 | 入口已验证 | 无真实权重 |

## 5. 统计与报告

| 要求 | 状态 | 证据/缺口 |
|---|---|---|
| 主终点 B2−B1 | 接口已验证 | 无真实差值 |
| 三种子配对/Bootstrap | 已验证 | 每种子差值、CI、完整性拒绝 |
| SVAMP 迁移 | 接口已验证 | 无真实结果 |
| 辅助能力/退化 | 接口已验证 | 无真实结果 |
| 成本/未知 usage | 已验证 | 不补零 |
| 失败案例/局限 | 模板已实现 | 无人工复核 |
| 冻结 test | 已验证 | 协议签名/数据哈希；改动拒绝 |

## 6. 交付与 GitHub

| 要求 | 状态 | 证据 |
|---|---|---|
| README 六段阅读顺序 | 已完成 | 目的/架构/快速开始/实验/产物/状态 |
| Windows 离线入口 | 已验证 | setup/check/demo |
| Linux 安装/服务/训练/评测/校验 | 已交付 | scripts/linux；未在 H20 实跑 |
| 两项目 JD 原文/来源 | 已复制同一版本 | docs/JD_RAW.md、data/manifests/jd_sources.json |
| 基座侧能力映射 | 已完成 | docs/JD_ANALYSIS.md |
| 真实结果模板 | 已完成 | PREREGISTRATION/EXPERIMENTS；不预填 |
| 上传 GitHub | 待执行 | 独立仓库 |

## 阶段验收

- [x] 阶段1：工程、契约、任务、评分、选样、报告、脚本。
- [x] 阶段2a：离线测试 + demo。
- [ ] 阶段2b：H20 数据下载与 fingerprint。
- [ ] 阶段3：B0 诊断 + B1/B2 选样 + 小步训练。
- [ ] 阶段4：完整 LoRA（8B/4B × 3 种子）。
- [ ] 阶段5：冻结 test、8B/4B 矩阵、统计与报告。
- [ ] 阶段6：独立复现、公开结果、两项目推送。
