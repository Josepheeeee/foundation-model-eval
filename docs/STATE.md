# 项目状态与开发记忆

更新时间：2026-10-22（Asia/Shanghai）。工程基座、离线闭环和实验协议已落地；真实数据、训练和模型评测仍待执行。

## 已确定的决策

- 独立目录与 Git：`foundation-model-eval`，Python 3.12，独立包名 `foundation_eval` 与入口 `foundation-eval`。
- 主模型 Qwen3-8B-Base；Qwen3-4B-Base 复核 GSM8K/SVAMP 与主要稳健性；开放基座 + 自行训练 LoRA，不称预训练研究。
- 主协议 completion；SFT 模型 native chat 另表；不合并两种协议。
- 主终点 B2−B1 的 GSM8K test 准确率；SVAMP 只做冻结后的迁移；不合成总分。
- GSM8K train 7473 → 固定哈希 dev 500 + pool 6973；官方 test 1319 不参与诊断/选样。
- B1/B2 各 2,048；10 个长度层；B2 答错权重 3；Token 预算差 ≤1%；种子 42/43/44。
- 训练 BF16 LoRA：rank32/alpha64/dropout0.05/lr1e-4/global batch16/len4096/1 epoch。
- Judge 只做辅助错误分析；主评分是确定性答案、规则和功能测试。
- 上游源码在 `external/`，按 `references.lock.yaml` 固定 commit，不提交。

## 已实现与已验证

- 八个源码模块：`schema`、`data`、`agent`、`judge`、`selection`、`report`、`run`、`__init__`；`agent.py` 保留位置但职责是被测模型适配。
- 固定 dev 留出、样本级哈希、训练/迁移/辅助/诊断任务清单、数据指纹占位与下载脚本。
- 污染检查：精确 prompt hash、5-word shingle 近重复（Jaccard≥0.8）、同源 group 跨划分检查；`check`在数据齐备时写入`data/manifests/contamination.json`并拒绝发现项；明确“只检查可见数据”。
- 选样：长度分层、B1 随机、B2 层内错误加权、Efraimidis-Spirakis 无放回抽样、同层交换匹配 Token、B2 全池诊断门禁、重叠与预算记录。
- 评分：GSM8K/SVAMP 数值、MMLU-Pro 选项、BBH 任务 exact match、IFEval 规则（unsupported 单独计）、HumanEval+ 受限执行、RULER 数值。
- 提取核对脚本：读取固定 lm-eval 上游 GSM8K 任务的 `regex_pattern`，对每条本地 gold answer 与我们的提取逐样本比较；真实数据下载后执行。
- 代码执行：Docker 默认（无网络、512m、pids 64、cap-drop、非 root、只读挂载）；Windows local 模式默认拒绝，需显式 `FOUNDATION_ALLOW_LOCAL_CODE=1`。
- 报告：每任务原生指标、Wilson 区间、状态对账、成本、失败案例、协议/数据/端点；无合成分数。
- 统计：B1/B2 三种子配对、样本级 Bootstrap、每种子差值、协议/数据/种子完整性拒绝。
- 运行治理：manifest、原始响应、逐样本评分、协议签名、test 冻结、失败/中断保留、权重登记。
- LlamaFactory 配置：8B/4B、B1/B2、种子 42/43/44；rank/alpha/dropout/lr/batch/len/epoch 固定；`dataset_info.json` 生成。
- Windows 离线入口：setup/check/demo；Linux：基础/训练/推理安装、数据下载、vLLM 服务、四卡训练、merge、评测矩阵、结果校验。
- `docs/JD_RAW.md` 与 `jd_sources.json` 复制同一版本；本项目的 `JD_ANALYSIS.md` 只写基座侧能力映射。

## 验证记录

| 检查 | 结果 |
|---|---|
| `pytest -q` | **18 passed**（2026-10-22 本机，Python 3.12，离线） |
| `foundation-eval check --offline` | 通过；缺数据被标为 pending，不伪造完成 |
| `foundation-eval demo` | 8 条自制 GSM8K 风格样例 + 脚本响应；生成选样、评分、报告与图；明确 scripted/no benchmark |
| 固定 dev 切分 | 1200 条合成数据上 500/700，重复运行一致、无重叠 |
| 污染检查 | 精确、近重复、同源三类均可检出；测试覆盖 |
| B1/B2 | 120 条池、2 种子、20 条/组；Token gap≤1%、B2 错题率≥B1、可复现 |
| 代码评分 | local 模式通过/失败样例；Docker 模式为 H20 默认，尚未在 H20 实跑 |
| RULER | 4K 固定种子生成一致；长度/任务注册拒绝非法值 |
| LlamaFactory 配置 | rank32/lr1e-4/4 卡×1×4=16 全局 batch；8B/4B、三种子 |
| 报告/比较 | 状态计数、unsupported 不入分母、配对 CI、协议不匹配拒绝 |

## 仓库

公开仓库：[Josepheeeee/foundation-model-eval](https://github.com/Josepheeeee/foundation-model-eval)（`main`）。提交前已做凭证样式扫描；runs、数据、权重和 `.env` 不进入 Git。

## 当前限制

- 没有真实 GSM8K/SVAMP/MMLU-Pro/BBH/IFEval/HumanEval+ 数据下载；`data/manifests` 只有计划，真实 fingerprint 待 H20 写入。
- 没有 B0 训练池诊断、没有 B1/B2 真实选样、没有 LoRA 训练、没有真实模型评测。
- 没有 Docker 实跑、没有 4 卡环境验证；本地脚本未在 H20 执行。
- 没有人工失败案例复核，没有真实主终点差值/CI。
- 数据污染检查不能证明预训练语料无污染；RULER 只是本地诊断，不是官方榜单。
- `prepare_datasets.py` 依赖 HF 数据集字段；真实下载时若上游字段变化，必须修正并记录，不可静默丢弃样本。

## 下一步

1. 在 H20 执行 `scripts/linux/install.sh`、`install_train.sh`、`install_infer.sh`，分别验证 base/train/infer 环境。
2. 运行 `download_data.sh`，核对 GSM8K 7473/500/6973/1319 与各任务 fingerprint；先做少量人工样例检查。
3. 起 B0 vLLM，采集 6973 训练池诊断；检查 extract_error/unsupported/执行失败比例。
4. 运行 B1/B2 选样，核对 2048、层分布、Token≤1%、诊断覆盖、重叠数；导出训练数据/配置。
5. 先做 10-step 小训练，确认模板、loss、显存、保存；再跑 8B/4B × B1/B2 × 3 种子。
6. 冻结 GSM8K/SVAMP 等 test 协议，跑 8B 矩阵；4B 复核；运行 `validate_results.sh`。
7. 生成主结果、迁移、辅助能力、成本、失败案例和局限报告；真实结果出来前保持“待执行”。
