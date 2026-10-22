# 下一步：从离线工程到 H20 真实实验

本文只写操作顺序和验收点。模板/入口/fixture 通过不等于真实实验完成。

## 0. 本机（Windows）先确认

```powershell
scripts\windows\setup.ps1
scripts\windows\check.ps1
```

验收：`pytest` 全绿；`check --offline` 无 issue；`demo` 生成 `runs/...-demo-.../report/report.md` 并明确 scripted/no benchmark。

## 1. H20 环境

```bash
git clone <repo> foundation-model-eval
cd foundation-model-eval
bash scripts/linux/install.sh
bash scripts/linux/install_train.sh
bash scripts/linux/install_infer.sh
```

三个环境分开：基础/CPU、训练、推理。训练与推理不要共用同一个 venv。验收：`torch.cuda.device_count()==4`；`vllm` 可导入；`pip check` 无冲突。

## 2. 数据

```bash
bash scripts/linux/download_data.sh
```

先做提取一致性核对（需要 `external/lm-evaluation-harness` 固定源码）：

```bash
python scripts/fetch_references.py lm-evaluation-harness
python scripts/check_upstream_gsm8k.py --answers data/raw/gsm8k_test.jsonl
```

验收：
- `data/manifests/gsm8k.json`：train 7473、test 1319；
- `data/processed/gsm8k/`：train 6973、dev 500、test 1319，无重叠；
- 各任务文件行数与 `data/manifests/datasets.json` 对照；fingerprint 写入；
- 抽查 5 条 GSM8K、2 条 IFEval、2 条 HumanEval+ 的 prompt/answer/test_code。

如果上游字段变化导致脚本失败，修正后记录，不静默丢样本。

## 3. B0 诊断

```bash
MODEL=Qwen/Qwen3-8B-Base scripts/linux/serve_vllm.sh 8000
# 另开终端
export TARGET_MODEL=target-model TARGET_BASE_URL=http://127.0.0.1:8000/v1 TARGET_API_KEY=local
python -m foundation_eval.run collect --task gsm8k --split train \
  --model-label B0 --protocol completion --diagnose-only \
  --samples data/processed/gsm8k/train.jsonl
```

验收：run manifest `complete`；actual=expected=6973；诊断覆盖率 100%；记录 extract_error 数；随机抽查 10 条响应/评分。若服务端不稳定，保留中断 run，修好后新 run，不覆盖。

## 4. B1/B2 选样与训练数据

```bash
python -m foundation_eval.run select --pool data/processed/gsm8k/train.jsonl \
  --diagnosis runs/<B0>/diagnosis.jsonl --output data/generated/selection
python -m foundation_eval.run train-export --selection data/generated/selection \
  --output data/generated/training
```

验收：每种子 B1/B2 各 2048；Token gap≤1%；层分布一致；B2 错题率高于 B1；诊断覆盖 100%；`training_matrix.json` 有 12 个任务（2 组 × 3 种子 × 8B/4B）。

## 5. 小步检查与完整训练

先复制一个配置，改 `max_steps: 10`（或命令行覆盖），确认模板/loss/显存/保存路径；再跑完整：

```bash
bash scripts/linux/train_lora.sh configs/generated/b1_seed42_8b/seed42.yaml
```

验收：4 卡均使用；global batch=16；日志无 NaN；checkpoint 落盘；配置/数据/种子/hash 写入训练记录。然后 `merge_lora.sh` 合并，`weight-register` 登记 adapter hash。

## 6. 冻结与评测

```bash
MODEL=outputs/... scripts/linux/serve_vllm.sh 8001
export TARGET_MODEL=target-model TARGET_BASE_URL=http://127.0.0.1:8001/v1 TARGET_API_KEY=local
MODEL_LABEL=B1 TRAINING_SEED=42 scripts/linux/run_eval_matrix.sh
```

验收：每个任务 freeze 文件存在；test 全量无 `--limit`；run `complete`；scores 的 ok 覆盖全部样本（HumanEval+ 执行失败按状态记录，不补 pass）。8B 全部矩阵跑完后跑 4B GSM8K/SVAMP。

## 7. 结果校验与报告

```bash
bash scripts/linux/validate_results.sh
```

验收：`runs/validation/matrix.md` 覆盖全部 run；B1/B2 三种子齐全时生成 `gsm8k_b2_minus_b1.json`；否则明确 pending。`compare` 只接受完整、同协议、同 test、同签名的 run。

最终报告必须包含：主结果、每种子差值、CI、SVAMP 迁移、辅助任务变化、成本、失败案例、污染局限、与预注册的差异。负结果和 CI 跨 0 原样报告。

## 8. 发布前

- 真实结果回填 `PREREGISTRATION.md`、`STATE.md`、`REPORT`（待建）；
- 检查无密钥、无原始私有数据、无权重/checkpoint；
- 复核数据/上游许可与引用；
- 保留失败 run，不用成功重跑覆盖异常；
- 两个项目分别提交/推送，不共享运行结果。
