# Linux 脚本

H20 上训练与推理隔离：`.venv`（基础/CPU）、`.venv-train`、`.venv-infer`。

| 脚本 | 用途 |
|---|---|
| `install.sh` | 基础环境与离线检查 |
| `install_train.sh` | `.venv-train`，LlamaFactory + 训练依赖 |
| `install_infer.sh` | `.venv-infer`，vLLM |
| `download_data.sh` | 下载公开任务、生成 fingerprint、GSM8K 500/6973 切分、RULER |
| `serve_vllm.sh` | 起 OpenAI 兼容服务；`MODEL`、`ADAPTER`、`TP` 可配 |
| `train_lora.sh` | 四卡 LoRA；检查 4 GPU，日志落 `logs/` |
| `merge_lora.sh` | 合并 adapter 供服务 |
| `run_eval_matrix.sh` | 冻结协议并跑 GSM8K/SVAMP/辅助/RULER |
| `validate_results.sh` | check + 全部 run 审计/矩阵/主比较 |

示例：

```bash
bash scripts/linux/install.sh
bash scripts/linux/install_train.sh
bash scripts/linux/install_infer.sh
bash scripts/linux/download_data.sh
MODEL=Qwen/Qwen3-8B-Base scripts/linux/serve_vllm.sh 8000
MODEL_LABEL=B0 TRAINING_SEED=42 scripts/linux/run_eval_matrix.sh
bash scripts/linux/validate_results.sh
```

脚本不内置密钥/价格/“已完成”声明；缺少数据冻结、4 卡或端点配置时明确停止。
