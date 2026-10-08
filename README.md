# COLA-Experiment

这是论文 *Stance Detection with Collaborative Role-Infused LLM-Based Agents* 的简化复现实验环境。代码在本目录独立运行，保留论文三阶段流程：三种文本分析、三种立场辩护、最终裁判。实验使用 OpenAI Chat Completions 兼容接口，因此可以切换 Qwen、OpenAI 或其他提供商。

## 已准备内容

- `data/smoke.csv`：5 条离线 smoke 样本。
- `data/dev15.csv`：15 条从训练镜像抽取的开发样本，仅用于开发验证；不能当作 SEM16 官方测试集。
- `cola/`：数据读取、指标、API 客户端和 COLA 流程。
- `run.py`：唯一实验入口。
- `results/`：运行后保存逐样本记录和汇总；每行立即落盘。
- `docs/experiment-protocol.md`：复现范围、协议差异和数据边界。
- `docs/dataset-sources.md`：SEM16、VAST、P-Stance 下载来源与许可边界。
- `scripts/download_datasets.ps1`：下载 SEM16 训练集并抓取 VAST 项目快照。
- `scripts/prepare_sem16.py`：把 SemEval 原始 TSV 转换为可直接评估的 CSV。

正式 SEM16 测试集和作者使用的 GPT-3.5 Turbo 快照不在本目录中，所以当前环境不能声称复现论文表格结果。

## 安装

```powershell
.\setup.ps1
```

脚本创建 `.venv`、安装 OpenAI 兼容客户端并运行离线检查。也可以手动执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 离线运行

```powershell
.\.venv\Scripts\python.exe run.py --mock --method both --csv data\smoke.csv --output results\smoke.jsonl
```

离线 mock 只验证流程、提示词拼装、标签解析和指标计算，不代表模型质量。开发集可这样检查：

```powershell
.\.venv\Scripts\python.exe run.py --mock --method cola --csv data\dev15.csv --output results\dev15-mock.jsonl
```

## 接入 Qwen 或其他提供商

复制 `configs/qwen.env.example` 为本地配置，填写 API key。也可以直接复用上级研究目录中已有的 `qwen.env`（不会复制或显示其中的密钥）：

```powershell
.\.venv\Scripts\python.exe run.py `
  --method cola `
  --csv data\dev15.csv `
  --env-file ..\COLA-Research\qwen.env `
  --model qwen-plus `
  --repeats 1 `
  --trace `
  --output results\qwen-dev15.jsonl
```

切换提供商只需替换 `OPENAI_BASE_URL`、`OPENAI_MODEL` 和 key；客户端仍使用同一套接口。`--trace` 会保存提示词和模型原文，含真实文本时应按数据许可保存。

## 真实实验建议

先用 `--limit 1` 确认端点、模型名和输出格式，再扩大到开发集；完整 COLA 每条样本需要 7 次请求。论文设置是 `temperature=0`、五次重复，运行示例：

```powershell
.\.venv\Scripts\python.exe run.py --method cola --csv data\dev15.csv --repeats 5 --env-file ..\COLA-Research\qwen.env --output results\dev15-five-runs.jsonl
```

`results` 中的 `status=summary` 行含 Accuracy、Macro-F1 和 `F_avg`（Favor/Against 两类 F1 的平均）。API 错误只记录类型和 HTTP 状态，不会把密钥写入结果。
