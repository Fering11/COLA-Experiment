# SEM16 适配

当前运行器需要 UTF-8 CSV，列为 `id,text,target,label`。SemEval-2016 Task 6 原始文件通常是制表符分隔，列为 `ID,Target,Tweet,Stance`。转换器会保留原始文本和目标，映射：`FAVOR -> favor`、`AGAINST -> against`、`NONE -> neutral`。

## 转换

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sem16.py `
  data\external\sem16\semeval2016-task6-trainingdata-utf-8.txt `
  --output data\sem16_train.csv
```

如果取得了官方测试文件，使用相同命令输出到 `data/sem16_test.csv`，不要覆盖训练集：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sem16.py `
  data\external\sem16\semeval2016-task6-testdata-utf-8.txt `
  --output data\sem16_test.csv
```

## 运行

先用一条真实样本检查接口：

```powershell
.\.venv\Scripts\python.exe run.py --method cola --csv data\sem16_train.csv --limit 1 --env-file .dp.env --trace --output results\sem16-first1.jsonl
```

接口正常后再扩大范围。完整 COLA 每条样本需要 7 次模型请求；`--mock` 只检查流程，不产生有效模型结果。运行器默认使用 `data/sem16_train.csv`、2 个外层 worker，并显示完成进度；请求遇到瞬时网络错误会退避重试，最终失败的样本仍会写入结果。

转换器遇到未知标签、重复 ID、空文本或缺少列时会停止并报告行号，避免生成错误评测文件。转换后终端输出的 SHA-256 建议记录到实验日志。
