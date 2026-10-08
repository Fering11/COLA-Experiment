# 论文数据集来源

## SEM16 / SemEval-2016 Task 6

- 官方任务页：<https://alt.qcri.org/semeval2016/task6/>
- 可信数据镜像：<https://github.com/emsrc/SemEval2016_T6_Stance_Detection>
- 训练集原始文件：<https://raw.githubusercontent.com/emsrc/SemEval2016_T6_Stance_Detection/master/semeval2016-task6-trainingdata-utf-8.txt>

该仓库中的训练文件可用于开发测试。官方测试标签通常不公开，完整测试评测需要按 SemEval 任务规则取得数据或提交预测到官方评测流程。

## VAST

- 论文项目页（作者仓库）：<https://github.com/emilyallaway/zero-shot-stance>
- 仓库数据目录：<https://github.com/emilyallaway/zero-shot-stance/tree/master/data>

VAST 的仓库结构和文件名可能随版本变化；下载后应记录 commit、文件名和 SHA-256，并核对 train/dev/test 划分。

## P-Stance（论文的另一主数据集）

- 作者/项目代码仓库：<https://github.com/declare-lab/stance-detection>
- GitHub 代码搜索入口：<https://github.com/search?q=P-Stance+stance+detection&type=repositories>

P-Stance 的推文文本受 Twitter/X 数据许可约束，部分发布物只提供 tweet ID，需要重新 hydrate；不要把仓库中的缓存或派生文件自动当作完整官方数据集。

## 下载与审计规则

下载后建议保存到 `data/external/<dataset>/`，同时记录：来源 URL、下载日期、Git commit 或 HTTP ETag、SHA-256、许可证和标签映射。统一转换为本项目格式：

```csv
id,text,target,label
```

标签映射为 `favor`、`against`、`neutral`。VAST 和 SEM16 可直接映射三分类；P-Stance 通常只有 `favor/against`。
