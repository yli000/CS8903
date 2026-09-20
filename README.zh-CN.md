[English](README.md) | 中文

# GRPO Fine-Tuning of Qwen3.5-9B for ImageCLEF 2026 Multimodal Reasoning task (MCQ track)

ImageCLEF 2026 多模态推理任务 MCQ 赛道，[我们 working notes](https://clef-staging.pages.dev/paper272.pdf) 里 Qwen baseline 和 GRPO 微调这两部分的代码。

任务是回答多语言、多学科的考试题。每道题是一张图片，图片里含题干（纯文字，或者带视觉元素）和多个选项。

## 目录结构

| 文件夹 | 内容 |
|---|---|
| [`baseline/`](baseline) | Qwen3.5 零样本推理、打分、提交格式 |
| [`experiment/`](experiment) | 用 veRL 做 GRPO 训练：环境、数据准备、reward、启动脚本、checkpoint 合并 |

`experiment/` 产出一个合并好的模型，再由 `baseline/run_inference.py` 和 `baseline/evaluate.py` 打分，用的是与 baseline 相同的那条路径。

代码注释是中文。

## 数据

两个数据集都需要申请权限。把 `HF_TOKEN` 设在环境变量里。

| | |
|---|---|
| 训练 | [`MBZUAI/EXAMS-V`](https://huggingface.co/datasets/MBZUAI/EXAMS-V)，train + validation split，`type == image_text` |
| 评测 | [`SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual`](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual)，官方测试集，1117 题 |

测试集的语言分布：英语 500、中文 342、保加利亚语 110、克罗地亚语 57、意大利语 54、塞尔维亚语 54。

`subject` 字段跨语言一共有 32 种写法。`baseline/normalize_subject.py` 按字符串中最先出现的学科关键字把它们归到 9 个类别，所以 `Science (Physics, Chemistry)` → Physics，`Science (Chemistry/Biology)` → Chemistry。

`answer_key` 字段有 23 种写法：拉丁字母 `A`-`E`、西里尔字母 `А`-`Д`、数字 `1`-`5`，大小写都有。比较之前全部归一。

## 打分

训练 reward 和评测用同一套抽取规则。一个回答算对，必须同时满足：

1. 含有闭合的 `</think>`；
2. 在该标签之后的最后 5 个非空行里出现 `ANSWER: X`、`FINAL ANSWER: X`、`答案：X`，或者单独一行只有那个字母；
3. 该字母与 ground truth 一致。

其余情况记为 `INVALID`，计入分母算错。

## 参考

- 任务方仓库 —— [mbzuai-nlp/ImageCLEF-MultimodalReasoning](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src)。`baseline/` 沿用它 `src/baselines/` 和 `src/evaluation/` 的结构，改成 Qwen3.5 和 vLLM 离线批量推理。
- veRL —— [文档](https://verl.readthedocs.io/en/latest/start/quickstart.html) · [examples](https://github.com/verl-project/verl/tree/main/examples)
- Working notes —— [paper272.pdf](https://clef-staging.pages.dev/paper272.pdf)

## 引用

```bibtex
@inproceedings{li2026dsgt,
  title     = {DS@GT at ImageCLEF 2026 MultimodalReasoning: Visual Multiple Choice
               Question Reasoning with Vision-Language Models},
  author    = {Li, Yue and Liu, Zhanxu and Zhang, Chengxi},
  booktitle = {CLEF 2026 Working Notes},
  address   = {Jena, Germany},
  year      = {2026}
}
```

## 许可

代码用 MIT 许可（[`LICENSE`](LICENSE)）。working notes 版权 © 2026 作者，CC BY 4.0。
