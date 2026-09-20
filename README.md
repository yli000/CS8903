English | [中文](README.zh-CN.md)

# GRPO Fine-Tuning of Qwen3.5-9B for ImageCLEF 2026 Multimodal Reasoning task (MCQ track)

Code for Qwen baseline and GRPO finetuning of [our working notes](https://clef-staging.pages.dev/paper272.pdf) in the imageCLEF 2026's multimodal reasoning task, MCQ track.

The task is to answer multilingual, multi-subject exam questions given as a single image containing the stem (pure text or with visuals), and multiple options.

## Layout

| Folder | Contents |
|---|---|
| [`baseline/`](baseline) | Zero-shot Qwen3.5 inference, scoring, submission formatting |
| [`experiment/`](experiment) | GRPO training with veRL: environment, data prep, reward, launch script, checkpoint merging |

`experiment/` produces a merged model. `baseline/run_inference.py` and `baseline/evaluate.py` then score it the same way they score the baselines.

Code comments are in Chinese.

## Data

Both datasets are gated. Set `HF_TOKEN` in the environment.

| | |
|---|---|
| Training | [`MBZUAI/EXAMS-V`](https://huggingface.co/datasets/MBZUAI/EXAMS-V), train + validation splits, `type == image_text` |
| Evaluation | [`SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual`](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual), official test set, 1117 questions |

Test set languages: English 500, Chinese 342, Bulgarian 110, Croatian 57, Italian 54, Serbian 54.

The `subject` field has 32 spellings across languages. `baseline/normalize_subject.py` maps them onto 9 canonical subjects by the earliest-occurring keyword in the string, so `Science (Physics, Chemistry)` → Physics and `Science (Chemistry/Biology)` → Chemistry.

The `answer_key` field has 23 surface forms: Latin `A`-`E`, Cyrillic `А`-`Д`, digits `1`-`5`, both cases. All normalized before comparison.

## Scoring

The training reward and the evaluation use one extraction rule. A response is correct only if:

1. it contains a closed `</think>`;
2. `ANSWER: X`, `FINAL ANSWER: X`, `答案：X`, or a line holding only the letter appears in the last 5 non-empty lines after that tag;
3. the letter matches the ground truth.

Anything else is `INVALID` and counts as wrong in the denominator.

## References

- Task organizers' repo — [mbzuai-nlp/ImageCLEF-MultimodalReasoning](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src). `baseline/` follows its `src/baselines/` and `src/evaluation/` structure, adapted to Qwen3.5 and offline vLLM batch inference.
- veRL — [docs](https://verl.readthedocs.io/en/latest/start/quickstart.html) · [examples](https://github.com/verl-project/verl/tree/main/examples)
- Working notes — [paper272.pdf](https://clef-staging.pages.dev/paper272.pdf)

## Citation

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

## License

MIT for the code ([`LICENSE`](LICENSE)). The working notes are © 2026 the authors, CC BY 4.0.
