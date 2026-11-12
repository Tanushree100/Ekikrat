# Ekikrat
The amalgamation for detecting Hallucination, Ambiguity and Semantic Understanding
# Ekikrat

**Detecting grammatical correctness, ambiguity and hallucination in question answering**

*One QA example. Three failure modes. One labelled corpus.*

Group 11 • Foundation of Large Language Models (FLLM, AI60213) • IIT Kharagpur

| Name | Roll No. |
|---|---|
| Hemangee De | 26CS91F02 |
| Tanushree Dhali | 26CS91R08 |
| Debanjan Naskar | 26CD91R01 |

---

## Overview

Question-answering (QA) systems can fail in three different ways:

| Axis | Failure mode | Question asked |
|---|---|---|
| **Grammar** | Malformed answer | Is the answer acceptable as sentence-level text? |
| **Ambiguity** | Under-specified question | Does the question admit several reasonable readings? |
| **Hallucination** | Unsupported answer | Does the answer contradict or lack support from the context? |

Public datasets label only one of these phenomena, so no single resource lets us ask, for example, whether ambiguous questions cause more hallucinated answers. **Ekikrat** merges three public datasets into one corpus of **1,500 QA examples** labelled on all three axes.

**Task definition:** given a QA example `x = (context, question, answer)` plus its source-dataset annotations, produce a label vector `y = (g, m, h)`, where each of `g`, `m`, `h` is in `{Yes, No, N/A}`. Ambiguity also records its sources.

## The corpus

500 rows are sampled from each source (seed = 42), normalised to a common schema, concatenated and shuffled:

| Source | Split | Rows |
|---|---|---|
| AmbigQA | validation | 500 |
| SQuAD 2.0 | validation | 500 |
| HaluBench | test | 500 |
| **Total** | | **1,500** |

- Original dataset labels are kept as reference columns.
- Answers are short: median 3 words (mean 7.9). 277 are empty, and all of them are SQuAD 2.0 unanswerables.
- Answer types: 637 sentence-level, 586 short span, 277 empty.
- HaluBench composition: HaluEval 343, FinanceBench 37, PubMedQA 32, RAGTruth 31, DROP 30, CovidQA 27.

## Labelling pipeline

Three public datasets → common schema → per-axis labellers → provenance + review flags → labelled corpus → model benchmark.

**Principle: ground truth first, models second.** Dataset annotations take priority. Automatic models are used only where no dataset label exists, and uncertainty is preserved for human review.

### 1. Grammar
- Empty answers and short spans (≤ 3 words, no final punctuation) → `N/A`.
- Remaining answers are scored sentence by sentence with `textattack/roberta-base-CoLA`.
- Mean score ≥ 0.5 → `Yes`. Scores in [0.3, 0.7] are flagged for human review.

### 2. Ambiguity
- **AmbigQA:** the original `multipleQAs` annotation is used (`multipleQAs` = ambiguous) and the disambiguated readings are kept.
- **SQuAD 2.0 / HaluBench:** a Qwen2.5-3B-Instruct judge returns a JSON verdict with sources.

### 3. Hallucination
- **HaluBench:** `FAIL` = hallucinated, `PASS` = faithful.
- **AmbigQA / SQuAD 2.0:** gold answers are labelled not hallucinated.
- The LLM verdict is cross-checked against dataset labels. Disagreements and `Unsure` outputs are flagged for review.

### Algorithm sketch (per QA row)

```text
if answer is empty: labels = N/A
else: clean answer
  if short span (<=3 words, no final punct.):
      grammar = N/A
  else: score sentences with RoBERTa-CoLA
      grammar = Yes if mean >= 0.5
      review if score in [0.3, 0.7]
judge (q, c, a) with Qwen2.5-3B
AmbigQA  : ambiguity = (type = multipleQAs)
           hallucination = No (gold answer)
SQuAD    : ambiguity = LLM verdict
           hallucination = No (gold answer)
HaluBench: ambiguity = LLM verdict
           hallucination = (label = FAIL)
flag dataset/LLM disagreement or Unsure
```

### Reliability safeguards
- **Runtime sanity check:** finds the CoLA "acceptable" output index and aborts if the grammatical-vs-scrambled score gap is below 0.2.
- **30-row pilot:** smoke-tests the judge and counts unparseable outputs before a full run.
- **Cached outputs:** runs resume after a disconnect and results are reproducible.
- **Explicit provenance:** every label records `dataset_label`, `llm` or `empty_answer`.

## Label coverage

| Axis | Labelled | Not yet labelled | N/A |
|---|---|---|---|
| Grammar | 637 | 0 | 863 |
| Ambiguity | 528 | 695 | 277 |
| Hallucination | 1,223 | 0 | 277 |

- **Grammar:** 92.5% of the 637 sentence-level answers are acceptable. 112 answers fall in the review band.
- **Ambiguity:** 277 of 500 AmbigQA questions (55.4%) are ambiguous. 500 dataset labels plus 28 LLM-pilot labels exist, and 695 rows await the full judge run.
- **Hallucination:** 226 of 500 HaluBench answers (45.2%) are hallucinated. Rates range from 19.4% (RAGTruth) to 56.7% (DROP).

## Benchmark

Four open instruction-tuned models, loaded in 4-bit NF4 on a free Colab T4: **Llama-3.1-8B**, **Qwen2.5-7B**, **Gemma-2-9B** and **Mistral-7B-v0.3**. Greedy decoding, seed 0, same sample for every model.

**Test A: models as answerers (n = 160, 4 groups × 40)**
AmbigQA single-answer, AmbigQA multi-answer, SQuAD 2.0 answerable and SQuAD 2.0 unanswerable.
Metrics: hallucination rate, wrong-answer rate, ambiguity-flag rate, interpretation coverage.

**Test B: models as judges**
Hallucination (277 rows), ambiguity (68 rows), understandability (217 rows), scored against corpus labels.
Metrics: accuracy, precision, recall, F1 of the positive class, % unparseable.

### Key results

| Model | SQuAD answerable acc. | SQuAD unanswerable halluc. ↓ | Halluc. judge F1 |
|---|---|---|---|
| Llama-3.1 | **87.5%** | 27.5% | 0.186 |
| Qwen2.5 | 80.0% | **15.0%** | 0.122 |
| Gemma-2 | 80.0% | 35.0% | 0.513 |
| Mistral | 67.5% | 22.5% | **0.551** |

- Qwen2.5 abstains most and has the lowest hallucination rate. Llama-3.1 is the most accurate on answerable questions.
- Mean interpretation coverage of ambiguous questions is only 0.06–0.21.
- Closed-book hallucination judging is still hard. Gemma and Mistral reach recall 1.00 on ambiguity by answering "Yes" almost uniformly.
- 0% of outputs were unparseable across all 12 model–task runs.

## Limitations

- The LLM judge was run on only 30 rows, and 695 ambiguity rows are still unlabelled.
- The 3B judge over-flagged ambiguity: 10 of 10 sampled SQuAD questions were marked ambiguous. It also answered "Unsure" on 15 of 20 HaluBench rows and caught 1 of 8 hallucinated answers.
- No human validation yet. Judge scores measure agreement with corpus labels, not necessarily human truth.
- HaluBench context-field bug: passages could not be re-attached, so those cases are judged closed-book.
- HaluEval dominates HaluBench (68.6%).
- Small 40-example groups, one prompt and one seed. One question moves a rate by 2.5 points, so small differences should not be over-read.

## Next steps

- Repair HaluBench context loading.
- Run a larger judge over the full corpus.
- Collect human annotations and measure inter-annotator agreement.
- Use ensemble judging and route disagreements to humans.
- Rebalance HaluBench sources and extend to other languages.

## Tech stack

Open-source Hugging Face Transformers pipeline • `textattack/roberta-base-CoLA` • Qwen2.5-3B-Instruct (labelling judge) • 4-bit NF4 quantisation • greedy, cached LLM judging • context truncated to 3,000 characters • runs on a free Colab T4 GPU.

## Repository structure

> Update this section to match your repo.

```text
.
├── data/            # sampled source data and the final labelled corpus
├── notebooks/       # labelling pipeline and benchmark notebooks (Colab)
├── results/         # benchmark outputs and metrics
├── slides/          # project presentation
└── README.md
```

## Getting started

> Update these steps to match your code.

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

Open the notebooks in Google Colab (a free T4 GPU is sufficient) and run them in order: corpus construction → labelling → benchmark.

## Datasets and models

- [AmbigQA](https://nlp.cs.washington.edu/ambigqa/)
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench)
- Models: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3, `textattack/roberta-base-CoLA`

Please follow each dataset's and model's own license and terms of use.

## Acknowledgements

Course project for **Foundation of Large Language Models (AI60213)**, IIT Kharagpur.
