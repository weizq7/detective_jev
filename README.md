# Jev as a Detective — Does a System-One Model Have Reasoning Intuition?

This repo evaluates [**Jev**](https://typesafe.ai) — a recently popular *System One* model from TypeSafe — on the [**DetectiveQA**](https://huggingface.co/datasets/Phospheneser/DetectiveQA) benchmark of English detective-novel multiple-choice questions.

The question we wanted to answer: **does Jev have any *reasoning intuition*?** Jev never emits explicit text or a chain of thought — it just returns typed, calibrated decisions (choice / score / yes-no) very fast and fully in parallel across questions. In that sense any reasoning it does is intuitive, not deliberative. So if it can read a whole detective novel and pick the right suspect / motive / clue, that's *intuition* about reasoning.

## The problem: Jev's context length is limited

Jev enforces two simultaneous context budgets — **64k tokens for `state` + all questions combined**, and **32k tokens for `state` + the single longest question** — and whichever limit is hit first is the binding constraint. But even the *shortest* novel in the dataset (novel 132, *The Body in the Library* by Agatha Christie) is **272,417 characters** — well over 60k tokens. There's no way to feed a whole detective novel in one shot.

## The pipeline: letting Jev compress its own context

`two_phase_detective_jev.py` implements a two-phase pipeline where **Jev selects its own context**:

**Phase 1 — chunk scoring.** The novel is split into ~50k-character chunks at paragraph boundaries. Every chunk is sent to Jev along with:

- a rolling background context (novel opening + tail of previously high-scoring chunks, so Jev is never reading cold), and
- one `score`-type question **per** multiple-choice question, asking how relevant the current chunk is to each question on a 5-level ordered scale (0 = unrelated → 4 = direct answer / key revelation).

Because Jev evaluates all questions in a single request in parallel, one HTTP call per chunk grades the chunk against every multiple-choice question at once. Each chunk gets a **total relevance score = sum of per-question scores**.

**Phase 2 — final answer selection.** The highest-scoring chunks are greedily picked until the final `state` reaches ~100k characters, then re-sorted into original narrative order (so the story still flows). A short meta-header tells Jev the passages are non-contiguous excerpts. The assembled `state` plus all multiple-choice questions are sent in one `choice`-type request and Jev's answers are scored against the gold labels.

The neat part: **which chunks to keep is decided by Jev itself**, using its own `score` primitive. No external model, no embedding search — just Jev asking itself "how relevant is this passage to answering these questions?"

## Results

Evaluated on all **86 English detective novels** in DetectiveQA. We swept `--chunk-size` across three values to see the trade-off between phase-1 granularity and total cost:

| `--chunk-size` | Macro-avg accuracy | Micro-avg accuracy | Correct / Total Qs | Total wall time | Time / novel |
|---------------:|-------------------:|-------------------:|-------------------:|----------------:|-------------:|
| 10,000 | **75.84%** | **76.33%** | 458 / 600 | 215 min | 150 s |
| 25,000 | 74.23% | 74.67% | 448 / 600 | 60 min | 42 s |
| 50,000 | 70.48% | 71.00% | 426 / 600 | 40 min | 28 s |

The trend is clean: **smaller chunks → higher accuracy, at a linear-ish cost in wall time**. Halving the chunk size doubles the number of phase-1 requests, but each additional split gives Jev a finer look at what is actually relevant. Going from 50k → 25k buys +3.8 accuracy points at 1.5× the cost; going 25k → 10k adds another +1.6 points at 3.6× the cost. Even all three sweeps combined (3 × 86 = 258 novel runs) cost **less than $2** in input tokens (Jev is $0.042 / M input tokens; output tokens are free).

At `--chunk-size = 50000`, Jev ingested ~45.1M chars of state across the sweep (~11.3M tokens at English's ~4 chars/token) in ~2,384 s of cumulative response time — an **end-to-end throughput of ~4,700 tokens per second**. Note that this counts full request wall time (network round-trip + SSL + auth + Jev's own compute), so it is a lower bound on Jev's actual model throughput; the pure-compute number is presumably higher. Even so, it's hard to match with a frontier chat model on a detective-novel benchmark.

### How Jev compares to LLMs

We borrow numbers for the open-source LLMs from the DetectiveQA paper as a reference point:

| Model | Context | DetectiveQA Accuracy |
|-------|--------:|---------------------:|
| **Jev + our two-phase harness** | **32k** | **76.33%** *(best sweep)* |
| Qwen2.5-7B-Instruct | 128k | 61.75% |
| InternLM2.5-7B-chat | 1M | 60.92% |
| GLM4-9B-chat | 1M | 59.00% |
| InternLM2-7B-chat | 200k | 57.95% |
| GLM3-6B | 128k | 40.58% |
| LLaMA3.1-8B-Instruct | 128k | 28.17% |

Jev with our two-phase harness beats every open-source model in that leaderboard by ≥13 points — while (a) being a System-One classifier, not a generative model, (b) fitting into a 32k budget vs 128k–1M, and (c) costing pennies.

### So — does Jev have reasoning intuition?

Based on our experiment, we'd cautiously say **yes**. Jev landed on the correct A/B/C/D in **76.33%** of DetectiveQA's long-context reasoning questions, without emitting a single word of reasoning and without ever holding more than 32k tokens of the novel at once. Even on a task explicitly built to require long-range narrative inference, Jev picks the right choice often enough to support the intuition hypothesis holds up in practice.

I think that if TypeSafe ever trains a Jev variant with a million-token native context, it would not only eliminate the extra end-to-end time our two-phase harness adds, but also almost certainly deliver higher accuracy — because as a non-generative model, Jev can only compress by *deleting* unimportant chunks and stitching together the disjoint fragments left behind, rather than reading a passage and writing its own summary the way a generative model would.

## Usage

Set your Jev API key, then:

```bash
# 1. Install dependencies
python -m pip install requests huggingface_hub

# 2. Download the dataset from HuggingFace
python download_dataset.py

# 3. Set your Jev key (get one from https://typesafe.ai)
export TYPESAFE_API_KEY="apikey_..."       # bash
# $env:TYPESAFE_API_KEY = "apikey_..."     # PowerShell

# 4a. Evaluate a single novel
python two_phase_detective_jev.py --novel-id 132

# 4b. Or evaluate all novels
python two_phase_detective_jev.py --all
```

Results are written to `jev_results_twophase/novel_{id}_{split}.json`, one file per novel, containing every phase-1 chunk score, the final phase-2 answers, and the accuracy.

## Further reading

For a very good reverse-engineered analysis of **what Jev actually is and how RLCD (their training recipe) probably works**, see:

- **[What is RLCD, the secret behind Jev? — Di Zhang](https://di-zhang-llm.github.io/blog/what-is-rlcd-the-secret-behind-jev/)**

TypeSafe hasn't open-sourced Jev or published training details, so any claims about the architecture and recipe are necessarily speculative — but the linked blog is the most reasonable public write-up I've seen so far.

## Files

- `download_dataset.py` — fetches the DetectiveQA dataset from Hugging Face (through a configurable HTTP proxy)
- `two_phase_detective_jev.py` — the two-phase Jev evaluation pipeline described above
