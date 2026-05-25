# AGENTS.md

## Role

This is the Task C implementation repository inside the unified AutoResearch workspace.

The controller one level up owns memory, experiment queue, Colab launching, and decisions. This repo owns runnable code, artifact-producing experiment scripts, packaging helpers, and reports.

## Current Best

- C111 public score: 74.70.
- Current unsubmitted candidate: C125 exact-stack zip, built from commit `f1d5f80`.
- Current implementation path: `Qwen/Qwen3-8B-AWQ` with vLLM `0.11.0`, `awq_marlin`, greedy decoding, short language-preserving prefix, and strict exact/task postprocessors through C125.
- Runtime target: 4000 questions under 15 minutes on one NVIDIA L4.
- Final container dependency policy is controlled by the submission Dockerfile; do not assume extra packages exist unless the experiment smoke installs or packages them.
- Submission zip < 10 GB.
- Docker image < 20 GB.

## Current Active Work

C131 tests one new mechanism: strict Russian morphology/grammar templates using `pymorphy3`.

The required implementation outcome is:

```bash
python scripts/run_experiment.py --id C131 --out /content/C131_artifacts
```

or an equivalent:

```bash
python scripts/c131_russian_morph_grammar_final_smoke.py --out /content/C131_artifacts
```

The runner must produce:

```text
/content/C131_artifacts.zip
  reports/C131_russian_morph_grammar_final_smoke_report.md
  results/C131/*.summary.json
  results/C131/*.outputs.json
  logs/C131/*.log
```

## C131 Mechanism

Keep the C125 model/prompt/exact stack unchanged. Add only a strict Russian morphology/grammar template solver that uses `pymorphy3` when available and otherwise abstains.

Allowed C131 templates:
- no-context case questions should answer that context is required;
- high-confidence phrase connection types;
- imperative one-member sentence type;
- single-word morphology with ambiguity abstention;
- simple part-of-speech tagging.

Rejected C131 templates:
- morphemic composition;
- full syntax parsing;
- sentence/essay/list generation;
- retrieval/RAG;
- any new arithmetic, physics, chemistry, or prompt changes.

## Hard Boundaries

Do not submit to leaderboard unless the controller explicitly reaches a `SUBMIT` decision and the user confirms.

Do not mix mechanisms unless explicitly authorized. C131 is only the Russian morphology/grammar template solver.

Forbidden in C131:

- no leaderboard submission;
- no prompt/model/sampling/cap changes;
- no retrieval/RAG;
- no exact-cache changes;
- no morphemic parser or full syntax parser;
- no SFT/LoRA;
- no unrelated refactor.

C126 already flagged the C125 zip for human review. Do not auto-submit or stop just because a candidate exists.

## Git / Colab Sync

Before Colab:

- commit all files needed by Colab;
- push to the remote repo.

In Colab:

- clone the repo;
- run committed scripts;
- save outputs under the requested artifact root;
- zip the artifact folder for download.

## Result Discipline

Reports must include:

- environment;
- exact commands/config;
- runtime measurements;
- output validity;
- smoke checks;
- dependency/package notes for `pymorphy3`;
- comparison to C125/C130 evidence;
- qualitative examples;
- recommendation;
- strongest reason against recommendation.

## Cleanup

Do not delete user data. Do not commit large model weights, raw data, old artifact zips, or credentials.
