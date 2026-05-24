# AGENTS.md

## Role

This is the Task C implementation repository inside the unified AutoResearch workspace.

The controller one level up owns memory, experiment queue, Colab launching, and decisions. This repo owns runnable code, artifact-producing experiment scripts, packaging helpers, and reports.

## Current Best

- C000 baseline public score: 46.00.
- Verdict: OK.
- Baseline model: `Qwen/Qwen3-0.6B`.
- Runtime target: 4000 questions under 15 minutes on one NVIDIA L4.
- Final container has no internet.
- Submission zip < 10 GB.
- Docker image < 20 GB.

## Current Active Work

C074 unblocks C073.

The required implementation outcome is:

```bash
python scripts/run_experiment.py --id C073 --out /content/C073_artifacts
```

or an equivalent:

```bash
python scripts/c073_short_prefix_output_control.py --out /content/C073_artifacts
```

The runner must produce:

```text
/content/C073_artifacts.zip
  reports/C073_qwen3_4b_short_prefix_output_control_report.md
  results/C073/*.summary.json
  results/C073/*.metrics.json
  results/C073/*.outputs.jsonl
  logs/C073/*.log
```

## C073 Mechanism

Run Qwen3-4B-Instruct-2507 with the same C071/C072 L4/vLLM setup, but prepend exactly one short instruction inside the user message:

```text
Ответь кратко и точно. Не повторяй условие. В конце дай итоговый ответ.
```

Run `short_prefix_320` first. Optionally run `short_prefix_384` only if 320 improves cap-hit rate but appears clipped.

## Hard Boundaries

Do not submit to leaderboard unless the controller explicitly reaches a `SUBMIT` decision and the user confirms.

Do not mix mechanisms unless explicitly authorized. C073 is only short user-prefix output control.

Forbidden in C073/C074:

- no leaderboard submission;
- no system prompt;
- no long prompt or numbered rule list;
- no router;
- no retrieval;
- no exact cache;
- no deterministic handlers;
- no SFT/LoRA;
- no packaging build;
- no unrelated refactor.

C010 long global system prompt is killed. Do not use it.

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
- cap hits and repetition suspects;
- comparison to C071 raw 384 and C072 cap-only 320;
- qualitative examples;
- recommendation;
- strongest reason against recommendation.

## Cleanup

Do not delete user data. Do not commit large model weights, raw data, old artifact zips, or credentials.

