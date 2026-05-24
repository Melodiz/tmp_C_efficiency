# AGENTS.md

## Role

This is the Task C implementation repository. It contains runnable code, data access, packaging, GPU experiments, and reports returned to the Meta-Orchestrator.

The Meta-Orchestrator repo is separate. This repo executes experiments and returns structured reports.

## Current Best

- C000 baseline public score: 46.00.
- Verdict: OK.
- Baseline model: `Qwen/Qwen3-0.6B`.
- Runtime target: 4000 questions under 15 minutes on one NVIDIA L4.
- Final container has no internet.
- Submission zip < 10 GB.
- Docker image < 20 GB.

## Hard Boundaries

Do not submit to leaderboard unless the active prompt explicitly authorizes it.

Do not mix mechanisms unless explicitly authorized. One experiment should test one mechanism.

Do not silently keep failed experiment changes. If an experiment is killed, reset or disable its behavior before the next experiment.

C010 long global system prompt is killed. Do not use it in future probes.

## Current Active Experiment

C072 Qwen3-4B output control.

Goal:
Measure whether output length control for `Qwen/Qwen3-4B-Instruct-2507` reduces truncation and repetition risk while preserving the C071 L4 runtime feasibility signal.

Rules:
- no leaderboard submission;
- no system prompt;
- user-message-only prompt shape;
- no router;
- no retrieval;
- no exact cache;
- no deterministic handlers;
- no SFT/LoRA;
- no model larger than 4B-5B;
- use Qwen3-4B only unless explicitly redirected;
- test output-control changes only; do not add prompt, router, retrieval, cache, deterministic handlers, SFT, or LoRA in C072.

## Workflow Reference v4 Distillation

Use the Colab loop:

1. Codex implements `.py` files and thin Colab cells.
2. User runs cells in Colab.
3. If a cell fails, user pastes error back to Codex.
4. Codex fixes code and updates cells.
5. Colab saves result artifacts as a zip.
6. User downloads zip and places it back in this repo.
7. Codex unzips, reviews outputs, writes complete report, and cleans temporary files.

Notebook cells should be thin wrappers around committed Python files. Do not put substantial experiment logic only in notebook cells.

## Git / Colab Sync

Before Colab:
- commit all files needed by Colab;
- push to the remote repo.

In Colab:
- clone or pull the repo;
- run committed scripts;
- save outputs under `results/<experiment_id>/` or the experiment artifact root;
- zip the result folder for download.

After Colab:
- user downloads zip;
- user places zip in this repo;
- Codex unzips and writes final report.

## Result Discipline

Every experiment writes:
- result artifacts under `results/<experiment_id>/`;
- report under `reports/<experiment_id>_report.md`.

For C071, use:
- `results/C071_l4_vllm_model_probe/`
- `reports/C071_l4_vllm_model_probe_report.md`

For C072 CLI artifacts, use the packaged artifact root:
- `reports/C072_qwen3_4b_output_control_report.md`
- `results/C072/*.summary.json`
- `results/C072/*.metrics.json`
- `results/C072/*.outputs.jsonl`
- `logs/C072/*.log`

The Colab CLI entrypoint is:

```bash
python scripts/run_experiment.py --id C072 --out /content/C072_artifacts
```

It must produce:

```text
/content/C072_artifacts.zip
```

The per-experiment equivalent is:

```bash
python scripts/c072_output_control.py --out /content/C072_artifacts
```

Reports must include:
- environment;
- exact commands/config;
- runtime measurements;
- output validity;
- qualitative examples;
- package feasibility;
- recommendation;
- strongest reason against recommendation.

## Cleanup

Do not delete user data. Move obsolete temporary files to an archive folder or leave them for user confirmation.

Do not commit large model weights unless the active prompt explicitly asks for packaging/submission work.
