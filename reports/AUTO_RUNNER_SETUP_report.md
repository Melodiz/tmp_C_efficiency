# Auto Hill-Climb Runner Setup Report

## Objective
- ID: AUTO_RUNNER_SETUP

## Files changed
- `.gitignore`
- `AGENTS.md`
- `scripts/run_experiment.py`
- `scripts/c072_output_control.py`
- `colab/C072_cli_cells.md`
- `reports/AUTO_RUNNER_SETUP_report.md`

## Commands added
- `python scripts/run_experiment.py --id C072 --out /content/C072_artifacts`
- `python scripts/c072_output_control.py --out /content/C072_artifacts`
- Dry-run form: `python scripts/run_experiment.py --id C072 --out artifacts/tmp/C072_artifacts --dry-run`

## Dry-run/local validation
- command: `python -m py_compile scripts/run_experiment.py scripts/c072_output_control.py`
- result: PASS
- command: `python scripts/run_experiment.py --id C072 --out artifacts/tmp/C072_artifacts --dry-run`
- result: PASS; created ignored dry-run artifact at `artifacts/tmp/C072_artifacts.zip` with `reports/C072_qwen3_4b_output_control_report.md`, `results/C072/*.summary.json`, `results/C072/*.metrics.json`, `results/C072/*.outputs.jsonl`, and `logs/C072/*.log`.
- command: `python scripts/c072_output_control.py --out artifacts/tmp/C072_direct_artifacts --dry-run --max-token-variants 256`
- result: PASS; per-experiment runner produced a sibling zip.

## Colab compatibility
- expected working directory: repo root after cloning or pulling under `/content`, for example `/content/tmp_C_efficiency`
- artifact path: `/content/C072_artifacts.zip`
- package/install assumptions: Python environment has `vllm==0.11.0`, `transformers==4.56.1`, `pandas`, `pyarrow`, and `huggingface_hub`; the script does not install dependencies or use Drive.

## C072 readiness
- ready to run via CLI: YES
- exact command: `python scripts/run_experiment.py --id C072 --out /content/C072_artifacts`
- expected artifact: `/content/C072_artifacts.zip`

## Risks
- Colab still needs the vLLM/Transformers dependency install step before the runner command.
- The default C072 command runs two max-token variants, `256` and `320`, so the run time is longer than a single probe.
- The wrapper reuses `scripts/c071_probe.py`; any future interface change there can affect C072.

## Next
- Commit and push these infrastructure files before launching Colab.
- In Colab, run `python scripts/run_experiment.py --id C072 --out /content/C072_artifacts` from the repo root.
