# C071 L4 vLLM Model Probe Report

## Research ID and objective
- ID: C071
- Objective: Measure whether `Qwen/Qwen3-4B-Instruct-2507` can run with vLLM on one NVIDIA L4 and improve qualitatively over the C000 `Qwen/Qwen3-0.6B` baseline.
- Leaderboard submission: NO

## Environment
- runtime provider:
- GPU:
- `nvidia-smi`:
- CUDA/PyTorch/vLLM/Transformers versions:
- available disk:
- Colab RAM mode:

## Commands/config
- repo commit:
- baseline command:
- Qwen3-4B 26-row command:
- Qwen3-4B 200-row command:
- fallback command, if used:
- prompt shape: user-message-only, no system prompt
- forbidden methods check: no router, retrieval, exact cache, deterministic handlers, SFT, or LoRA

## Candidates tested
| Candidate | Engine | Precision/quantization | Weight size | Package/image estimate | Status |
|---|---|---|---:|---:|---|
| C000 `Qwen/Qwen3-0.6B` | vLLM | BF16 | | | |
| `Qwen/Qwen3-4B-Instruct-2507` | vLLM | BF16 | | | |
| `Qwen/Qwen3-1.7B` fallback | vLLM | BF16 | | | |

## Runtime measurements
| Candidate | startup | sample size | avg input tokens | avg output tokens | throughput | projected 4000q runtime | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| C000 baseline | | | | | | | |
| Qwen3-4B 26-row | | | | | | | |
| Qwen3-4B 200-row | | | | | | | |
| Qwen3-1.7B fallback | | | | | | | |

## Validity
- output JSONL format:
- one answer per input:
- no thinking traces:
- no generation loops:
- max-token hit rows:
- empty answer rows:
- language/style observations:

## Local quality evidence
- validation/audit sample used:
- category mix:
- qualitative delta vs C000:
- strongest wins:
- strongest regressions:

## Packaging feasibility
- zip size estimate:
- image size estimate:
- internet-free inference feasibility:
- fragile dependencies:

## Recommendation
- Best candidate:
- Should a submission experiment be created? YES/NO
- Required config for next experiment:
- Kill conditions:

## Risks/regressions
- runtime risk:
- package risk:
- quality risk:
- operational risk:

## Decision recommendation
MERGE / KILL / MUTATE / SUBMIT / INVESTIGATE

## Strongest reason against recommendation
- ...
