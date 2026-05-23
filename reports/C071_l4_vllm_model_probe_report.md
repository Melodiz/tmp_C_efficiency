# C071 L4 vLLM Model Probe Report

## Research ID and objective
- ID: C071
- Objective: Measure whether `Qwen/Qwen3-4B-Instruct-2507` can run with vLLM on one NVIDIA L4 and improve qualitatively over the C000 `Qwen/Qwen3-0.6B` baseline.
- Leaderboard submission: NO.

## Environment
- runtime provider: Colab Pro+ style notebook run, results returned as `C071_l4_vllm_model_probe_results.zip`.
- GPU: 1x NVIDIA L4, 23,659,151,360 bytes VRAM.
- CUDA/PyTorch/vLLM/Transformers versions: CUDA 12.8, PyTorch `2.8.0+cu128`, vLLM `0.11.0`, Transformers `4.56.1`.
- available disk: 236 GB filesystem, 173-182 GB free during probes.
- Colab RAM mode: not recorded in artifacts.
- Prompt shape: user-message-only chat template; no system prompt; `enable_thinking=False` passed through the template path.
- Forbidden methods check: no router, retrieval, exact cache, deterministic handlers, SFT, or LoRA were used.

## Commands/config
- repo commit used by Colab: `9d7d62b` expected from the prepared pipeline.
- baseline command: `python scripts/c071_probe.py --candidate baseline --sample-source hard_audit --sample-size 26 --max-model-len 4096 --max-tokens 384 --temperature 0.0 --top-k -1 --gpu-memory-utilization 0.9 --no-fail`
- Qwen3-4B 26-row command: `python scripts/c071_probe.py --candidate qwen3-4b --sample-source hard_audit --sample-size 26 --max-model-len 4096 --max-tokens 384 --temperature 0.0 --top-k -1 --gpu-memory-utilization 0.9 --no-fail`
- Qwen3-4B 200-row command: `python scripts/c071_probe.py --candidate qwen3-4b --sample-source hard_audit --sample-size 200 --max-model-len 4096 --max-tokens 384 --temperature 0.0 --top-k -1 --gpu-memory-utilization 0.9 --no-fail`
- fallback command used: `python scripts/c071_probe.py --candidate qwen3-1.7b --sample-source hard_audit --sample-size 26 --max-model-len 4096 --max-tokens 384 --temperature 0.0 --top-k -1 --gpu-memory-utilization 0.9 --no-fail`

## Candidates tested
| Candidate | Engine | Precision/quantization | Weight size | Package/image estimate | Status |
|---|---|---|---:|---:|---|
| C000 `Qwen/Qwen3-0.6B` | vLLM 0.11.0 | BF16 | 1.52 GB HF files | existing zip ~1.20 GB; image ~6-8 GB | Completed 26-row reference. |
| `Qwen/Qwen3-4B-Instruct-2507` | vLLM 0.11.0 | BF16 | 8.06 GB HF files | zip estimate 6.4-8.1 GB; image ~14-16 GB | Completed 26-row and 200-row probes. |
| `Qwen/Qwen3-1.7B` fallback | vLLM 0.11.0 | BF16 | 4.08 GB HF files | zip estimate 3.2-4.1 GB; image ~10-12 GB | Completed optional 26-row fallback; not attractive. |

## Runtime measurements
| Candidate | startup | sample size | avg input tokens | avg output tokens | throughput | projected 4000q runtime | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| C000 baseline | 127.93s | 26 | 45.04 | 194.04 | 1,565 out tok/s; 8.07 q/s | 623.78s / 10.40m | 21,182 MB |
| Qwen3-4B 26-row | 148.63s | 26 | 41.04 | 283.73 | 475 out tok/s; 1.67 q/s | 2,537.79s / 42.30m | 21,308 MB |
| Qwen3-4B 200-row | 54.47s | 200 | 34.99 | 290.00 | 2,010 out tok/s; 6.93 q/s | 631.66s / 10.53m | 21,306 MB |
| Qwen3-1.7B fallback | 119.99s | 26 | 45.04 | 284.31 | 962 out tok/s; 3.38 q/s | 1,302.37s / 21.71m | 21,226 MB |

Notes:
- The 26-row Qwen3-4B projection is too pessimistic because the small batch underuses vLLM batching and includes cold-load overhead. The 200-row result is the better runtime signal.
- Peak VRAM is close to vLLM's reserved-memory behavior rather than pure model size. All candidates stayed inside L4 memory with `gpu_memory_utilization=0.9`.

## Validity
- output JSONL format: PASS for all runs.
- one answer per input: PASS for all runs.
- no thinking traces: PASS; zero `<think>` rows in all runs.
- no empty answers: PASS; zero empty rows.
- max-token hit rows:
  - C000 26-row: 2/26.
  - Qwen3-4B 26-row: 14/26.
  - Qwen3-4B 200-row: 112/200.
  - Qwen3-1.7B 26-row: 10/26.
- suspected repetition/loop rows:
  - C000 26-row: 6/26.
  - Qwen3-4B 26-row: 9/26.
  - Qwen3-4B 200-row: 37/200.
  - Qwen3-1.7B 26-row: 15/26.
- language/style observations: Qwen3-4B is much more fluent and often more correct, but it is too verbose under the current user-only/greedy setup and frequently truncates at 384 tokens.

## Local quality evidence
- validation/audit sample used: C009 `hard_audit_set` joined against `data/dataset_ml_challenge.parquet`.
- 26-row category mix: exactly 2 rows each from English grammar, Russian grammar, Russian morphology, algebra/equations, arithmetic/fractions, chemistry balancing, chemistry explanation, geometry, history/geography/biology, literature/essay, other, translation, and word problems.
- 200-row category mix: English grammar 8, Russian grammar 18, Russian morphology 18, algebra/equations 13, arithmetic/fractions 29, chemistry balancing 8, chemistry explanation 8, geometry 11, history/geography/biology 11, literature/essay 20, other 31, translation 10, word problems 15.
- qualitative delta vs C000:
  - Clear 4B wins: row 2506 English comparative answered `more challenging` correctly while C000 answered `challenging`; row 9176 Roman numerals answered `MMXXIV` while C000 produced an invalid numeral; row 969 computed 15% of 3,000,000 as 450,000 while C000 inverted the relationship; row 3968 correctly gave 72,300 tens while C000 contradicted its own arithmetic.
  - Clear 4B regressions/risks: row 7190 interpreted `корень 36 64` as `sqrt(3664)` and truncated, while C000 and 1.7B answered `sqrt(36)=6`, `sqrt(64)=8`; row 736 morphology of `заставили` still contained grammatical hallucinations and hit the cap; row 4533 hallucinated country symbols in English instead of using the likely Russian national symbol pattern; long literature/essay rows often hit the token cap.
- strongest wins: basic factual/math correction, English grammar, arithmetic/word-problem reliability, and fluent structure.
- strongest regressions: verbosity, truncation, occasional over-interpretation of terse math queries, and hallucinated details when the prompt is underspecified.

## Packaging feasibility
- zip size estimate: 4B raw HF file total is 8.06 GB, within the 10 GB submission limit but with modest margin.
- image size estimate: likely 14-16 GB with the existing CUDA/PyTorch/vLLM base, under the 20 GB image cap.
- internet-free inference feasibility: feasible only if all model shards/tokenizer/config files are bundled. Not built in C071.
- fragile dependencies: vLLM 0.11.0, CUDA 12.8 compatibility, memory preallocation, and output cap behavior.

## Recommendation
- Best candidate: `Qwen/Qwen3-4B-Instruct-2507` is the best model candidate tested, but not ready for a submission experiment in this exact configuration.
- Should a submission experiment be created? NO.
- Required config for next experiment:
  - Keep Qwen3-4B and user-message-only prompt.
  - Test output control before packaging: `max_tokens=256` and `max_tokens=320` on the same 200-row hard audit sample; consider Qwen recommended non-thinking sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`) only as a separate mutation from greedy decoding.
  - Re-run 200-row timing and max-token-hit checks after each config mutation.
  - Only consider packaging if max-token hits fall sharply without breaking runtime.
- Kill conditions:
  - 4000-question projection exceeds 12 minutes on 200-row or larger sample.
  - Max-token hits remain above 25%.
  - Any thinking traces appear.
  - Qualitative review shows arithmetic/basic grammar regressions against C000.
  - Packaged zip exceeds 9.5 GB or image exceeds 18 GB.

## Risks/regressions
- runtime risk: Medium. 4B fits and the 200-row projection is 10.53 minutes, but higher `max_tokens` would likely exceed budget.
- package risk: Medium. Model files fit the zip limit but leave little room for extra assets.
- quality risk: High in current config due to truncation and some hallucinated analyses.
- operational risk: Medium. Offline packaging and vLLM memory behavior still need a packaging build test.

## Decision recommendation
MUTATE

## Strongest reason against recommendation
- Qwen3-4B passes the L4 runtime feasibility gate, but 112/200 outputs hit the 384-token cap. That is too much truncation risk for a leaderboard submission path without another output-control mutation.

## Artifacts reviewed
- `results/C071_l4_vllm_model_probe/20260523T140034Z_baseline_26.summary.json`
- `results/C071_l4_vllm_model_probe/20260523T140034Z_baseline_26.outputs.jsonl`
- `results/C071_l4_vllm_model_probe/20260523T140306Z_qwen3-4b_26.summary.json`
- `results/C071_l4_vllm_model_probe/20260523T140306Z_qwen3-4b_26.outputs.jsonl`
- `results/C071_l4_vllm_model_probe/20260523T140609Z_qwen3-4b_200.summary.json`
- `results/C071_l4_vllm_model_probe/20260523T140609Z_qwen3-4b_200.outputs.jsonl`
- `results/C071_l4_vllm_model_probe/20260523T140748Z_qwen3-1.7b_26.summary.json`
- `results/C071_l4_vllm_model_probe/20260523T140748Z_qwen3-1.7b_26.outputs.jsonl`
