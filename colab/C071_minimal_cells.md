# C071 Minimal Colab Cells

Copy each fenced block into a separate Colab cell. Use a Colab Pro+ runtime with an NVIDIA L4 GPU. These cells do not submit to the leaderboard and do not add model weights to git.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone Or Pull Repo

```bash
%%bash
cd /content
if [ -d tmp_C_efficiency/.git ]; then
  cd tmp_C_efficiency
  git pull --ff-only
else
  git clone https://github.com/Melodiz/tmp_C_efficiency
  cd tmp_C_efficiency
fi
git rev-parse --short HEAD
```

## 3. Check Runtime And Install Dependencies

```bash
%%bash
nvidia-smi
free -h
df -h .
pip install -q --upgrade "vllm==0.11.0" "transformers==4.56.1" pandas pyarrow huggingface_hub
python - <<'PY'
import torch, transformers
import vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY
```

If Colab asks for a runtime restart after installing vLLM, restart and rerun cells 2 and 3.

## 4. Run 26-Row Baseline Reference

This uses `simple_solution/weights` if present; otherwise it downloads `Qwen/Qwen3-0.6B` from Hugging Face into the Colab cache.

```bash
%%bash
cd /content/tmp_C_efficiency
python scripts/c071_probe.py \
  --candidate baseline \
  --sample-source hard_audit \
  --sample-size 26 \
  --max-model-len 4096 \
  --max-tokens 384 \
  --temperature 0.0 \
  --top-k -1 \
  --gpu-memory-utilization 0.9 \
  --no-fail
```

## 5. Run 26-Row Qwen3-4B Probe

```bash
%%bash
cd /content/tmp_C_efficiency
python scripts/c071_probe.py \
  --candidate qwen3-4b \
  --sample-source hard_audit \
  --sample-size 26 \
  --max-model-len 4096 \
  --max-tokens 384 \
  --temperature 0.0 \
  --top-k -1 \
  --gpu-memory-utilization 0.9 \
  --no-fail
```

Inspect the newest `*.summary.json`. Continue only if status is `completed`, projected 4000-question time is safe, VRAM is safe, and validity has no thinking traces or loops.

```bash
%%bash
cd /content/tmp_C_efficiency
ls -lh results/C071_l4_vllm_model_probe
python - <<'PY'
import json
from pathlib import Path
latest = sorted(Path("results/C071_l4_vllm_model_probe").glob("*qwen3-4b_26.summary.json"))[-1]
print(latest)
print(json.dumps(json.loads(latest.read_text()), ensure_ascii=False, indent=2)[:6000])
PY
```

## 6. Run 200-Row Qwen3-4B Probe If 26-Row Passes

```bash
%%bash
cd /content/tmp_C_efficiency
python scripts/c071_probe.py \
  --candidate qwen3-4b \
  --sample-source hard_audit \
  --sample-size 200 \
  --max-model-len 4096 \
  --max-tokens 384 \
  --temperature 0.0 \
  --top-k -1 \
  --gpu-memory-utilization 0.9 \
  --no-fail
```

## 7. Optional Qwen3-1.7B Fallback

Run this only if 4B is blocked, unsafe, or too slow.

```bash
%%bash
cd /content/tmp_C_efficiency
python scripts/c071_probe.py \
  --candidate qwen3-1.7b \
  --sample-source hard_audit \
  --sample-size 26 \
  --max-model-len 4096 \
  --max-tokens 384 \
  --temperature 0.0 \
  --top-k -1 \
  --gpu-memory-utilization 0.9 \
  --no-fail
```

## 8. Zip Results To Drive

```bash
%%bash
cd /content/tmp_C_efficiency
mkdir -p /content/drive/MyDrive/task_c_results
zip -r /content/drive/MyDrive/task_c_results/C071_l4_vllm_model_probe_results.zip results/C071_l4_vllm_model_probe
ls -lh /content/drive/MyDrive/task_c_results/C071_l4_vllm_model_probe_results.zip
```
