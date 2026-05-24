# C072 CLI Colab Cells

Copy each fenced block into a separate Colab cell. Use a Colab Pro+ runtime with an NVIDIA L4 GPU. These cells do not submit to the leaderboard and do not use Google Drive.

## 1. Clone Or Pull Repo

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

## 2. Check Runtime And Install Dependencies

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

If Colab asks for a runtime restart after installing vLLM, restart and rerun cells 1 and 2.

## 3. Run C072 And Package Artifacts

```bash
%%bash
cd /content/tmp_C_efficiency
python scripts/run_experiment.py --id C072 --out /content/C072_artifacts
ls -lh /content/C072_artifacts.zip
```

The downloadable artifact is `/content/C072_artifacts.zip`.
