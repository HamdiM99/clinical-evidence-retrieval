# Fine-tuning the re-ranker on a free GPU (Kaggle)

The training script is self-contained (`scripts/train_reranker.py`, torch + transformers only), so the
notebook only needs that file and the training data. Colab works the same way.

## 1. Prepare the files locally

```powershell
.\.venv\Scripts\cer.exe candidates configs/nfcorpus_hybrid.yaml --split train --split dev
.\.venv\Scripts\cer.exe candidates configs/scifact_hybrid.yaml --split train
.\.venv\Scripts\python.exe scripts/build_rerank_train.py          # -> data/rerank-train/
Copy-Item scripts/train_reranker.py data/rerank-train/
```

`data/rerank-train/` now holds `train.jsonl` (≈ 137 MB), `valid.jsonl` (≈ 15 MB), `stats.json` and the
script. These files are derived from BEIR datasets: keep the Kaggle dataset **private**.

## 2. Kaggle setup (once)

1. Create an account and verify a phone number (Settings → Phone verification): required for GPU and
   internet access in notebooks.
2. Datasets → New dataset → upload the four files of `data/rerank-train/`, visibility **Private**,
   name it e.g. `cer-rerank-train`.
3. From the dataset page → New notebook. Settings: Accelerator **GPU T4** (or P100), Internet **On** (to
   download the base model from Hugging Face). Kaggle mounts the dataset under
   `/kaggle/input/datasets/<user>/<dataset>/`: the cell below finds it.

## 3. Train

In a notebook cell:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip list 2>/dev/null | grep -E "^(torch|transformers) "
import glob, os
D = os.path.dirname(glob.glob("/kaggle/input/**/train_reranker.py", recursive=True)[0])
print(D)   # e.g. /kaggle/input/datasets/<user>/cer-rerank-train
!python {D}/train_reranker.py --train {D}/train.jsonl --valid {D}/valid.jsonl \
    --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out /kaggle/working/minilm-l6-nfc-scifact \
    --epochs 1 --batch-size 32 --lr 2e-5 --eval-every 500
!cd /kaggle/working && zip -qr minilm-l6-nfc-scifact.zip minilm-l6-nfc-scifact
```

The first log line (`"step": 0`) evaluates the first stage and the base model on the validation set;
for MiniLM it should be close to the step-3 tuning numbers (NFCorpus dev ≈ 0.331, SciFact train[:200]
≈ 0.730 at top 20). Each evaluation prints nDCG@10 per dataset; the best checkpoint is kept.

To let it run without keeping the browser open: **Save Version → Save & Run All**; outputs stay in the
version's Output tab.

### Ablation: loss, learning rate, dataset mix

The first run above (pointwise BCE, lr 2e-5, both datasets) never beat the base model (see
`reports/m2-reranking.md`). Four runs, each testing one hypothesis, ≈ 40 GPU minutes in total; selection on
validation only:

```python
runs = {
    "A-bce-lr5e6":          "--loss bce --lr 5e-6",
    "B-listwise":           "--loss listwise --group-size 8 --lr 1e-5",
    "C1-listwise-nfcorpus": "--loss listwise --group-size 8 --lr 1e-5 --only-dataset nfcorpus",
    "C2-listwise-scifact":  "--loss listwise --group-size 8 --lr 1e-5 --only-dataset scifact --eval-every 50",
}
for name, extra in runs.items():
    every = "" if "--eval-every" in extra else "--eval-every 250"
    !python {D}/train_reranker.py --train {D}/train.jsonl --valid {D}/valid.jsonl         --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out /kaggle/working/{name} --batch-size 32 {every} {extra}

import json
for name in runs:
    i = json.load(open(f"/kaggle/working/{name}/training_info.json"))
    print(name, "best step", i["best_step"], "/", i["steps"], "| selected on", i["select_on"],
          "| best", {k: round(v, 4) for k, v in i["best"].items()})
!cd /kaggle/working && for d in A-* B-* C1-* C2-*; do zip -qr $d.zip $d; done; ls -la /kaggle/working
```

Only download the zips whose best step is > 0 (a best step of 0 means the base model was kept).

Optional, larger model (≈ 10× slower):

```python
!python {D}/train_reranker.py --train {D}/train.jsonl --valid {D}/valid.jsonl \
    --base BAAI/bge-reranker-base --out /kaggle/working/bge-base-nfc-scifact \
    --epochs 1 --batch-size 16 --lr 1e-5 --max-length 256 --eval-every 1000
```

## 4. Bring the model back and evaluate

Download the zip from the Output tab, unzip it into `models/` (ignored by git), then:

```powershell
# validation first: must match the best validation numbers printed in the notebook
.\.venv\Scripts\cer.exe rerank runs/nfcorpus-hybrid/candidates-dev.jsonl --model models/minilm-l6-nfc-scifact --top-n 20
# test, once
.\.venv\Scripts\cer.exe rerank runs/nfcorpus-hybrid/candidates-test.jsonl --model models/minilm-l6-nfc-scifact --top-n 20
.\.venv\Scripts\cer.exe rerank runs/scifact-hybrid/candidates-test.jsonl --model models/minilm-l6-nfc-scifact --top-n 20
```

`top_n = 20` is the value chosen in step 3 for the off-the-shelf model; it is kept for the fine-tuned model
so that the comparison isolates the effect of fine-tuning.

The model directory contains `README.md` (model card), `training_info.json` and `train_log.jsonl`;
weights are not committed. Publishing them (e.g. Hugging Face Hub) is a separate decision.

## 5. GPU latency benchmark (optional)

Latency depends on the architecture, not on the weights: the fine-tuned C2 has the architecture of
`cross-encoder/ms-marco-MiniLM-L-6-v2`, so `scripts/bench_reranker_gpu.py` benchmarks the base models from the
Hub (MiniLM-L6 and bge-reranker-base, fp32 and fp16) on real first-stage passages.

```powershell
.\.venv\Scripts\python.exe scripts/bench_reranker_gpu.py --export runs/nfcorpus-hybrid/candidates-dev.jsonl --passages data/rerank-train/bench_passages.jsonl
Copy-Item scripts/bench_reranker_gpu.py data/rerank-train/
```

Add both files to the Kaggle dataset (New Version), then in a GPU notebook:

```python
import glob, os
D = os.path.dirname(glob.glob("/kaggle/input/**/bench_reranker_gpu.py", recursive=True)[0])
!python {D}/bench_reranker_gpu.py --passages {D}/bench_passages.jsonl --out /kaggle/working/gpu_latency.json
```

It prints a markdown table (tokenization on CPU, forward pass on GPU, total p50/p95 per query) and writes
`gpu_latency.json`; copy it to `reports/` to keep the measured numbers with the repository.
