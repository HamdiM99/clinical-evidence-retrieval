"""Fine-tune a cross-encoder re-ranker on data/rerank-train (pointwise BCE on hard negatives).

Self-contained on purpose (torch + transformers only, no import from this repo), so the same file runs on
a GPU notebook (Kaggle, Colab) and locally. The saved directory is a standard Hugging Face sequence
classification model with one logit: `sentence_transformers.CrossEncoder(path)` and `cer rerank --model path`
load it directly.

Validation during training re-ranks the first-stage top-k of the validation queries (valid.jsonl) and
reports nDCG@10 per dataset; step 0 evaluates the base model, so the gain of fine-tuning is read directly.
The best checkpoint by mean validation nDCG@10 is kept.

Usage:
  python scripts/train_reranker.py --train data/rerank-train/train.jsonl --valid data/rerank-train/valid.jsonl \
      --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out models/minilm-l6-nfc-scifact
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit]


def ndcg_at_10(ranking: list[str], rels: dict[str, int]) -> float:
    """Same definition as cer.eval.metrics.ndcg_at_k (linear gains, log2 discount)."""
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:10]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def score_pairs(model, tok, pairs: list[tuple[str, str]], max_length: int, batch_size: int, device) -> list[float]:
    model.eval()
    out: list[float] = []
    for i in range(0, len(pairs), batch_size):
        q, p = zip(*pairs[i: i + batch_size])
        enc = tok(list(q), list(p), truncation="longest_first", max_length=max_length, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out += model(**enc).logits.view(-1).float().tolist()
    return out


def evaluate(model, tok, valid: list[dict], max_length: int, batch_size: int, device) -> dict[str, float]:
    pairs = [(v["query"], c["passage"]) for v in valid for c in v["candidates"]]
    scores = iter(score_pairs(model, tok, pairs, max_length, batch_size, device))
    per_ds: dict[str, list[float]] = {}
    for v in valid:
        ids = [c["doc_id"] for c in v["candidates"]]
        s = [next(scores) for _ in ids]
        order = [d for _, _, d in sorted(zip(s, range(len(ids)), ids), key=lambda x: (-x[0], x[1]))]
        per_ds.setdefault(v["dataset"], []).append(ndcg_at_10(order, v["qrels"]))
    res = {f"ndcg@10/{ds}": sum(x) / len(x) for ds, x in sorted(per_ds.items())}
    res["ndcg@10/mean"] = sum(res.values()) / len(res)
    return res


def first_stage(valid: list[dict]) -> dict[str, float]:
    per_ds: dict[str, list[float]] = {}
    for v in valid:
        per_ds.setdefault(v["dataset"], []).append(ndcg_at_10([c["doc_id"] for c in v["candidates"]], v["qrels"]))
    res = {f"ndcg@10/{ds}": sum(x) / len(x) for ds, x in sorted(per_ds.items())}
    res["ndcg@10/mean"] = sum(res.values()) / len(res)
    return res


def model_card(args, info: dict) -> str:
    rows = "\n".join(f"| {k} | {info['first_stage'][k]:.4f} | {info['base'][k]:.4f} | {info['best'][k]:.4f} |"
                     for k in info["best"])
    return f"""# {Path(args.out).name}

Cross-encoder re-ranker fine-tuned from `{args.base}` for medical evidence retrieval
(clinical-evidence-retrieval, milestone M2). Not a medical device; research use.

## Data
- Training pairs: `{args.train}` ({info['n_train']} pairs, {info['pos_share']:.1%} positives), built by
  `scripts/build_rerank_train.py` from BEIR NFCorpus and SciFact **train** splits, with hard negatives
  mined from the hybrid (BM25 + multilingual-e5-base, RRF) first stage. Test queries were never used.
- Validation: `{args.valid}` ({info['n_valid']} queries: NFCorpus dev, SciFact train[:200] held out),
  first-stage top {info['valid_top_n']} re-ranked.

## Training
Pointwise binary cross-entropy on one logit, AdamW, lr {args.lr}, linear schedule with
{args.warmup:.0%} warm-up, batch {args.batch_size}, {args.epochs} epoch(s), max length {args.max_length},
seed {args.seed}, fp16 autocast on GPU. Best checkpoint at step {info['best_step']} of {info['steps']}
by mean validation nDCG@10. Hardware: {info['hardware']}, {info['minutes']:.1f} min.

## Validation nDCG@10 (first-stage top {info['valid_top_n']} re-ranked)
| set | first stage | base model | fine-tuned |
|---|---|---|---|
{rows}

## Limitations
Trained on two small English benchmarks (nutrition questions, biomedical claims); not evaluated on French
queries or clinical guidelines. Labels come from benchmark judgments, not from clinicians.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/rerank-train/train.jsonl")
    ap.add_argument("--valid", default="data/rerank-train/valid.jsonl")
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="models/minilm-l6-nfc-scifact")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test)")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-valid", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({platform.processor() or 'unknown'})"
    train = read_jsonl(args.train, args.limit_train)
    valid = read_jsonl(args.valid, args.limit_valid)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = min(args.max_steps or 10**12, steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    log_path = out / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    def log(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    fs = first_stage(valid)
    base = evaluate(model, tok, valid, args.max_length, args.eval_batch_size, device)
    print(json.dumps({"step": 0, "first_stage": fs, "base": base}), flush=True)
    log({"step": 0, "first_stage": fs, **base})
    best, best_step = base, 0
    model.save_pretrained(out)
    tok.save_pretrained(out)

    t0, step = time.time(), 0
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            if step >= total:
                break
            batch = [train[j] for j in order[i: i + args.batch_size]]
            model.train()
            enc = tok([b["query"] for b in batch], [b["passage"] for b in batch], truncation="longest_first",
                      max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**enc).logits.view(-1)
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(json.dumps({"step": step, "epoch": epoch, "loss": round(loss.item(), 4),
                                  "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}), flush=True)
            if step % args.eval_every == 0 or step == total:
                res = evaluate(model, tok, valid, args.max_length, args.eval_batch_size, device)
                log({"step": step, "loss": loss.item(), **res})
                print(json.dumps({"step": step, **{k: round(v, 4) for k, v in res.items()}}), flush=True)
                if res["ndcg@10/mean"] > best["ndcg@10/mean"]:
                    best, best_step = res, step
                    model.save_pretrained(out)
                    tok.save_pretrained(out)

    info = {"first_stage": fs, "base": base, "best": best, "best_step": best_step, "steps": step,
            "n_train": len(train), "pos_share": sum(b["label"] for b in train) / max(len(train), 1),
            "n_valid": len(valid), "valid_top_n": len(valid[0]["candidates"]) if valid else 0,
            "hardware": hardware, "minutes": (time.time() - t0) / 60}
    (out / "training_info.json").write_text(json.dumps({**info, "args": vars(args)}, indent=2), encoding="utf-8")
    (out / "README.md").write_text(model_card(args, info), encoding="utf-8")
    print(json.dumps({"best_step": best_step, "best": best, "base": base}, indent=2))


if __name__ == "__main__":
    main()
