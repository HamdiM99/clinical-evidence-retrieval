"""GPU latency benchmark of cross-encoders (self-contained: torch + transformers, for a GPU notebook).

Latency depends on the architecture, not on the weights: the fine-tuned C2 has the architecture of
`cross-encoder/ms-marco-MiniLM-L-6-v2`, so the base model is benchmarked directly from the Hub.

Input: bench_passages.jsonl ({"query", "passages": [...]} per line, real first-stage candidates), written by
`python scripts/bench_reranker_gpu.py --export runs/nfcorpus-hybrid/candidates-dev.jsonl`.
Per query it times (a) tokenization on CPU and (b) the forward pass on GPU (CUDA-synchronised), fp32 and fp16.

Usage (notebook):  python bench_reranker_gpu.py --passages bench_passages.jsonl --out gpu_latency.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]
SETTINGS = [(20, 20), (50, 50), (100, 100), (100, 32)]  # (passages per query, batch size)


def export(candidates: str, out: str, n_queries: int) -> None:
    with open(candidates, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= n_queries:
                break
            r = json.loads(line)
            ps = [f"{c['title']}. {c['text']}" if c["title"] else c["text"] for c in r["candidates"][:100]]
            g.write(json.dumps({"query": r["query"], "passages": ps}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None, help="candidates file to export bench passages from (local step)")
    ap.add_argument("--passages", default="bench_passages.jsonl")
    ap.add_argument("--queries", type=int, default=33)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", default="gpu_latency.json")
    args = ap.parse_args()
    if args.export:
        export(args.export, args.passages, args.queries)
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    data = [json.loads(x) for x in Path(args.passages).read_text(encoding="utf-8").splitlines() if x.strip()]
    results = []
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        for precision in ("fp32", "fp16"):
            for n, bs in SETTINGS:
                tok_ms, gpu_ms = [], []
                for i, r in enumerate(data):
                    ps = r["passages"][:n]
                    t0 = time.perf_counter()
                    encs = [tok([r["query"]] * len(ps[j: j + bs]), ps[j: j + bs], truncation="longest_first",
                                max_length=args.max_length, padding=True, return_tensors="pt")
                            for j in range(0, len(ps), bs)]
                    t1 = time.perf_counter()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
                        for e in encs:
                            model(**e.to(dev)).logits.float().cpu()
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    if i >= args.warmup:
                        tok_ms.append((t1 - t0) * 1e3)
                        gpu_ms.append((t3 - t2) * 1e3)
                tot = [a + b for a, b in zip(tok_ms, gpu_ms)]
                res = {"model": name, "precision": precision, "passages": n, "batch_size": bs, "queries": len(tot),
                       "tokenize_p50_ms": percentile(tok_ms, 50), "model_p50_ms": percentile(gpu_ms, 50),
                       "total_p50_ms": percentile(tot, 50), "total_p95_ms": percentile(tot, 95)}
                results.append(res)
                print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        del model
        torch.cuda.empty_cache()
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "max_length": args.max_length,
            "results": results}
    Path(args.out).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("| model | precision | passages | batch | tokenize p50 | model p50 | total p50 | total p95 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model'].split('/')[-1]} | {r['precision']} | {r['passages']} | {r['batch_size']} | "
              f"{r['tokenize_p50_ms']:.1f} | {r['model_p50_ms']:.1f} | {r['total_p50_ms']:.1f} | {r['total_p95_ms']:.1f} |")


if __name__ == "__main__":
    main()
