"""Figures and numbers for the article, computed from the experiment artifacts (no hand-typed results).

Inputs: runs/*/metrics.json, runs/*/run.trec, runs/*-hybrid/candidates-test.jsonl, data/<ds>/qrels,
models/C2-listwise-scifact/train_log.jsonl, docs/article/data/kaggle_runs.json, runs/bench-rerank/results.json,
reports/rerank-latency-gpu.md.
Outputs: docs/article/figures/*.png (300 dpi) and docs/article/figures/numbers.json.

Usage: python docs/article/make_figures.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cer.data.beir import load_qrels  # noqa: E402
from cer.eval.metrics import per_query  # noqa: E402
from cer.eval.runner import read_trec  # noqa: E402
from cer.eval.significance import paired_permutation_test  # noqa: E402
from cer.rerank.candidates import read_candidates  # noqa: E402

OUT = ROOT / "docs/article/figures"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "runs"

INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
C = {"bm25": "#A3B1C6", "dense": "#5B7BA6", "hybrid": "#1E3A8A", "minilm": "#D97706", "c2": "#0F766E",
     "bge": "#7C3AED", "budget": "#B91C1C", "accent": "#0F766E"}
DS = {"nfcorpus": "NFCorpus", "scifact": "SciFact"}
MINILM = "ms-marco-minilm-l-6-v2-L256"
C2 = "c2-listwise-scifact-L256"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "mathtext.fontset": "dejavuserif",
})
NUM: dict = {}


def metrics(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("figure", name)


# ----------------------------------------------------------------------------- numbers
def collect_numbers() -> None:
    for ds in DS:
        for sys_ in ("bm25", "dense", "hybrid", f"hybrid-{MINILM}-top20", f"hybrid-{C2}-top20"):
            m = metrics(f"{ds}-{sys_}")
            NUM[f"{ds}/{sys_.replace(MINILM, 'minilm').replace(C2, 'c2')}"] = {
                k: round(v, 4) for k, v in m["metrics"].items()} | {"p50_ms": round(m["latency_ms"]["total"]["p50"], 1)}
    NUM["n_queries"] = {ds: metrics(f"{ds}-hybrid")["n_queries"] for ds in DS}
    NUM["chunks"] = {ds: metrics(f"{ds}-hybrid")["index"]["chunks"] for ds in DS}
    NUM["train_stats"] = json.loads((ROOT / "data/rerank-train/stats.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- 1. pipeline
def box(ax, x, y, w, h, text, fc, ec=None, color="white", fs=8, weight="semibold", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc,
                                ec=ec or fc, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fs, weight=weight,
            linespacing=1.25)


def arrow(ax, a, b, color=MUTED, style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.97, "OFFLINE  ·  indexing", fontsize=7.5, color=MUTED, weight="bold", va="top")
    box(ax, 0.00, 0.70, 0.13, 0.17, "Documents\n(BEIR)", "#E5E7EB", color=INK)
    box(ax, 0.17, 0.70, 0.13, 0.17, "Chunking\n250 words", "#CBD5E1", color=INK)
    box(ax, 0.34, 0.70, 0.15, 0.17, "Embeddings\nmultilingual-e5", "#CBD5E1", color=INK)
    box(ax, 0.53, 0.70, 0.17, 0.17, "Elasticsearch\nBM25 + int8 HNSW", C["hybrid"])
    for a, b in [((0.13, 0.785), (0.17, 0.785)), ((0.30, 0.785), (0.34, 0.785)), ((0.49, 0.785), (0.53, 0.785))]:
        arrow(ax, a, b)
    ax.text(0.0, 0.58, "ONLINE  ·  one query", fontsize=7.5, color=MUTED, weight="bold", va="top")
    box(ax, 0.00, 0.22, 0.11, 0.24, "Query", "#E5E7EB", color=INK)
    box(ax, 0.15, 0.34, 0.14, 0.12, "BM25 top 100", C["bm25"], color=INK, fs=7.5)
    box(ax, 0.15, 0.18, 0.14, 0.12, "kNN top 100", C["dense"], fs=7.5)
    box(ax, 0.33, 0.22, 0.15, 0.24, "chunk → doc\n(max) + RRF\nfusion", C["hybrid"])
    box(ax, 0.53, 0.22, 0.17, 0.24, "Cross-encoder\nre-ranks top 20\n(this article)", C["c2"])
    box(ax, 0.75, 0.22, 0.12, 0.24, "Ranked\nevidence +\npassages", "#E5E7EB", color=INK)
    arrow(ax, (0.11, 0.37), (0.15, 0.40))
    arrow(ax, (0.11, 0.31), (0.15, 0.24))
    arrow(ax, (0.29, 0.40), (0.33, 0.37))
    arrow(ax, (0.29, 0.24), (0.33, 0.29))
    arrow(ax, (0.48, 0.34), (0.53, 0.34))
    arrow(ax, (0.70, 0.34), (0.75, 0.34))
    arrow(ax, (0.615, 0.70), (0.30, 0.47), color=C["hybrid"], ls="--", lw=0.9)
    ax.add_patch(FancyBboxPatch((0.22, 0.015), 0.58, 0.13, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#F0FDFA", ec=C["c2"], lw=0.9, ls="--"))
    ax.text(0.51, 0.08, "Evaluation replay: the fused top 100 is exported once per query,\n"
                          "every re-ranker is scored offline on exactly the same candidates",
            ha="center", va="center", fontsize=7, color=C["c2"])
    arrow(ax, (0.405, 0.22), (0.405, 0.14), color=C["c2"], ls="--", lw=0.9)
    ax.text(0.93, 0.34, "budget\np95 < 300 ms\nend to end", ha="center", va="center", fontsize=7, color=C["budget"])
    save(fig, "f1_pipeline")


# ----------------------------------------------------------------------------- 2. baselines
def fig_baselines() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    systems = [("bm25", "BM25"), ("dense", "Dense (e5)"), ("hybrid", "Hybrid RRF")]
    for ax, metric, title in zip(axes, ["ndcg@10", "recall@100"], ["nDCG@10 (ranking quality)", "Recall@100 (coverage)"]):
        for i, (s, lab) in enumerate(systems):
            vals = [NUM[f"{ds}/{s}"][metric] for ds in DS]
            xs = [j + (i - 1) * 0.26 for j in range(2)]
            bars = ax.bar(xs, vals, 0.25, color=C[s], label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(2), list(DS.values()))
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
    save(fig, "f2_baselines")


# ----------------------------------------------------------------------------- 3. recall ceiling
def fig_recall_curve() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    NUM["recall_at"] = {}
    for ds in DS:
        qrels = load_qrels(ROOT / "data" / ds, "test")
        recs = list(read_candidates(RUNS / f"{ds}-hybrid/candidates-test.jsonl"))
        ks = list(range(1, 101))
        curve = []
        for k in ks:
            vals = []
            for r in recs:
                rel = {d for d, v in qrels.get(r["qid"], {}).items() if v > 0}
                if rel:
                    vals.append(len(rel & {c["doc_id"] for c in r["candidates"][:k]}) / len(rel))
            curve.append(sum(vals) / len(vals))
        NUM["recall_at"][ds] = {k: round(curve[k - 1], 4) for k in (10, 20, 50, 100)}
        ax.plot(ks, curve, color=C["hybrid"] if ds == "nfcorpus" else C["c2"], lw=1.6, label=DS[ds])
    ax.axvspan(1, 20, color=C["c2"], alpha=0.07)
    ax.text(10.5, 0.42, "re-ranked\nwindow\n(top 20)", ha="center", fontsize=6.5, color=C["c2"])
    ax.set_xlabel("depth k of the first-stage list")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 100)
    ax.legend(loc="center right")
    ax.set_title("What the re-ranker can see", loc="left")
    save(fig, "f3_recall_ceiling")


# ----------------------------------------------------------------------------- 4. top_n tuning
def fig_topn() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fused = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))["first_stage"]
    NUM["topn"] = {}
    for ds, split in [("nfcorpus", "dev"), ("scifact", "train")]:
        ys = [metrics(f"{ds}-hybrid-{MINILM}-top{n}-{split}")["metrics"]["ndcg@10"] for n in (20, 50, 100)]
        NUM["topn"][ds] = {"fused": round(fused[ds], 4), **{str(n): round(y, 4) for n, y in zip((20, 50, 100), ys)}}
        delta = [y - fused[ds] for y in ys]
        ax.plot([20, 50, 100], delta, marker="o", ms=4, lw=1.5, color=C["hybrid"] if ds == "nfcorpus" else C["c2"],
                label=f"{DS[ds]} ({'dev' if split == 'dev' else 'train[:200]'})")
    ax.axhline(0, color=INK, lw=0.8)
    ax.text(100, 0.0015, "first stage (no re-ranking)", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks([20, 50, 100])
    ax.set_xlabel("top_n re-ranked by off-the-shelf MiniLM")
    ax.set_ylabel("Δ nDCG@10 vs first stage")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5))
    ax.set_title("Deeper is not better", loc="left")
    save(fig, "f4_topn")


# ----------------------------------------------------------------------------- 5. hard negatives
def fig_pairs() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.92, "First-stage ranking for one training query", fontsize=7.5, color=MUTED, weight="bold")
    labels = ["+", "?", "?", "−", "+", "−", "−", "−", "+", "−", "−", "−"]
    for i, lab in enumerate(labels):
        x = 0.02 + i * 0.066
        if i < 3:
            fc, txt, col = "#F3F4F6", "skip", MUTED
        elif lab == "+":
            fc, txt, col = C["c2"], "pos", "white"
        else:
            fc, txt, col = "#FDE68A", "neg", INK
        if lab == "+" and i < 3:
            fc, txt, col = C["c2"], "pos", "white"
        box(ax, x, 0.42, 0.058, 0.32, f"#{i + 1}\n{txt}", fc, color=col, fs=6.5, r=0.01)
    ax.text(0.83, 0.58, "…  rank 50", fontsize=7, color=MUTED, va="center")
    ax.annotate("", xy=(0.215, 0.32), xytext=(0.02, 0.32), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(0.02, 0.16, "unjudged documents at ranks 1–3 are never\nused as negatives (often relevant on NFCorpus)",
            ha="left", fontsize=6.5, color=MUTED)
    ax.text(0.40, 0.16, "positives = judged relevant (also those the retriever missed);  negatives = the hardest\n"
                        "non-relevant candidates, 4 per positive, taken from the top 50",
            ha="left", fontsize=6.5, color=INK)
    save(fig, "f5_pairs")


# ----------------------------------------------------------------------------- 6. learning curves
def fig_learning() -> None:
    k = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))
    c2 = [json.loads(x) for x in (ROOT / k["runs"]["C2"]["log"]).read_text(encoding="utf-8").splitlines() if x]
    traj = {"1": k["runs"]["1"]["trajectory"], "A": k["runs"]["A"]["trajectory"],
            "C2": [[r["step"], r["ndcg@10/nfcorpus"], r["ndcg@10/scifact"]] for r in c2]}
    style = {"1": (C["minilm"], "Run 1 · BCE, lr 2e-5, both datasets"),
             "A": ("#B45309", "Run A · BCE, lr 5e-6, both datasets"),
             "C2": (C["c2"], "Run C2 · listwise, SciFact only")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=False)
    for j, (ax, ds) in enumerate(zip(axes, DS)):
        for run, pts in traj.items():
            col, lab = style[run]
            ax.plot([p[0] for p in pts], [p[1 + j] for p in pts], marker="o", ms=2.6, lw=1.4, color=col, label=lab,
                    ls="--" if run == "A" else "-")
        ax.axhline(k["base"][ds], color=INK, lw=0.9, ls=":")
        ax.axhline(k["first_stage"][ds], color=C["hybrid"], lw=0.9, ls="-.")
        ax.text(2935, k["base"][ds], " base MiniLM", fontsize=6.5, va="bottom", ha="right", color=INK)
        ax.text(2935, k["first_stage"][ds], " no re-ranking", fontsize=6.5, va="top" if ds == "nfcorpus" else "bottom",
                ha="right", color=C["hybrid"])
        ax.set_title(f"{DS[ds]} validation", loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel("nDCG@10")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncols=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    save(fig, "f6_learning_curves")


# ----------------------------------------------------------------------------- 7. ablation
def fig_ablation() -> None:
    k = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))
    c2 = [json.loads(x) for x in (ROOT / k["runs"]["C2"]["log"]).read_text(encoding="utf-8").splitlines() if x]
    best_c2 = next(r for r in c2 if r["step"] == k["runs"]["C2"]["best_step"])
    best = {"1": k["base"], "A": k["base"], "B": k["runs"]["B"]["best"], "C1": k["runs"]["C1"]["best"],
            "C2": {"nfcorpus": best_c2["ndcg@10/nfcorpus"], "scifact": best_c2["ndcg@10/scifact"]}}
    labels = {"1": "1\nBCE\n2e-5", "A": "A\nBCE\n5e-6", "B": "B\nlistwise\nboth", "C1": "C1\nlistwise\nNFCorpus",
              "C2": "C2\nlistwise\nSciFact"}
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    runs = list(labels)
    NUM["ablation"] = {}
    for i, ds in enumerate(DS):
        d = [best[r][ds] - k["base"][ds] for r in runs]
        NUM["ablation"][ds] = {r: round(v, 4) for r, v in zip(runs, d)}
        xs = [j + (i - 0.5) * 0.36 for j in range(len(runs))]
        bars = ax.bar(xs, d, 0.34, color=C["hybrid"] if ds == "nfcorpus" else C["c2"], label=DS[ds])
        for j, (b, v, r) in enumerate(zip(bars, d, runs)):
            if k["runs"][r]["best_step"] == 0:
                if i == 0:
                    ax.text(j, 0.0008, "no checkpoint beat\nthe base model", ha="center", fontsize=6.2, color=MUTED)
                continue
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.0008, f"{v:+.3f}", ha="center", fontsize=6.2,
                    color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(runs)), [labels[r] for r in runs], fontsize=7)
    ax.set_ylabel("Δ nDCG@10 vs base model\n(best validation checkpoint)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Only clean in-domain labels + a ranking loss move the needle", loc="left")
    save(fig, "f7_ablation")


# ----------------------------------------------------------------------------- 8. test forest plot
def per_query_ndcg(run_name: str, ds: str) -> dict[str, float]:
    qrels = load_qrels(ROOT / "data" / ds, "test")
    return per_query(read_trec(RUNS / run_name / "run.trec"), qrels, ["ndcg@10"])["ndcg@10"]


def bootstrap_ci(a: dict, b: dict, n: int = 10000, seed: int = 7) -> tuple[float, float]:
    qs = sorted(a)
    d = [a[q] - b[q] for q in qs]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def fig_forest() -> None:
    comps = [("minilm", "hybrid", "Off-the-shelf MiniLM − hybrid"), ("c2", "hybrid", "C2 (fine-tuned) − hybrid"),
             ("c2", "minilm", "C2 − off-the-shelf MiniLM")]
    names = {"hybrid": "{ds}-hybrid", "minilm": f"{{ds}}-hybrid-{MINILM}-top20", "c2": f"{{ds}}-hybrid-{C2}-top20"}
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    NUM["forest"] = {}
    y = 0
    yt, yl = [], []
    for ds in DS:
        pq = {k: per_query_ndcg(v.format(ds=ds), ds) for k, v in names.items()}
        for a, b, lab in comps:
            t = paired_permutation_test(pq[a], pq[b])
            lo, hi = bootstrap_ci(pq[a], pq[b])
            NUM["forest"][f"{ds}/{a}-{b}"] = {"delta": t["delta"], "p": t["p_value"], "ci": [lo, hi],
                                             "n": t["n_queries"]}  # full precision: rounded once, when printed
            col = C["c2"] if a == "c2" else C["minilm"]
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.plot([t["delta"]], [y], "o", color=col, ms=5, mec="white", mew=0.8)
            p = t["p_value"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(0.071, y, f"{t['delta']:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {ptxt}", va="center", fontsize=6.8,
                    color=INK)
            yt.append(y)
            yl.append(f"{DS[ds]} · {lab}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yt, yl, fontsize=7)
    ax.set_xlim(-0.075, 0.15)
    ax.set_xticks([-0.06, -0.03, 0, 0.03, 0.06])
    ax.set_xlabel("Δ nDCG@10 on test (point: mean; bar: 95% bootstrap CI over queries)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "f8_forest")


# ----------------------------------------------------------------------------- 9-10. latency
def gpu_rows() -> list[dict]:
    rows = []
    for line in (ROOT / "reports/rerank-latency-gpu.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+) \| (fp\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            rows.append({"model": m[1], "precision": m[2], "passages": int(m[3]), "batch": int(m[4]),
                         "tok": float(m[5]), "model_ms": float(m[6]), "p50": float(m[7]), "p95": float(m[8])})
    return rows


def fig_latency() -> None:
    cpu = json.loads((RUNS / "bench-rerank/results.json").read_text(encoding="utf-8"))
    gpu = gpu_rows()
    NUM["gpu"] = gpu
    NUM["cpu"] = [{k: r[k] for k in ("backend", "passages", "batch_size", "p50_ms", "p95_ms")} for r in cpu["results"]]
    NUM["cpu_hw"], NUM["cpu_versions"] = cpu["hardware"], cpu["versions"]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))

    def best_cpu(backend: str) -> list[tuple[int, float]]:
        pts = {}
        for r in cpu["results"]:
            if r["backend"] == backend:
                pts[r["passages"]] = min(pts.get(r["passages"], 1e9), r["p95_ms"])
        return sorted(pts.items())

    series = [("PyTorch fp32 · laptop CPU", best_cpu("PyTorch fp32"), "#9CA3AF", "s", "--"),
              ("ONNX int8 · laptop CPU", best_cpu("ONNX Runtime int8 (dynamic)"), "#4B5563", "s", "--")]
    for model, prec, col, lab in [("ms-marco-MiniLM-L-6-v2", "fp32", "#5EEAD4", "MiniLM fp32 · T4"),
                                  ("ms-marco-MiniLM-L-6-v2", "fp16", C["c2"], "MiniLM fp16 · T4  (= C2)"),
                                  ("bge-reranker-base", "fp16", C["bge"], "bge-reranker-base fp16 · T4")]:
        pts = sorted((r["passages"], r["p95"]) for r in gpu if r["model"] == model and r["precision"] == prec
                     and r["batch"] == r["passages"])
        series.append((lab, pts, col, "o", "-"))
    for lab, pts, col, mk, ls in series:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.5, lw=1.5, color=col, ls=ls, label=lab)
    ax.axhline(100, color=C["budget"], lw=1)
    ax.text(19, 108, "≈ 100 ms re-ranking budget", fontsize=6.8, color=C["budget"], ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks([20, 50, 100])
    ax.set_xlabel("passages re-ranked per query")
    ax.set_ylabel("p95 latency per query (ms, log)")
    ax.legend(loc="upper left", fontsize=6.4, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Latency vs depth, hardware and precision", loc="left")
    save(fig, "f9_latency")

    fig, ax = plt.subplots(figsize=(2.75, 2.8))
    sel = [r for r in gpu if r["precision"] == "fp16" and r["batch"] == r["passages"]]
    labs, tok, mod = [], [], []
    for r in sel:
        labs.append(f"{'MiniLM' if 'MiniLM' in r['model'] else 'bge'}\n{r['passages']}")
        tok.append(r["tok"])
        mod.append(r["model_ms"])
    xs = range(len(sel))
    ax.bar(xs, tok, 0.6, color="#F59E0B", label="tokenization (CPU)")
    ax.bar(xs, mod, 0.6, bottom=tok, color=C["c2"], label="forward pass (GPU)")
    ax.set_xticks(list(xs), labs, fontsize=6.5)
    ax.set_ylabel("p50 ms per query (T4, fp16)")
    ax.legend(loc="upper left", fontsize=6.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Where the time goes", loc="left")
    save(fig, "f10_breakdown")


# ----------------------------------------------------------------------------- equations
def equation(name: str, tex: str, w: float = 6.0, h: float = 0.45, fs: int = 12) -> None:
    fig = plt.figure(figsize=(w, h))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=fs, color=INK)
    fig.savefig(OUT / f"{name}.png", transparent=True)
    plt.close(fig)
    print("equation", name)


def equations() -> None:
    equation("eq_rrf", r"$\mathrm{score}(d)=\sum_{s\,\in\,\{\mathrm{BM25},\,\mathrm{kNN}\}}\ \frac{1}{k+\mathrm{rank}_s(d)},\quad k=60$")
    equation("eq_ndcg", r"$\mathrm{nDCG@10}=\frac{1}{\mathrm{IDCG}}\sum_{i=1}^{10}\frac{\mathrm{rel}_i}{\log_2(i+1)}$")
    equation("eq_bce", r"$\mathcal{L}_{\mathrm{pointwise}}=-\,y\,\log\sigma(s_{q,d})-(1-y)\log\left(1-\sigma(s_{q,d})\right)$")
    equation("eq_listwise", r"$\mathcal{L}_{\mathrm{listwise}}=-\log\frac{\exp(s_{q,d^{+}})}{\exp(s_{q,d^{+}})+\sum_{j=1}^{7}\exp(s_{q,d_j^{-}})}$",
             h=0.6)


def main() -> None:
    collect_numbers()
    fig_pipeline()
    fig_baselines()
    fig_recall_curve()
    fig_topn()
    fig_pairs()
    fig_learning()
    fig_ablation()
    fig_forest()
    fig_latency()
    equations()
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=2), encoding="utf-8")
    print("numbers.json written")


if __name__ == "__main__":
    main()
