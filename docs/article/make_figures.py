"""Figures and numbers for the article, computed from the experiment artifacts (no hand-typed results).

Inputs: runs/*/metrics.json, runs/*/run.trec, runs/*/per_query.csv, runs/*-hybrid/candidates-test.jsonl,
runs/*/rerank-cache/*.jsonl, data/<ds>/qrels, data/scifact-official/claims_dev.jsonl (official SciFact stance
labels, see scripts/extract_scifact_labels.py), data/rerank-train/stats.json, models/C2-listwise-scifact/
train_log.jsonl, docs/article/data/kaggle_runs.json, runs/bench-rerank/results.json, reports/rerank-latency-gpu.md.
Outputs: docs/article/figures/*.png (300 dpi, drawn at print size: no text below 8 pt), figures/numbers.json and
docs/article/post-numbers.md.

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
from cer.data.beir import load_qrels, load_queries  # noqa: E402
from cer.eval.metrics import ndcg_at_k, per_query  # noqa: E402
from cer.eval.runner import read_trec  # noqa: E402
from cer.eval.significance import paired_permutation_test  # noqa: E402
from cer.rerank.candidates import read_candidates  # noqa: E402

OUT = ROOT / "docs/article/figures"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "runs"
FULL, HALF = 6.7, 3.2  # inches: text width of the A4 layout and half of it (figures are printed at 1:1)

INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
C = {"bm25": "#A3B1C6", "dense": "#5B7BA6", "hybrid": "#1E3A8A", "minilm": "#D97706", "c2": "#0F766E",
     "bge": "#7C3AED", "budget": "#B91C1C"}
DS = {"nfcorpus": "NFCorpus", "scifact": "SciFact"}
MINILM = "ms-marco-minilm-l-6-v2-L256"
C2 = "c2-listwise-scifact-L256"
RUN_NAME = {"bm25": "{ds}-bm25", "dense": "{ds}-dense", "hybrid": "{ds}-hybrid",
            "minilm": f"{{ds}}-hybrid-{MINILM}-top20", "c2": f"{{ds}}-hybrid-{C2}-top20"}
NEGATION = re.compile(r"\b(no|not|doesn't|don't|cannot|never|fails?)\b", re.I)  # same rule as the M2 report
FS = 8.5  # smallest font used in figures

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": FS, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.fontset": "dejavuserif",
})
NUM: dict = {}
SRC: dict = {}  # number name -> source, for post-numbers.md


def metrics(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("figure", name)


def per_query_ndcg(run_name: str, qrels: dict) -> dict[str, float]:
    return per_query(read_trec(RUNS / run_name / "run.trec"), qrels, ["ndcg@10"])["ndcg@10"]


def bootstrap_mean_ci(values: list[float], n: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired(a: dict, b: dict, qids=None) -> dict:
    qids = sorted(qids if qids is not None else a)
    a2, b2 = {q: a[q] for q in qids}, {q: b[q] for q in qids}
    t = paired_permutation_test(a2, b2)
    lo, hi = bootstrap_mean_ci([a2[q] - b2[q] for q in qids])
    wins = sum(a2[q] > b2[q] for q in qids)
    losses = sum(a2[q] < b2[q] for q in qids)
    return {"delta": t["delta"], "p": t["p_value"], "ci": [lo, hi], "n": len(qids), "wins": wins, "losses": losses,
            "ties": len(qids) - wins - losses}


# ----------------------------------------------------------------------------- numbers
def collect_numbers() -> None:
    for ds in DS:
        for key, tmpl in RUN_NAME.items():
            name = tmpl.format(ds=ds)
            m = metrics(name)
            NUM[f"{ds}/{key}"] = dict(m["metrics"]) | {"p50_ms": m["latency_ms"]["total"]["p50"],
                                                       "p95_ms": m["latency_ms"]["total"]["p95"]}
            SRC[f"{ds}/{key}"] = f"runs/{name}/metrics.json"
        if "first_stage" in metrics(RUN_NAME["c2"].format(ds=ds))["latency_ms"]:
            NUM[f"{ds}/first_stage_p50_ms"] = metrics(RUN_NAME["c2"].format(ds=ds))["latency_ms"]["first_stage"]["p50"]
    NUM["n_queries"] = {ds: metrics(f"{ds}-hybrid")["n_queries"] for ds in DS}
    NUM["chunks"] = {ds: metrics(f"{ds}-hybrid")["index"]["chunks"] for ds in DS}
    NUM["train_stats"] = json.loads((ROOT / "data/rerank-train/stats.json").read_text(encoding="utf-8"))
    SRC["train_stats"] = "data/rerank-train/stats.json"


def collect_comparisons() -> None:
    """Paired tests on per-query nDCG@10 (test): every comparison used in the text."""
    NUM["cmp"] = {}
    for ds in DS:
        qrels = load_qrels(ROOT / "data" / ds, "test")
        pq = {k: per_query_ndcg(t.format(ds=ds), qrels) for k, t in RUN_NAME.items()}
        for a, b in [("hybrid", "bm25"), ("hybrid", "dense"), ("minilm", "hybrid"), ("c2", "hybrid"),
                     ("c2", "minilm")]:
            NUM["cmp"][f"{ds}/{a}-{b}"] = paired(pq[a], pq[b])
            SRC[f"cmp {ds}/{a}-{b}"] = f"runs/{RUN_NAME[a].format(ds=ds)}/run.trec vs runs/{RUN_NAME[b].format(ds=ds)}/run.trec"
        if ds == "scifact":
            queries = load_queries(ROOT / "data/scifact")
            neg = [q for q in pq["hybrid"] if NEGATION.search(queries[q])]
            other = [q for q in pq["hybrid"] if q not in set(neg)]
            NUM["negation"] = {"rule": NEGATION.pattern,
                               "minilm-hybrid/negated": paired(pq["minilm"], pq["hybrid"], neg),
                               "minilm-hybrid/other": paired(pq["minilm"], pq["hybrid"], other),
                               "c2-hybrid/negated": paired(pq["c2"], pq["hybrid"], neg),
                               "c2-hybrid/other": paired(pq["c2"], pq["hybrid"], other)}
            SRC["negation"] = "per-query nDCG@10 from run.trec files, claims selected by regex on data/scifact/queries"


def collect_official_labels() -> None:
    """What sits at rank 1, according to the official SciFact stance annotations (BEIR test = SciFact dev)."""
    claims = {str(json.loads(x)["id"]): json.loads(x) for x in
              (ROOT / "data/scifact-official/claims_dev.jsonl").read_text(encoding="utf-8").splitlines() if x}
    qrels = load_qrels(ROOT / "data/scifact", "test")
    pairs = [(q, d) for q in qrels for d in qrels[q] if qrels[q][d] > 0]
    nei = sum(1 for q, d in pairs if not claims[q]["evidence"].get(d))
    only_nei = sum(1 for q in qrels if all(not claims[q]["evidence"].get(d) for d in qrels[q]))
    NUM["labels"] = {"relevant_pairs": len(pairs), "pairs_without_rationale": nei, "claims": len(qrels),
                     "claims_only_without_rationale": only_nei,
                     "pairs_support": sum(1 for q, d in pairs if (claims[q]["evidence"].get(d) or [{}])[0].get("label") == "SUPPORT"),
                     "pairs_contradict": sum(1 for q, d in pairs if (claims[q]["evidence"].get(d) or [{}])[0].get("label") == "CONTRADICT")}
    SRC["labels"] = "data/scifact-official/claims_dev.jsonl vs data/scifact/qrels/test.tsv"

    def category(q: str, d: str) -> str:
        ev = claims[q]["evidence"].get(d)
        if ev:
            return ev[0]["label"].lower()
        return "cited_no_rationale" if int(d) in claims[q]["cited_doc_ids"] else "not_cited"

    NUM["rank1"] = {}
    ev_qrels = {q: {d: 1 for d in claims[q]["evidence"]} for q in qrels if claims[q]["evidence"]}
    NUM["evidence_ndcg"] = {"claims": len(ev_qrels)}
    pq_ev = {}
    for k in ("hybrid", "minilm", "c2"):
        run = read_trec(RUNS / RUN_NAME[k].format(ds="scifact") / "run.trec")
        cats = {}
        for q in qrels:
            top = max(run[q], key=lambda d: (run[q][d], -int(d)))
            cats[q] = category(q, top)
        res = {}
        for cat in ("support", "contradict", "cited_no_rationale", "not_cited"):
            v = [1.0 if cats[q] == cat else 0.0 for q in qrels]
            res[cat] = {"share": sum(v) / len(v), "ci": list(bootstrap_mean_ci(v)), "count": int(sum(v))}
        v = [1.0 if cats[q] in ("support", "contradict") else 0.0 for q in qrels]
        res["evidence"] = {"share": sum(v) / len(v), "ci": list(bootstrap_mean_ci(v)), "count": int(sum(v))}
        NUM["rank1"][k] = res
        pq_ev[k] = per_query(run, ev_qrels, ["ndcg@10"])["ndcg@10"]
        NUM["evidence_ndcg"][k] = sum(pq_ev[k].values()) / len(pq_ev[k])
    NUM["evidence_ndcg"]["c2-hybrid"] = paired(pq_ev["c2"], pq_ev["hybrid"])
    NUM["evidence_ndcg"]["minilm-hybrid"] = paired(pq_ev["minilm"], pq_ev["hybrid"])

    # where the BEIR-qrels differences come from: claims with official evidence vs claims whose cited docs have none
    pq = {k: per_query_ndcg(RUN_NAME[k].format(ds="scifact"), qrels) for k in ("hybrid", "minilm", "c2")}
    with_ev = sorted(ev_qrels)
    without = sorted(q for q in qrels if q not in ev_qrels)
    NUM["label_split"] = {"with_evidence": len(with_ev), "without_evidence": len(without)}
    for a, b in [("minilm", "hybrid"), ("c2", "hybrid"), ("c2", "minilm")]:
        NUM["label_split"][f"{a}-{b}/with_evidence"] = paired(pq[a], pq[b], with_ev)
        NUM["label_split"][f"{a}-{b}/without_evidence"] = paired(pq[a], pq[b], without)
    train_claims = {str(json.loads(x)["id"]): json.loads(x) for x in
                    (ROOT / "data/scifact-official/claims_train.jsonl").read_text(encoding="utf-8").splitlines() if x}
    qt = load_qrels(ROOT / "data/scifact", "train")
    tp = [(q, d) for q in qt for d in qt[q] if qt[q][d] > 0]
    NUM["label_split"]["train_positives"] = len(tp)
    NUM["label_split"]["train_positives_without_rationale"] = sum(1 for q, d in tp
                                                                   if not train_claims[q]["evidence"].get(d))
    SRC["label_split"] = "per-query nDCG@10 (BEIR qrels) split by official SciFact annotation; train: claims_train.jsonl"
    SRC["rank1"] = "rank-1 document of runs/scifact-*/run.trec, labelled with data/scifact-official/claims_dev.jsonl"

    # the two worked examples (same claims as the social visual)
    titles = {}
    for line in (ROOT / "data/scifact/corpus.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        titles[d["_id"]] = d.get("title", "").replace("�", "–")
    NUM["examples"] = {}
    for q in ("185", "742", "743"):
        rel = [d for d in claims[q]["evidence"]]
        ranks = {}
        for k in ("hybrid", "minilm", "c2"):
            r = read_trec(RUNS / RUN_NAME[k].format(ds="scifact") / "run.trec")[q]
            order = sorted(r, key=lambda d: (-r[d], d))
            ranks[k] = min(order.index(d) + 1 for d in rel)
        NUM["examples"][q] = {"claim": claims[q]["claim"], "doc": rel[0], "title": titles[rel[0]],
                              "label": claims[q]["evidence"][rel[0]][0]["label"], "ranks": ranks}
    SRC["examples"] = "runs/scifact-*/run.trec + data/scifact-official/claims_dev.jsonl"


def collect_int8_agreement() -> None:
    """int8 vs fp32 C2 on every validation query, from the cached scores (no new run)."""
    from scipy.stats import kendalltau

    res = {}
    for ds, split in [("nfcorpus", "dev"), ("scifact", "train")]:
        cache = RUNS / f"{ds}-hybrid/rerank-cache"
        rd = lambda p: {json.loads(x)["qid"]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x}
        fp, q8 = rd(cache / f"{C2}-{split}.jsonl"), rd(cache / f"c2-listwise-scifact-onnx-int8-L256-{split}.jsonl")
        same, taus = 0, []
        for q in q8:
            a, b = fp[q], q8[q]
            n = min(len(b["doc_ids"]), 20)
            sa = dict(zip(a["doc_ids"][:n], a["scores"][:n]))
            sb = dict(zip(b["doc_ids"][:n], b["scores"][:n]))
            top = lambda s: set(sorted(s, key=lambda d: -s[d])[:10])
            same += top(sa) == top(sb)
            docs = list(sb)
            taus.append(kendalltau([sa[d] for d in docs], [sb[d] for d in docs]).statistic)
        res[ds] = {"queries": len(q8), "same_top10_set": same, "kendall_tau_mean": sum(taus) / len(taus)}
        res[ds]["ndcg_fp32"] = metrics(f"{ds}-hybrid-{C2}-top20-{split}")["metrics"]["ndcg@10"]
        res[ds]["ndcg_int8"] = metrics(f"{ds}-hybrid-c2-listwise-scifact-onnx-int8-L256-top20-{split}")["metrics"]["ndcg@10"]
    NUM["int8"] = res
    SRC["int8"] = "runs/*-hybrid/rerank-cache/c2-* (fp32 vs onnx-int8 scores, validation queries) + metrics.json"


# ----------------------------------------------------------------------------- 1. pipeline
def box(ax, x, y, w, h, text, fc, ec=None, color="white", fs=FS, weight="semibold", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc,
                                ec=ec or fc, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fs, weight=weight,
            linespacing=1.2)


def arrow(ax, a, b, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(FULL, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.98, "OFFLINE  ·  indexing", fontsize=FS, color=MUTED, weight="bold", va="top")
    box(ax, 0.00, 0.71, 0.14, 0.17, "Documents\n(BEIR)", "#E5E7EB", color=INK)
    box(ax, 0.18, 0.71, 0.14, 0.17, "Chunking\n250 words", "#CBD5E1", color=INK)
    box(ax, 0.36, 0.71, 0.17, 0.17, "Embeddings\nmultilingual-e5", "#CBD5E1", color=INK)
    box(ax, 0.57, 0.71, 0.21, 0.17, "Elasticsearch\nBM25 + int8 HNSW", C["hybrid"])
    for a, b in [((0.14, 0.795), (0.18, 0.795)), ((0.32, 0.795), (0.36, 0.795)), ((0.53, 0.795), (0.57, 0.795))]:
        arrow(ax, a, b)
    ax.text(0.0, 0.60, "ONLINE  ·  one query", fontsize=FS, color=MUTED, weight="bold", va="top")
    box(ax, 0.00, 0.23, 0.10, 0.25, "Query", "#E5E7EB", color=INK)
    box(ax, 0.13, 0.36, 0.15, 0.12, "BM25 top 100", C["bm25"], color=INK)
    box(ax, 0.13, 0.20, 0.15, 0.12, "kNN top 100", C["dense"])
    box(ax, 0.32, 0.23, 0.16, 0.25, "chunk → doc\n+ RRF fusion", C["hybrid"])
    box(ax, 0.52, 0.23, 0.19, 0.25, "Cross-encoder\nre-ranks top 20", C["c2"])
    box(ax, 0.75, 0.23, 0.13, 0.25, "Ranked\nevidence", "#E5E7EB", color=INK)
    arrow(ax, (0.10, 0.38), (0.13, 0.42))
    arrow(ax, (0.10, 0.32), (0.13, 0.26))
    arrow(ax, (0.28, 0.42), (0.32, 0.39))
    arrow(ax, (0.28, 0.26), (0.32, 0.30))
    arrow(ax, (0.48, 0.355), (0.52, 0.355))
    arrow(ax, (0.71, 0.355), (0.75, 0.355))
    arrow(ax, (0.675, 0.71), (0.30, 0.49), color=C["hybrid"], ls="--", lw=0.9)
    ax.add_patch(FancyBboxPatch((0.12, 0.0), 0.78, 0.14, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#F0FDFA", ec=C["c2"], lw=0.9, ls="--"))
    ax.text(0.51, 0.07, "Evaluation replay: the fused top 100 is exported once per query;\n"
                        "every re-ranker is scored offline on exactly the same candidates",
            ha="center", va="center", fontsize=FS, color=C["c2"])
    arrow(ax, (0.40, 0.23), (0.40, 0.14), color=C["c2"], ls="--", lw=0.9)
    ax.text(0.945, 0.355, "target\np95 < 300 ms", ha="center", va="center", fontsize=FS, color=C["budget"])
    save(fig, "f1_pipeline")


# ----------------------------------------------------------------------------- 2. baselines
def fig_baselines() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    systems = [("bm25", "BM25"), ("dense", "Dense (e5)"), ("hybrid", "Hybrid RRF")]
    for ax, metric, title in zip(axes, ["ndcg@10", "recall@100"], ["nDCG@10 (ranking quality)", "Recall@100 (coverage)"]):
        for i, (s, lab) in enumerate(systems):
            vals = [NUM[f"{ds}/{s}"][metric] for ds in DS]
            xs = [j + (i - 1) * 0.26 for j in range(2)]
            bars = ax.bar(xs, vals, 0.25, color=C[s], label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=FS - 0.5,
                        color=INK, rotation=90 if False else 0)
        ax.set_xticks(range(2), list(DS.values()))
        ax.set_ylim(0, 1.12)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.02), columnspacing=0.8, handlelength=1.2)
    save(fig, "f2_baselines")


# ----------------------------------------------------------------------------- 3. recall ceiling
def fig_recall_curve() -> None:
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    NUM["recall_at"] = {}
    for ds in DS:
        qrels = load_qrels(ROOT / "data" / ds, "test")
        recs = list(read_candidates(RUNS / f"{ds}-hybrid/candidates-test.jsonl"))
        curve = []
        for k in range(1, 101):
            vals = []
            for r in recs:
                rel = {d for d, v in qrels.get(r["qid"], {}).items() if v > 0}
                if rel:
                    vals.append(len(rel & {c["doc_id"] for c in r["candidates"][:k]}) / len(rel))
            curve.append(sum(vals) / len(vals))
        NUM["recall_at"][ds] = {k: curve[k - 1] for k in (10, 20, 50, 100)}
        ax.plot(range(1, 101), curve, color=C["hybrid"] if ds == "nfcorpus" else C["c2"], lw=1.6, label=DS[ds])
    ax.axvspan(1, 20, color=C["c2"], alpha=0.07)
    ax.text(11.5, 0.45, "re-ranked\nwindow\n(top 20)", ha="center", fontsize=FS, color=C["c2"])
    ax.set_xlabel("depth k of the first-stage list")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 100)
    ax.legend(loc="center right")
    ax.set_title("What the re-ranker can see", loc="left")
    save(fig, "f3_recall_ceiling")


# ----------------------------------------------------------------------------- 4. top_n tuning
def fig_topn() -> None:
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    fused = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))["first_stage"]
    NUM["topn"] = {}
    for ds, split in [("nfcorpus", "dev"), ("scifact", "train")]:
        ys = [metrics(f"{ds}-hybrid-{MINILM}-top{n}-{split}")["metrics"]["ndcg@10"] for n in (20, 50, 100)]
        NUM["topn"][ds] = {"fused": fused[ds], **{str(n): y for n, y in zip((20, 50, 100), ys)}}
        ax.plot([20, 50, 100], [y - fused[ds] for y in ys], marker="o", ms=4, lw=1.5,
                color=C["hybrid"] if ds == "nfcorpus" else C["c2"],
                label=f"{DS[ds]} ({'dev' if split == 'dev' else 'train[:200]'})")
    ax.axhline(0, color=INK, lw=0.8)
    ax.text(100, 0.0015, "no re-ranking", ha="right", va="bottom", fontsize=FS, color=MUTED)
    ax.set_xticks([20, 50, 100])
    ax.set_xlabel("top_n re-ranked (off-the-shelf MiniLM)")
    ax.set_ylabel("Δ nDCG@10 vs first stage")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5))
    ax.set_title("Deeper is not better", loc="left")
    save(fig, "f4_topn")


# ----------------------------------------------------------------------------- 5. hard negatives
def fig_pairs() -> None:
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.95, "First-stage ranking for one training query (illustrative)", fontsize=FS, color=MUTED,
            weight="bold")
    labels = ["+", "?", "?", "−", "+", "−", "−", "−", "+", "−", "−", "−"]
    for i, lab in enumerate(labels):
        x = 0.0 + i * 0.07
        if lab == "+":
            fc, txt, col = C["c2"], "pos", "white"
        elif i < 3:
            fc, txt, col = "#F3F4F6", "skip", MUTED
        else:
            fc, txt, col = "#FDE68A", "neg", INK
        box(ax, x, 0.50, 0.062, 0.30, f"#{i + 1}\n{txt}", fc, color=col, r=0.01)
    ax.text(0.85, 0.65, "…  rank 50", fontsize=FS, color=MUTED, va="center")
    ax.text(0.0, 0.30, "Unjudged documents at ranks 1–3 are never used as negatives (often relevant on NFCorpus).",
            fontsize=FS, color=MUTED, va="center")
    ax.text(0.0, 0.10, "Positives = judged relevant, also those the retriever missed.  Negatives = the hardest "
                       "non-relevant candidates, 4 per positive.", fontsize=FS, color=INK, va="center")
    save(fig, "f5_pairs")


# ----------------------------------------------------------------------------- 6. learning curves
def fig_learning() -> None:
    k = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))
    c2 = [json.loads(x) for x in (ROOT / k["runs"]["C2"]["log"]).read_text(encoding="utf-8").splitlines() if x]
    traj = {"1": k["runs"]["1"]["trajectory"], "A": k["runs"]["A"]["trajectory"],
            "C2": [[r["step"], r["ndcg@10/nfcorpus"], r["ndcg@10/scifact"]] for r in c2]}
    NUM["run1_scifact_min"] = min(p[2] for p in traj["1"])
    NUM["c2_validation_evaluations"] = sum(1 for r in c2 if r["step"] > 0)
    NUM["c2_best_step"], NUM["c2_steps"] = k["runs"]["C2"]["best_step"], k["runs"]["C2"]["steps"]
    SRC["run1_scifact_min"] = "docs/article/data/kaggle_runs.json (Kaggle training log)"
    SRC["c2_validation_evaluations"] = k["runs"]["C2"]["log"]
    style = {"1": (C["minilm"], "-", "Run 1 · BCE, lr 2e-5, both datasets"),
             "A": ("#92400E", "--", "Run A · BCE, lr 5e-6, both datasets"),
             "C2": (C["c2"], "-", "Run C2 · listwise, SciFact only")}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9))
    for j, (ax, ds) in enumerate(zip(axes, DS)):
        for run, pts in traj.items():
            col, ls, lab = style[run]
            ax.plot([p[0] for p in pts], [p[1 + j] for p in pts], marker="o", ms=2.6, lw=1.4, color=col, ls=ls,
                    label=lab)
        ax.axhline(k["base"][ds], color=INK, lw=0.9, ls=":", label="base model (step 0)")
        ax.axhline(k["first_stage"][ds], color=C["hybrid"], lw=0.9, ls="-.", label="no re-ranking")
        ax.set_title(f"{DS[ds]} validation", loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel("nDCG@10")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.8)
    save(fig, "f6_learning_curves")


# ----------------------------------------------------------------------------- 7. ablation
def fig_ablation() -> None:
    k = json.loads((ROOT / "docs/article/data/kaggle_runs.json").read_text(encoding="utf-8"))
    c2 = [json.loads(x) for x in (ROOT / k["runs"]["C2"]["log"]).read_text(encoding="utf-8").splitlines() if x]
    best_c2 = next(r for r in c2 if r["step"] == k["runs"]["C2"]["best_step"])
    best = {"1": k["base"], "A": k["base"], "B": k["runs"]["B"]["best"], "C1": k["runs"]["C1"]["best"],
            "C2": {"nfcorpus": best_c2["ndcg@10/nfcorpus"], "scifact": best_c2["ndcg@10/scifact"]}}
    labels = {"1": "1 · BCE\nlr 2e-5", "A": "A · BCE\nlr 5e-6", "B": "B · listwise\nboth", "C1": "C1 · listwise\nNFCorpus",
              "C2": "C2 · listwise\nSciFact"}
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    runs = list(labels)
    NUM["ablation"] = {}
    for i, ds in enumerate(DS):
        d = [best[r][ds] - k["base"][ds] for r in runs]
        NUM["ablation"][ds] = dict(zip(runs, d))
        xs = [j + (i - 0.5) * 0.36 for j in range(len(runs))]
        bars = ax.bar(xs, d, 0.34, color=C["hybrid"] if ds == "nfcorpus" else C["c2"], label=DS[ds])
        for j, (b, v, r) in enumerate(zip(bars, d, runs)):
            if k["runs"][r]["best_step"] == 0:
                if i == 0:
                    ax.text(j, 0.0009, "base kept", ha="center", fontsize=FS, color=MUTED)
                continue
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.0008, f"{v:+.3f}", ha="center", fontsize=FS,
                    color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(runs)), [labels[r] for r in runs])
    ax.set_ylabel("Δ nDCG@10 vs base\n(best validation checkpoint)")
    ax.set_ylim(-0.008, 0.024)
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "f7_ablation")


# ----------------------------------------------------------------------------- 8. test forest plot
def fig_forest() -> None:
    comps = [("minilm", "hybrid", "MiniLM − hybrid"), ("c2", "hybrid", "C2 − hybrid"), ("c2", "minilm", "C2 − MiniLM")]
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    y, yt, yl = 0, [], []
    for ds in DS:
        for a, b, lab in comps:
            t = NUM["cmp"][f"{ds}/{a}-{b}"]
            lo, hi = t["ci"]
            col = C["c2"] if a == "c2" else C["minilm"]
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round")
            ax.plot([t["delta"]], [y], "o", color=col, ms=5.5, mec="white", mew=0.8)
            ptxt = "p < 0.001" if t["p"] < 0.001 else f"p = {t['p']:.3f}"
            ax.text(0.068, y, f"{t['delta']:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {ptxt}", va="center", fontsize=FS,
                    color=INK)
            yt.append(y)
            yl.append(f"{DS[ds]} · {lab}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yt, yl, fontsize=FS)
    ax.set_xlim(-0.07, 0.16)
    ax.set_xticks([-0.06, -0.03, 0, 0.03, 0.06])
    ax.set_xlabel("Δ nDCG@10 on test  (dot: mean · bar: 95% bootstrap CI over queries)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "f8_forest")


# ----------------------------------------------------------------------------- 9. latency
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
    SRC["gpu"], SRC["cpu"] = "reports/rerank-latency-gpu.md", "runs/bench-rerank/results.json (reports/rerank-latency.md)"

    def best_cpu(backend: str) -> list[tuple[int, float]]:
        pts = {}
        for r in cpu["results"]:
            if r["backend"] == backend:
                pts[r["passages"]] = min(pts.get(r["passages"], 1e9), r["p95_ms"])
        return sorted(pts.items())

    fig = plt.figure(figsize=(FULL, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    series = [("PyTorch fp32 · laptop CPU", best_cpu("PyTorch fp32"), "#9CA3AF", "s", "--"),
              ("ONNX int8 · laptop CPU", best_cpu("ONNX Runtime int8 (dynamic)"), "#4B5563", "s", "--")]
    for model, prec, col, lab in [("ms-marco-MiniLM-L-6-v2", "fp32", "#5EEAD4", "MiniLM fp32 · T4"),
                                  ("ms-marco-MiniLM-L-6-v2", "fp16", C["c2"], "MiniLM fp16 · T4 (= C2)"),
                                  ("bge-reranker-base", "fp16", C["bge"], "bge-base fp16 · T4")]:
        pts = sorted((r["passages"], r["p95"]) for r in gpu if r["model"] == model and r["precision"] == prec
                     and r["batch"] == r["passages"])
        series.append((lab, pts, col, "o", "-"))
    for lab, pts, col, mk, ls in series:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.5, lw=1.5, color=col, ls=ls, label=lab)
    ax.axhline(100, color=C["budget"], lw=1)
    ax.text(104, 100, "100 ms\nbudget", fontsize=FS, color=C["budget"], ha="left", va="center")
    ax.set_yscale("log")
    ax.set_xlim(15, 125)
    ax.set_xticks([20, 50, 100])
    ax.set_xlabel("passages re-ranked per query")
    ax.set_ylabel("p95 per query (ms, log)")
    ax.set_title("Latency vs depth and hardware", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, columnspacing=1.0, handlelength=1.8)

    ax2 = fig.add_subplot(gs[1])
    sel = [r for r in gpu if r["precision"] == "fp16" and r["batch"] == r["passages"]]
    tok, mod = [r["tok"] for r in sel], [r["model_ms"] for r in sel]
    xs = [i + (0.6 if "bge" in r["model"] else 0) for i, r in enumerate(sel)]
    ax2.bar(xs, tok, 0.62, color="#F59E0B", label="tokenization (CPU)")
    ax2.bar(xs, mod, 0.62, bottom=tok, color=C["c2"], label="forward pass (GPU)")
    ax2.set_xticks(xs, [str(r["passages"]) for r in sel])
    for name, grp in (("MiniLM-L6", [x for x, r in zip(xs, sel) if "MiniLM" in r["model"]]),
                      ("bge-base", [x for x, r in zip(xs, sel) if "bge" in r["model"]])):
        ax2.text(sum(grp) / len(grp), -0.16, name, transform=ax2.get_xaxis_transform(), ha="center", va="top",
                 fontsize=FS, color=INK)
    ax2.set_ylabel("p50 ms per query (T4, fp16)")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Where the time goes", loc="left")
    save(fig, "f9_latency")


# ----------------------------------------------------------------------------- 10. quality vs latency
def fig_quality_latency() -> None:
    gpu20 = next(r for r in NUM["gpu"] if r["model"] == "ms-marco-MiniLM-L-6-v2" and r["precision"] == "fp16"
                 and r["passages"] == 20 and r["batch"] == 20)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8))
    NUM["c2_t4_estimate_p50_ms"] = {}
    names = {"bm25": "BM25", "dense": "Dense (e5)", "hybrid": "Hybrid RRF", "minilm": "+ MiniLM (laptop CPU)",
             "c2": "+ C2 (laptop CPU)"}
    for ax, ds in zip(axes, DS):
        ax.axvspan(1, 300, color="#ECFDF5", zorder=0, label="< 300 ms target" if ds == "nfcorpus" else None)
        for k in names:
            x, yv = NUM[f"{ds}/{k}"]["p50_ms"], NUM[f"{ds}/{k}"]["ndcg@10"]
            ax.plot(x, yv, "o", ms=6.5, color=C[k], zorder=3, label=names[k] if ds == "nfcorpus" else None)
        est = NUM[f"{ds}/first_stage_p50_ms"] + gpu20["p50"]
        NUM["c2_t4_estimate_p50_ms"][ds] = est
        yv = NUM[f"{ds}/c2"]["ndcg@10"]
        ax.plot(est, yv, "o", ms=7.5, mfc="white", mec=C["c2"], mew=1.6, zorder=4,
                label="+ C2, T4 fp16 (estimate: first stage + GPU re-ranking)" if ds == "nfcorpus" else None)
        ax.set_xscale("log")
        ax.set_xlim(40, 4000)
        lo = min(NUM[f"{ds}/{k}"]["ndcg@10"] for k in names)
        hi = max(NUM[f"{ds}/{k}"]["ndcg@10"] for k in names)
        ax.set_ylim(lo - 0.012, hi + 0.012)
        ax.set_title(DS[ds], loc="left")
        ax.set_xlabel("p50 latency per query (ms, log)")
        ax.set_ylabel("nDCG@10 (test)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncols=3, bbox_to_anchor=(0.5, -0.07), columnspacing=1.0, handlelength=1.2)
    save(fig, "f10_quality_latency")


# ----------------------------------------------------------------------------- 11. what is at rank 1
def fig_rank1() -> None:
    cats = [("support", "SUPPORTS the claim", C["c2"]), ("contradict", "CONTRADICTS the claim", C["hybrid"]),
            ("cited_no_rationale", "cited, but no evidence in it", "#F59E0B"), ("not_cited", "not cited", "#D1D5DB")]
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    systems = [("hybrid", "Hybrid RRF"), ("minilm", "+ MiniLM"), ("c2", "+ C2")]
    for i, (k, lab) in enumerate(systems[::-1]):
        left = 0.0
        for cat, name, col in cats:
            v = NUM["rank1"][k][cat]["share"] * 100
            ax.barh(i, v, left=left, color=col, height=0.6, label=name if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=FS,
                        color="white" if col not in ("#D1D5DB", "#F59E0B") else INK)
            left += v
    ax.set_yticks(range(3), [s[1] for s in systems[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the 300 SciFact test claims, by what the rank-1 document is (official labels)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncols=4, columnspacing=0.9, handlelength=1.2)
    save(fig, "f11_rank1")


# ----------------------------------------------------------------------------- 12. where the SciFact differences come from
def fig_label_split() -> None:
    ls = NUM["label_split"]
    groups = [("with_evidence", f"claims whose relevant document\nSUPPORTS or CONTRADICTS it (n = {ls['with_evidence']})"),
              ("without_evidence", f"claims whose 'relevant' document\nis cited but settles nothing (n = {ls['without_evidence']})")]
    comps = [("minilm-hybrid", "off-the-shelf MiniLM − hybrid", C["minilm"]), ("c2-hybrid", "C2 − hybrid", C["c2"]),
             ("c2-minilm", "C2 − off-the-shelf MiniLM", "#64748B")]
    fig, ax = plt.subplots(figsize=(FULL, 2.7))
    for gi, (g, glab) in enumerate(groups):
        for ci, (c, clab, col) in enumerate(comps):
            t = ls[f"{c}/{g}"]
            x = gi + (ci - 1) * 0.24
            ax.bar(x, t["delta"], 0.22, color=col, label=clab if gi == 0 else None)
            ax.errorbar(x, t["delta"], yerr=[[t["delta"] - t["ci"][0]], [t["ci"][1] - t["delta"]]], fmt="none",
                        ecolor=INK, elinewidth=0.9, capsize=2.5)
            ptxt = "p<0.001" if t["p"] < 0.001 else f"p={t['p']:.3f}"
            yv = t["ci"][1] + 0.006 if t["delta"] >= 0 else t["ci"][0] - 0.006
            ax.text(x, yv, f"{t['delta']:+.3f}\n{ptxt}", ha="center", va="bottom" if t["delta"] >= 0 else "top",
                    fontsize=FS - 0.5, color=INK, linespacing=1.1)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylabel("Δ nDCG@10 (BEIR qrels)")
    ax.set_ylim(-0.21, 0.12)
    ax.grid(axis="x", visible=False)
    fig.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.2)
    save(fig, "f12_label_split")


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


# ----------------------------------------------------------------------------- post numbers
def write_post_numbers() -> None:
    f3 = lambda x: f"{x:.3f}"
    sg = lambda x: f"{x:+.3f}"
    pv = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    cm = NUM["cmp"]

    def cmp_line(key: str, label: str) -> str:
        c = cm[key]
        return (f"| {label} | {sg(c['delta'])} | {pv(c['p'])} | [{sg(c['ci'][0])}, {sg(c['ci'][1])}] | "
                f"{c['wins']} / {c['losses']} / {c['ties']} | {c['n']} | `{SRC['cmp ' + key]}` |")

    gpu20 = next(r for r in NUM["gpu"] if r["model"] == "ms-marco-MiniLM-L-6-v2" and r["precision"] == "fp16"
                 and r["passages"] == 20 and r["batch"] == 20)
    cpu20 = next(r for r in NUM["cpu"] if r["backend"] == "ONNX Runtime int8 (dynamic)" and r["passages"] == 20)
    neg, ts, ex, lb = NUM["negation"], NUM["train_stats"], NUM["examples"], NUM["labels"]
    lines = [
        "# Numbers for the LinkedIn post",
        "",
        "Generated by `docs/article/make_figures.py` (same values and rounding as the article). Paired permutation "
        "tests (10,000 permutations) and 95% bootstrap CIs on per-query nDCG@10, test sets.",
        "",
        "## Comparisons (nDCG@10, test)",
        "",
        "| comparison | Δ | p | 95% CI | wins / losses / ties | queries | source |",
        "|---|---|---|---|---|---|---|",
        cmp_line("nfcorpus/hybrid-bm25", "NFCorpus · hybrid vs BM25"),
        cmp_line("scifact/hybrid-bm25", "SciFact · hybrid vs BM25"),
        cmp_line("nfcorpus/minilm-hybrid", "NFCorpus · MiniLM vs hybrid"),
        cmp_line("scifact/minilm-hybrid", "SciFact · MiniLM vs hybrid"),
        cmp_line("nfcorpus/c2-minilm", "NFCorpus · C2 vs MiniLM"),
        cmp_line("scifact/c2-minilm", "SciFact · C2 vs MiniLM"),
        cmp_line("nfcorpus/c2-hybrid", "NFCorpus · C2 vs hybrid"),
        cmp_line("scifact/c2-hybrid", "SciFact · C2 vs hybrid"),
        "",
        "## Negated claims (SciFact, MiniLM vs hybrid)",
        "",
        f"Rule: regex `{neg['rule']}` on the claim text. Source: {SRC['negation']}.",
        "",
        f"- negated claims: Δ {sg(neg['minilm-hybrid/negated']['delta'])} (n = {neg['minilm-hybrid/negated']['n']}, "
        f"p = {pv(neg['minilm-hybrid/negated']['p'])})",
        f"- other claims: Δ {sg(neg['minilm-hybrid/other']['delta'])} (n = {neg['minilm-hybrid/other']['n']}, "
        f"p = {pv(neg['minilm-hybrid/other']['p'])})",
        "",
        "## Worked examples (SciFact test, official labels)",
        "",
    ]
    for q, e in ex.items():
        r = e["ranks"]
        lines.append(f"- claim {q}: \"{e['claim']}\" · doc {e['doc']} · {e['label']} · rank hybrid {r['hybrid']}, "
                     f"MiniLM {r['minilm']}, C2 {r['c2']}")
    lines += [
        f"\nSource: {SRC['examples']}.",
        "",
        "## Latency, 20 passages per query",
        "",
        f"- laptop CPU, C2 ONNX int8: p50 {cpu20['p50_ms']:.0f} ms, p95 {cpu20['p95_ms']:.0f} ms ({NUM['cpu_hw']}) · "
        f"source: {SRC['cpu']}",
        f"- Tesla T4, MiniLM-L6 (= C2 architecture) fp16: p50 {gpu20['p50']:.1f} ms, p95 {gpu20['p95']:.1f} ms · "
        f"source: {SRC['gpu']}",
        f"- first stage (hybrid, laptop, p50): NFCorpus {NUM['nfcorpus/hybrid']['p50_ms']:.0f} ms, SciFact "
        f"{NUM['scifact/hybrid']['p50_ms']:.0f} ms · source: runs/*-hybrid/metrics.json",
        "",
        "## Data",
        "",
        f"- NFCorpus training pairs: {ts['datasets']['nfcorpus']['pairs']:,} · SciFact training claims: "
        f"{ts['datasets']['scifact']['queries']} ({ts['datasets']['scifact']['pairs']:,} pairs) · source: {SRC['train_stats']}",
        f"- passages indexed: {NUM['chunks']['nfcorpus'] + NUM['chunks']['scifact']:,} chunks "
        f"({NUM['chunks']['nfcorpus']:,} NFCorpus + {NUM['chunks']['scifact']:,} SciFact) · source: runs/*-hybrid/metrics.json",
        f"- BEIR SciFact test: {lb['pairs_without_rationale']} of {lb['relevant_pairs']} relevant pairs have no "
        f"rationale in the official annotation · source: {SRC['labels']}",
    ]
    (ROOT / "docs/article/post-numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("post-numbers.md written")


def main() -> None:
    collect_numbers()
    collect_comparisons()
    collect_official_labels()
    collect_int8_agreement()
    fig_pipeline()
    fig_baselines()
    fig_recall_curve()
    fig_topn()
    fig_pairs()
    fig_learning()
    fig_ablation()
    fig_forest()
    fig_latency()
    fig_quality_latency()
    fig_rank1()
    fig_label_split()
    equations()
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=2), encoding="utf-8")
    write_post_numbers()
    print("numbers.json written")


if __name__ == "__main__":
    main()
