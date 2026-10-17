"""Command line: cer download | ingest | search | eval | compare | report."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from cer.config import ExperimentConfig

app = typer.Typer(add_completion=False, help="Clinical Evidence Retrieval")


@app.command()
def download(dataset: str = typer.Argument(..., help="BEIR name, e.g. nfcorpus, scifact, trec-covid"),
             data_dir: str = "data"):
    """Download a BEIR dataset."""
    from cer.data import beir
    path = beir.download(dataset, data_dir)
    typer.echo(f"ok: {path}")


@app.command()
def ingest(config: Path, swap: bool = typer.Option(True, help="point the alias to the new index")):
    """Build a new versioned index for the config's dataset (Elasticsearch backend)."""
    from cer.data import beir
    from cer.embed import build_embedder
    from cer.index import build_backend
    from cer.ingest.pipeline import ingest as run_ingest

    cfg = ExperimentConfig.from_yaml(config)
    embedder = build_embedder(cfg.embedder)
    backend = build_backend(cfg.backend)
    rep = run_ingest(beir.iter_corpus(cfg.dataset_dir), backend, embedder, version=cfg.backend.index_version,
                     max_words=cfg.chunking.max_words, overlap_words=cfg.chunking.overlap_words,
                     batch_size=cfg.embedder.batch_size, swap_alias=swap)
    typer.echo(json.dumps({**rep.__dict__, "chunks_per_second": round(rep.chunks_per_second, 1)}, indent=2))


@app.command()
def search(config: Path, query: str, k: int = 10):
    """Run one query and print the top documents with their supporting passage."""
    from cer.embed import build_embedder
    from cer.index import build_backend
    from cer.retrieve.retriever import Retriever

    cfg = ExperimentConfig.from_yaml(config)
    if cfg.backend.type == "memory":
        raise typer.BadParameter("interactive search needs a persistent backend (elasticsearch)")
    emb = build_embedder(cfg.embedder) if cfg.retrieval.mode != "bm25" else None
    res = Retriever(build_backend(cfg.backend), emb, cfg.retrieval).search(query)
    for i, r in enumerate(res.results[:k], start=1):
        typer.echo(f"{i:2d}. {r.doc_id}  score={r.score:.4f}  ranks={r.ranks}\n    {r.best_text[:200]}")
    typer.echo(f"timings ms: { {k: round(v, 1) for k, v in res.timings_ms.items()} }")


@app.command("eval")
def eval_cmd(config: list[Path], quiet: bool = False):
    """Evaluate one or more configs; writes runs/<name>/."""
    from cer.eval.runner import run_experiment
    for c in config:
        s = run_experiment(ExperimentConfig.from_yaml(c), progress=not quiet)
        typer.echo(json.dumps({"name": s["name"], **{k: round(v, 4) for k, v in s["metrics"].items()},
                               "p95_ms": round(s["latency_ms"]["total"]["p95"], 1)}))


@app.command()
def compare(run_a: Path, run_b: Path, qrels_dir: Path, split: str = "test", metric: str = "ndcg@10"):
    """Paired permutation test between two run files."""
    from cer.data.beir import load_qrels
    from cer.eval.metrics import per_query
    from cer.eval.runner import read_trec
    from cer.eval.significance import paired_permutation_test

    qrels = load_qrels(qrels_dir, split)
    a = per_query(read_trec(run_a), qrels, [metric])[metric]
    b = per_query(read_trec(run_b), qrels, [metric])[metric]
    typer.echo(json.dumps(paired_permutation_test(a, b), indent=2))


README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def replace_block(text: str, block: str) -> str:
    """Replace what lies between the README result markers (markers kept)."""
    i, j = text.find(README_START), text.find(README_END)
    if i < 0 or j < i:
        raise ValueError(f"markers {README_START} / {README_END} not found")
    return text[: i + len(README_START)] + "\n" + block + "\n" + text[j:]


@app.command()
def report(runs_dir: Path = Path("runs"), out: Path = Path("reports/results.md"), readme: Path = Path("README.md")):
    """Collect runs/*/metrics.json into one markdown table (also written between the README result markers)."""
    rows = []
    for m in sorted(runs_dir.glob("*/metrics.json")):
        s = json.loads(m.read_text(encoding="utf-8"))
        rows.append(s)
    if not rows:
        raise typer.Exit("no runs found")
    metrics = list(rows[0]["metrics"])
    lines = ["| run | dataset | " + " | ".join(metrics) + " | p50 ms | p95 ms |",
             "|---|---|" + "---|" * (len(metrics) + 2)]
    for s in rows:
        lt = s["latency_ms"]["total"]
        lines.append(f"| {s['name']} | {s['dataset']} | " + " | ".join(f"{s['metrics'][k]:.4f}" for k in metrics)
                     + f" | {lt['p50']:.1f} | {lt['p95']:.1f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if readme.exists() and README_START in (text := readme.read_text(encoding="utf-8")):
        readme.write_text(replace_block(text, "\n".join(lines)), encoding="utf-8")
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()
