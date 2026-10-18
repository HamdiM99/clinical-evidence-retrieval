"""Social-media visual: where the right study lands for two real SciFact test claims, per system.
Stances (SUPPORTS / CONTRADICTS) are the official SciFact annotations (claims_dev.jsonl).

Ranks are read from the run files (no hand-typed numbers). Output: docs/article/social/rank_example_{fr,en}.png
(1080 x 1350, LinkedIn portrait).

Usage: python docs/article/make_social.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cer.data.beir import load_qrels  # noqa: E402
from cer.eval.runner import read_trec  # noqa: E402

OUT = ROOT / "docs/article/social"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, SLOT = "#111827", "#6B7280", "#E5E7EB"
COL = {"hybrid": "#1E3A8A", "minilm": "#D97706", "c2": "#0F766E"}
RUNS = {"hybrid": "scifact-hybrid", "minilm": "scifact-hybrid-ms-marco-minilm-l-6-v2-L256-top20",
        "c2": "scifact-hybrid-c2-listwise-scifact-L256-top20"}
CLAIMS = ["185", "742"]
SLOTS = 14

TEXT = {
    "fr": {
        "title": "Où arrive la bonne étude ?",
        "subtitle": "Deux affirmations réelles (jeu de test SciFact), trois systèmes de recherche",
        "claims": {"185": "« Le cancer du sein est déterminé exclusivement par des facteurs génétiques. »",
                   "742": "« Les macrolides n'ont aucun effet protecteur\ncontre l'infarctus du myocarde. »"},
        "evidence": {"185": "Million Women Study, gènes et environnement chez 7 610 patientes · la CONTREDIT",
                     "742": "« Antibiotics and risk of subsequent first-time acute myocardial infarction » · la CONFIRME"},
        "systems": {"hybrid": "Recherche hybride\n(mots-clés + embeddings)", "minilm": "Re-ranker\nprêt à l'emploi",
                    "c2": "Re-ranker\naffiné (608 affirmations)"},
        "zone": "top 5 lu\npar le LLM",
        "rank": "rang",
        "footer": "Rangs calculés depuis les runs de clinical-evidence-retrieval (top 20 re-classé) · "
                  "CONTREDIT / CONFIRME : annotations officielles SciFact",
    },
    "en": {
        "title": "Where does the right study land?",
        "subtitle": "Two real claims (SciFact test set), three retrieval systems",
        "claims": {"185": "“Breast cancer development is determined exclusively\nby genetic factors.”",
                   "742": "“Macrolides have no protective effect\nagainst myocardial infarction.”"},
        "evidence": {"185": "Million Women Study, genes and environment in 7,610 patients · CONTRADICTS it",
                     "742": "“Antibiotics and risk of subsequent first-time acute myocardial infarction” · SUPPORTS it"},
        "systems": {"hybrid": "Hybrid search\n(keywords + embeddings)", "minilm": "Off-the-shelf\nre-ranker",
                    "c2": "Fine-tuned\nre-ranker (608 claims)"},
        "zone": "top 5 read\nby the LLM",
        "rank": "rank",
        "footer": "Ranks computed from the clinical-evidence-retrieval run files (top 20 re-ranked) · "
                  "SUPPORTS / CONTRADICTS: official SciFact annotations",
    },
}


def ranks() -> dict[str, dict[str, int]]:
    qrels = load_qrels(ROOT / "data/scifact", "test")
    out = {}
    for qid in CLAIMS:
        rel = {d for d, v in qrels[qid].items() if v > 0}
        out[qid] = {}
        for k, run in RUNS.items():
            r = read_trec(ROOT / "runs" / run / "run.trec")[qid]
            order = sorted(r, key=lambda d: (-r[d], d))
            out[qid][k] = min(order.index(d) + 1 for d in rel if d in order)
    return out


def panel(ax, qid: str, rk: dict[str, int], t: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    claim = t["claims"][qid]
    ax.text(0.0, 0.985, claim, fontsize=13, color=INK, weight="semibold", va="top", linespacing=1.25)
    ax.text(0.0, 0.90 - 0.07 * claim.count("\n"), t["evidence"][qid], fontsize=9.5, color=MUTED, va="top",
            style="italic")
    top, bottom = 0.76, 0.03
    h = (top - bottom) / SLOTS
    ax.add_patch(FancyBboxPatch((0.205, top - 5 * h + 0.002), 0.79, 5 * h - 0.004,
                                boxstyle="round,pad=0,rounding_size=0.012", fc="#F0FDFA", ec="none"))
    ax.text(0.10, top - 2.5 * h, t["zone"], fontsize=9, color="#0F766E", ha="center", va="center", weight="semibold")
    for i in range(SLOTS):
        y = top - (i + 0.5) * h
        ax.text(0.19, y, str(i + 1), fontsize=7.5, color=MUTED, ha="right", va="center")
    for j, k in enumerate(("hybrid", "minilm", "c2")):
        x = 0.24 + j * 0.255
        for i in range(SLOTS):
            y = top - (i + 1) * h
            hit = i + 1 == rk[k]
            ax.add_patch(FancyBboxPatch((x, y + 0.0035), 0.2, h - 0.007, boxstyle="round,pad=0,rounding_size=0.006",
                                        fc=COL[k] if hit else SLOT, ec="none", alpha=1 if hit else 0.75))
            if hit:
                ax.text(x + 0.1, y + h / 2, f"#{rk[k]}", fontsize=11, color="white", ha="center", va="center",
                        weight="bold")
    return None


def system_headers(ax, t: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for j, k in enumerate(("hybrid", "minilm", "c2")):
        ax.text(0.24 + j * 0.255 + 0.1, 0.5, t["systems"][k], fontsize=10.5, color=COL[k], ha="center",
                va="center", weight="bold", linespacing=1.2)


def make(lang: str, rk: dict) -> None:
    t = TEXT[lang]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"]})
    fig = plt.figure(figsize=(7.2, 9.0), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.955, t["title"], fontsize=24, weight="bold", color=INK, va="top")
    fig.text(0.06, 0.905, t["subtitle"], fontsize=11, color=MUTED, va="top")
    system_headers(fig.add_axes([0.06, 0.83, 0.88, 0.05]), t)
    panel(fig.add_axes([0.06, 0.45, 0.88, 0.37]), CLAIMS[0], rk[CLAIMS[0]], t)
    panel(fig.add_axes([0.06, 0.07, 0.88, 0.37]), CLAIMS[1], rk[CLAIMS[1]], t)
    fig.text(0.06, 0.03, t["footer"], fontsize=7.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.445, 0.445], color=SLOT, lw=1))
    path = OUT / f"rank_example_{lang}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(path, rk)


if __name__ == "__main__":
    r = ranks()
    for lang in ("fr", "en"):
        make(lang, r)
