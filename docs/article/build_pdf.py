"""Build the article PDF from docs/article/figures (run make_figures.py first). Every number comes from
figures/numbers.json, itself computed from the experiment artifacts.

Usage: python docs/article/build_pdf.py  ->  docs/article/reranking-little-gain-real-risks.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
N = json.loads((FIG / "numbers.json").read_text(encoding="utf-8"))
OUT = HERE / "reranking-little-gain-real-risks.pdf"
KR = json.loads((HERE / "data/kaggle_runs.json").read_text(encoding="utf-8"))

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "cambria.ttc"), subfontIndex=0))
for name, file in [("Body-Bold", "cambriab.ttf"), ("Body-Italic", "cambriai.ttf"),
                   ("Body-BoldItalic", "cambriaz.ttf"), ("Sans", "segoeui.ttf"), ("Sans-Semi", "seguisb.ttf"),
                   ("Sans-Bold", "segoeuib.ttf"), ("Sans-Italic", "segoeuii.ttf"), ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")

INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#0F766E")
NAVY = colors.HexColor("#1E3A8A")
SOFT = colors.HexColor("#F0FDFA")
SOFT2 = colors.HexColor("#F8FAFC")
RULE = colors.HexColor("#E5E7EB")

W, H = A4
M = 20 * mm
TW = W - 2 * M

S = {
    "kicker": ParagraphStyle("kicker", fontName="Sans-Semi", fontSize=8, textColor=ACCENT, leading=10,
                             spaceAfter=4),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=21, leading=25, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Body-Italic", fontSize=11.5, leading=15, textColor=MUTED,
                               spaceAfter=8),
    "byline": ParagraphStyle("byline", fontName="Sans", fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13, leading=16, textColor=INK, spaceBefore=12,
                         spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Semi", fontSize=10.5, leading=13, textColor=ACCENT, spaceBefore=7,
                         spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.4, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "body_left": ParagraphStyle("body_left", fontName="Body", fontSize=10, leading=14.4, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10, leading=14.2, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=3, alignment=TA_LEFT),
    "caption": ParagraphStyle("caption", fontName="Sans", fontSize=7.8, leading=10.2, textColor=MUTED,
                              spaceBefore=3, spaceAfter=10),
    "box_title": ParagraphStyle("box_title", fontName="Sans-Semi", fontSize=8.6, leading=11, textColor=ACCENT,
                                spaceAfter=3),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=8.4, leading=11.8, textColor=INK, spaceAfter=2),
    "tldr": ParagraphStyle("tldr", fontName="Sans", fontSize=8.9, leading=12.6, textColor=INK, leftIndent=10,
                           bulletIndent=0, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.8, leading=10, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Sans-Semi", fontSize=7.8, leading=10, textColor=colors.white),
    "ref": ParagraphStyle("ref", fontName="Sans", fontSize=7.6, leading=10, textColor=INK, leftIndent=12,
                          firstLineIndent=-12, spaceAfter=2.5),
}

story: list = []
fig_no = [0]
tab_no = [0]


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def pv(p: float) -> str:
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def P(text: str, style: str = "body") -> None:
    story.append(Paragraph(text, S[style]))


def H1(text: str) -> None:
    P(text, "h1")


def H2(text: str) -> None:
    P(text, "h2")


def bullets(items: list[str], style: str = "bullet") -> None:
    for it in items:
        story.append(Paragraph(it, S[style], bulletText="•"))


def figure(name: str, caption: str, width: float = TW) -> None:
    fig_no[0] += 1
    path = FIG / f"{name}.png"
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    story.append(KeepTogether([img, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", S["caption"])]))


def figures_side(left: tuple[str, float], right: tuple[str, float], caption: str) -> None:
    fig_no[0] += 1
    cells = []
    for name, frac in (left, right):
        path = FIG / f"{name}.png"
        w, h = PILImage.open(path).size
        width = TW * frac
        cells.append(Image(str(path), width=width, height=width * h / w))
    t = Table([cells], colWidths=[TW * left[1] + 2 * mm, TW * right[1] + 2 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(KeepTogether([t, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", S["caption"])]))


def equation(name: str, width: float) -> None:
    path = FIG / f"{name}.png"
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1, 3))


def callout(title: str, lines: list[str], fill=SOFT, edge=ACCENT) -> None:
    content = [Paragraph(title, S["box_title"])] + [Paragraph(x, S["box"]) for x in lines]
    t = Table([[content]], colWidths=[TW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("LINEBEFORE", (0, 0), (0, -1), 2.2, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(Spacer(1, 2))
    story.append(KeepTogether(t))
    story.append(Spacer(1, 7))


def table(header: list[str], rows: list[list[str]], widths: list[float], caption: str,
          highlight: list[int] | None = None) -> None:
    tab_no[0] += 1
    data = [[Paragraph(h, S["cellh"]) for h in header]] + [[Paragraph(c, S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=[TW * w for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), SOFT2))
    for i in highlight or []:
        st.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", S["caption"]), t]))
    story.append(Spacer(1, 9))


# ============================================================================= numbers used in the text
nf, sf = "nfcorpus", "scifact"
R = {d: {k: N[f"{d}/{k}"] for k in ("bm25", "dense", "hybrid", "minilm", "c2")} for d in (nf, sf)}
cm = N["cmp"]
ts = N["train_stats"]
lb, r1, ev, ls = N["labels"], N["rank1"], N["evidence_ndcg"], N["label_split"]
neg, i8, ex = N["negation"], N["int8"], N["examples"]
gpu = {(r["model"], r["precision"], r["passages"], r["batch"]): r for r in N["gpu"]}
mini16 = {n: gpu[("ms-marco-MiniLM-L-6-v2", "fp16", n, n)] for n in (20, 50, 100)}
mini32 = {n: gpu[("ms-marco-MiniLM-L-6-v2", "fp32", n, n)] for n in (20, 50, 100)}
bge16 = {n: gpu[("bge-reranker-base", "fp16", n, n)] for n in (20, 50, 100)}
cpu = {(r["backend"], r["passages"], r["batch_size"]): r for r in N["cpu"]}
cpu_int8_20 = cpu[("ONNX Runtime int8 (dynamic)", 20, 20)]
cpu_fp32_20 = cpu[("PyTorch fp32", 20, 20)]
ab, tn, ra = N["ablation"], N["topn"], N["recall_at"]
n_sf_claims = ts["datasets"]["scifact"]["queries"]
n_nf_pairs = ts["datasets"]["nfcorpus"]["pairs"]
pct = lambda a, b: f"{100 * a / b:.0f}%"


def c(key: str) -> str:
    """'+0.010 (p = 0.098)' for a test comparison."""
    return f"{sg(cm[key]['delta'])} ({pv(cm[key]['p'])})"


def ci(d: dict) -> str:
    return f"[{sg(d['ci'][0])}, {sg(d['ci'][1])}]"


def lsd(key: str) -> dict:
    return ls[key]


# ============================================================================= front matter
P("TECHNICAL NOTE  ·  MEDICAL INFORMATION RETRIEVAL  ·  RE-RANKING", "kicker")
P("Re-ranking: little gain, real risks", "title")
P("Before you add a cross-encoder on top of hybrid retrieval: gains, risks and serving cost in biomedical "
  "search, measured from retrieval quality to serving cost.", "subtitle")
REPO = "https://github.com/HamdiM99/clinical-evidence-retrieval"
P(f"Mustapha HAMDI  ·  October 2026  ·  Code, configs and run files: "
  f"<a href='{REPO}' color='#0F766E'><u>github.com/HamdiM99/clinical-evidence-retrieval</u></a>", "byline")

tldr = [
    f"<b>Little gain.</b> No re-ranker beats the hybrid first stage significantly: C2 (fine-tuned) "
    f"{c('nfcorpus/c2-hybrid')} on NFCorpus, {c('scifact/c2-hybrid')} on SciFact, in nDCG@10.",
    f"<b>The benchmark decides the story.</b> The off-the-shelf MS MARCO re-ranker lowers SciFact nDCG@10 "
    f"{sg(cm['scifact/minilm-hybrid']['delta'])} (p = {cm['scifact/minilm-hybrid']['p']:.3f}, not significant). "
    f"The whole drop sits on the {ls['without_evidence']} claims whose \"relevant\" document holds no evidence "
    f"({sg(lsd('minilm-hybrid/without_evidence')['delta'])}); on the {ls['with_evidence']} claims with real "
    f"evidence it is {sg(lsd('minilm-hybrid/with_evidence')['delta'])} ({pv(lsd('minilm-hybrid/with_evidence')['p'])}).",
    f"<b>Fine-tuning mostly learns the labels.</b> Broad NFCorpus labels never help. A listwise model trained on "
    f"{n_sf_claims} SciFact claims beats the off-the-shelf one by {c('scifact/c2-minilm')}, but that gain is "
    f"{sg(lsd('c2-minilm/without_evidence')['delta'])} on no-evidence claims and "
    f"{sg(lsd('c2-minilm/with_evidence')['delta'])} ({pv(lsd('c2-minilm/with_evidence')['p'])}) on evidence ones: "
    f"{pct(ls['train_positives_without_rationale'], ls['train_positives'])} of its training positives contain no "
    "evidence either.",
    "<b>Real risks.</b> A re-ranker scores relevance, not stance: one study gets the same rank for a claim and for its "
    f"negation. Int8 quantization keeps the mean but changes the top-10 set on "
    f"{pct(i8[nf]['queries'] - i8[nf]['same_top10_set'], i8[nf]['queries'])} of queries.",
    f"<b>Serving.</b> {mini16[20]['p95']:.0f} ms p95 for 20 passages on a T4 in fp16, ≈ "
    f"{cpu_int8_20['p95_ms'] / 1000:.1f} s on a laptop CPU even in int8. Cost is solvable; usefulness is the question.",
]
tl = [Paragraph("TL;DR", S["box_title"])] + [Paragraph(x, S["tldr"], bulletText="•") for x in tldr]
t = Table([[tl]], colWidths=[TW])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
                       ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                       ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
story.append(t)
story.append(Spacer(1, 6))

# ============================================================================= 1
H1("1. Why re-ranking, and why look twice")
P("A clinical assistant that answers from guidelines and studies is only as good as the passages it retrieves: "
  "if the right recommendation is not in the context, no language model can recover it, and if a weak or "
  "irrelevant source is ranked first, it will be cited confidently. Production retrieval is therefore usually a "
  "<b>two-stage</b> pipeline (Figure 1): a fast first stage searches the whole collection and returns about a "
  "hundred candidates, then a slower, more accurate model <i>re-ranks</i> the best of them.")
figure("f1_pipeline", "The pipeline studied here. Offline, documents are chunked, embedded and indexed once. "
       "Online, BM25 and vector search run in parallel, their lists are merged with Reciprocal Rank Fusion, and a "
       "cross-encoder re-orders the top 20. For evaluation, the fused top 100 is exported once and every re-ranker "
       "is replayed on exactly the same candidates.")
callout("Bi-encoder vs cross-encoder, in one paragraph", [
    "A <b>bi-encoder</b> (the embedding model of the first stage) turns the query and each passage into vectors "
    "<i>separately</i>; passages are encoded once at indexing time and search is a nearest-neighbour lookup, which "
    "scales to millions of documents. A <b>cross-encoder</b> reads the query and one passage <i>together</i>, so "
    "every word of the query can attend to every word of the passage. It judges relevance better, but must run once "
    "per (query, passage) pair at query time: affordable for 20 candidates, not for a corpus."])
P("This note measures three things: whether a re-ranker helps after a strong hybrid first stage, what the "
  "benchmark actually rewards when it says it does, and what it costs to serve. The answers are less flattering to "
  "re-ranking than the usual RAG recipe suggests, and the most useful finding concerns the evaluation itself.")

# ============================================================================= 2
H1("2. Setup: data, first stage and an evaluation you can trust")
H2("Two public benchmarks")
P(f"<b>NFCorpus</b> (Boteva et al., 2016): nutrition and medical questions written for the public, linked to "
  f"PubMed articles, about 40 relevant documents per query. <b>SciFact</b> (Wadden et al., 2020): scientific "
  f"<i>claims</i> (\"Macrolides protect against myocardial infarction\") and the abstracts cited for them. Both come "
  f"from BEIR (Thakur et al., 2021); test sets have {N['n_queries'][nf]} and {N['n_queries'][sf]} queries, indexes "
  f"{N['chunks'][nf]:,} and {N['chunks'][sf]:,} chunks. For SciFact we also use the <b>official annotation</b> of "
  "the original release, which says for each cited abstract whether it SUPPORTS the claim, CONTRADICTS it, or "
  "neither; BEIR keeps only \"cited = relevant\" (Section 6).")
H2("A hybrid first stage")
P("Documents are split into chunks of about 250 words and indexed in Elasticsearch twice over: as text for BM25 "
  "and as vectors from <i>multilingual-e5-base</i> (Wang et al., 2024). For each query both lists are retrieved in "
  "one round trip, chunks are mapped to their documents, and the rankings are merged with Reciprocal Rank Fusion "
  "(Cormack et al., 2009), which only uses ranks and needs no score calibration:")
equation("eq_rrf", TW * 0.62)
figure("f2_baselines", f"First-stage results on the test sets. Hybrid fusion beats BM25 on both datasets "
       f"(NFCorpus {c('nfcorpus/hybrid-bm25')}, SciFact {c('scifact/hybrid-bm25')}); it is the baseline every "
       "re-ranker has to beat.")
callout("Reading the metrics", [
    "<b>nDCG@10</b> scores the first ten results: a relevant document counts more the higher it is ranked "
    "(logarithmic discount), normalised by the best possible ranking, so 1.0 is perfect. <b>Recall@100</b> is the "
    "share of relevant documents anywhere in the top 100: what a second stage could promote. <b>MRR@10</b> looks "
    "only at the rank of the first relevant result."])
equation("eq_ndcg", TW * 0.5)
H2("Protocol")
bullets([
    "<b>Replay, not re-run.</b> The fused top 100 of every query is exported once; every re-ranker, depth and "
    "checkpoint is scored on identical candidates. Replaying the export without re-ranking reproduces the "
    "first-stage runs exactly.",
    "<b>Validation for decisions, test once.</b> Depths, checkpoints and model choices are made on NFCorpus dev and "
    "on 200 held-out SciFact training claims; each final system is evaluated once on test, with the decision rule "
    "written down before looking.",
    "<b>Paired significance.</b> Systems are compared query by query with a paired permutation test (10,000 "
    "permutations; Smucker et al., 2007) and 95% bootstrap confidence intervals.",
])

# ============================================================================= 3
H1("3. An off-the-shelf re-ranker: no significant gain")
P(f"We first re-rank with <i>cross-encoder/ms-marco-MiniLM-L-6-v2</i> (22M parameters). The depth is tuned on "
  f"validation (Figure 3, right): re-ranking more never helps, and on SciFact every depth is below the first stage "
  f"({f3(tn[sf]['20'])} at top 20 vs {f3(tn[sf]['fused'])}). We keep top 20 and evaluate once on test (Table 1).")
figures_side(("f3_recall_ceiling", 0.478), ("f4_topn", 0.478),
             f"Left: share of relevant documents within the first k candidates (test). On NFCorpus only "
             f"{ra[nf]['20'] * 100:.0f}% are in the re-ranked window and {ra[nf]['100'] * 100:.0f}% in the top 100: "
             f"a hard ceiling for any re-ranker. Right: effect of the depth on validation.")
table(["Test, nDCG@10", "BM25", "Dense", "Hybrid RRF", "+ MiniLM (MS MARCO)", "Δ vs hybrid (p)", "95% CI"],
      [[DS_, f3(R[d]["bm25"]["ndcg@10"]), f3(R[d]["dense"]["ndcg@10"]), f3(R[d]["hybrid"]["ndcg@10"]),
        f3(R[d]["minilm"]["ndcg@10"]), c(f"{d}/minilm-hybrid"), ci(cm[f"{d}/minilm-hybrid"])]
       for d, DS_ in ((nf, "NFCorpus"), (sf, "SciFact"))],
      [0.15, 0.09, 0.09, 0.12, 0.17, 0.2, 0.18], "Off-the-shelf re-ranking on test (top 20 re-ranked).")
P(f"On NFCorpus the re-ranker moves the first relevant result up (MRR@10 {f3(R[nf]['hybrid']['mrr@10'])} → "
  f"{f3(R[nf]['minilm']['mrr@10'])}) but the nDCG gain is not significant. On SciFact it <i>lowers</i> nDCG@10, "
  f"with an interval that reaches zero (p = {cm['scifact/minilm-hybrid']['p']:.3f}). The natural reading is a "
  "domain mismatch, since MS MARCO teaches \"does this passage answer the question?\" while SciFact asks for the "
  "evidence on a claim. Section 6 shows that this reading is mostly wrong.")

# ============================================================================= 4
H1("4. Fine-tuning: an ablation, one factor at a time")
P("The fix one would try is to fine-tune the re-ranker on the training splits, with <b>hard negatives</b> mined "
  "by our own first stage: non-relevant documents the retriever ranks high, the mistakes the re-ranker must "
  "correct (Figure 4).")
figure("f5_pairs", "Building training pairs for one query from the first-stage ranking.")
P(f"The set has {ts['train_pairs']:,} pairs, {ts['positives_share'] * 100:.0f}% positive; training queries whose "
  "text equals a test or validation query were removed. The first run uses pointwise binary cross-entropy:")
equation("eq_bce", TW * 0.66)
P(f"Figure 5 shows validation during training: the model first forgets (SciFact falls to "
  f"{f3(N['run1_scifact_min'])}), NFCorpus slowly recovers, SciFact never does, and no checkpoint beats the starting "
  "model. We then changed one factor per run (Figure 6): a 4× lower learning rate (A) only delays the collapse; a "
  "<b>listwise</b> loss that ranks one positive against seven hard negatives of the same query (Gao et al., 2021) "
  "helps (B); training on NFCorpus alone never improves anything (C1); training on SciFact alone does (C2).")
equation("eq_listwise", TW * 0.6)
figure("f6_learning_curves", "Validation nDCG@10 during fine-tuning (Kaggle T4). Pointwise runs degrade with a high "
       "or a low learning rate; the listwise run on SciFact only improves and stays there.")
figure("f7_ablation", "Best validation checkpoint of each run, relative to the off-the-shelf model. \"Base kept\": no "
       "checkpoint beat it.")
P(f"With {n_nf_pairs:,} broad NFCorpus pairs the model got worse; with {n_sf_claims} SciFact claims, C2, it got "
  f"better on SciFact ({sg(ab[sf]['C2'])} on validation) without moving NFCorpus ({sg(ab[nf]['C2'])}). It was selected "
  f"among {N['c2_validation_evaluations']} evaluations of the same validation queries, so that number is optimistic; "
  "test decides.")

# ============================================================================= 5
H1("5. Test results with BEIR labels")
P("C2 was evaluated once on both test sets, with the rule fixed beforehand: keep it if it is not significantly worse "
  "than the first stage and beats the off-the-shelf model on SciFact (Figure 7, Table 2).")
figure("f8_forest", "Paired differences in nDCG@10 on test, BEIR relevance labels.")
table(["Test", "Hybrid RRF", "+ MiniLM", "+ C2", "C2 vs MiniLM", "C2 vs hybrid", "C2 vs hybrid: W / L / T"],
      [[f"{DS_} nDCG@10", f3(R[d]["hybrid"]["ndcg@10"]), f3(R[d]["minilm"]["ndcg@10"]), f3(R[d]["c2"]["ndcg@10"]),
        c(f"{d}/c2-minilm"), c(f"{d}/c2-hybrid"),
        f"{cm[f'{d}/c2-hybrid']['wins']} / {cm[f'{d}/c2-hybrid']['losses']} / {cm[f'{d}/c2-hybrid']['ties']}"]
       for d, DS_ in ((nf, "NFCorpus"), (sf, "SciFact"))],
      [0.17, 0.11, 0.1, 0.09, 0.19, 0.17, 0.17], "Final comparison on test (top 20 re-ranked).")
P(f"By the BEIR labels, C2 meets the rule: it beats the off-the-shelf model on SciFact ({c('scifact/c2-minilm')}) "
  f"and is level with the first stage ({c('scifact/c2-hybrid')}). On NFCorpus nothing is significant. Across queries "
  f"it wins some and loses others ({cm['scifact/c2-hybrid']['wins']} vs {cm['scifact/c2-hybrid']['losses']} on "
  "SciFact): it moves errors more than it removes them. Read with BEIR labels alone, this looks like \"the "
  "off-the-shelf model has a domain problem and fine-tuning fixes it\". The next section shows what is underneath.")

# ============================================================================= 6
H1("6. What the benchmark actually rewards")
P(f"In BEIR, a SciFact document is relevant if the claim <i>cites</i> it. The official annotation is stricter: "
  f"of the {lb['relevant_pairs']} relevant (claim, document) pairs of the test set, {lb['pairs_support']} SUPPORT "
  f"the claim, {lb['pairs_contradict']} CONTRADICT it, and <b>{lb['pairs_without_rationale']} "
  f"({pct(lb['pairs_without_rationale'], lb['relevant_pairs'])}) contain no sentence that settles it</b>. For "
  f"{lb['claims_only_without_rationale']} of the {lb['claims']} claims, every \"relevant\" document is of that "
  "kind. Splitting the test claims along this line changes the conclusions (Figure 8).")
figure("f12_label_split", "Differences in nDCG@10 (BEIR labels), split by the official annotation of the claim's "
       "relevant documents. Bars: mean; whiskers: 95% bootstrap CI.")
bullets([
    f"<b>The off-the-shelf model's loss is entirely on claims without evidence</b>: "
    f"{sg(lsd('minilm-hybrid/without_evidence')['delta'])} ({pv(lsd('minilm-hybrid/without_evidence')['p'])}) on the "
    f"{ls['without_evidence']} claims whose cited document settles nothing, "
    f"{sg(lsd('minilm-hybrid/with_evidence')['delta'])} ({pv(lsd('minilm-hybrid/with_evidence')['p'])}) on the "
    f"{ls['with_evidence']} others. It pushes down documents that are cited but say nothing about the claim, which "
    "is arguably the right behaviour, and the BEIR metric penalises it.",
    f"<b>C2's advantage over the off-the-shelf model is mostly the same effect in reverse</b>: "
    f"{sg(lsd('c2-minilm/without_evidence')['delta'])} ({pv(lsd('c2-minilm/without_evidence')['p'])}) on claims without "
    f"evidence, {sg(lsd('c2-minilm/with_evidence')['delta'])} ({pv(lsd('c2-minilm/with_evidence')['p'])}) on claims with "
    f"evidence. Its training positives are BEIR labels too, and {ls['train_positives_without_rationale']} of "
    f"{ls['train_positives']} ({pct(ls['train_positives_without_rationale'], ls['train_positives'])}) contain no "
    "evidence: C2 learned the benchmark's citation convention as much as evidence matching.",
    f"<b>Scored on evidence only</b> (a document is relevant only if it SUPPORTS or CONTRADICTS, {ev['claims']} "
    f"claims), the order is hybrid {f3(ev['hybrid'])}, MiniLM {f3(ev['minilm'])}, C2 {f3(ev['c2'])}; neither "
    f"difference with hybrid is significant (C2 {sg(ev['c2-hybrid']['delta'])}, "
    f"p = {ev['c2-hybrid']['p']:.3f}).",
])
P(f"Figure 9 gives the same picture from the top of the list. A document that actually SUPPORTS or CONTRADICTS the "
  f"claim is at rank 1 for {r1['hybrid']['evidence']['share'] * 100:.0f}% of claims with the hybrid first stage "
  f"(95% CI {r1['hybrid']['evidence']['ci'][0] * 100:.0f}–{r1['hybrid']['evidence']['ci'][1] * 100:.0f}%), "
  f"{r1['minilm']['evidence']['share'] * 100:.0f}% with MiniLM and {r1['c2']['evidence']['share'] * 100:.0f}% with "
  f"C2: overlapping intervals, no measurable difference. Contradicting evidence is at rank 1 for "
  f"{r1['hybrid']['contradict']['share'] * 100:.0f}–{max(r1[k]['contradict']['share'] for k in r1) * 100:.0f}% of "
  "claims: that is correct retrieval (refuting evidence is evidence), but it is exactly where a generator can turn "
  "a refutation into support.")
figure("f11_rank1", "What sits at rank 1 for the 300 SciFact test claims, by official label. \"Not cited\": a "
       "document the claim does not cite (no judgement available).")
table(["SciFact claim", "Study (official stance)", "Hybrid", "+ MiniLM", "+ C2"],
      [[f"<b>{q}</b> · {ex[q]['claim']}", f"{ex[q]['title']} <b>({ex[q]['label']})</b>",
        str(ex[q]["ranks"]["hybrid"]), str(ex[q]["ranks"]["minilm"]), str(ex[q]["ranks"]["c2"])] for q in ex],
      [0.33, 0.43, 0.08, 0.08, 0.08], "Rank of the evidence document for three real test claims (ranks from the run "
      "files, stance from the official annotation). Claims 742 and 743 say opposite things; the same sentence of the "
      "same study SUPPORTS one and CONTRADICTS the other, and every system ranks it identically for both.")

# ============================================================================= 7
H1("7. What it costs to serve")
P("A re-ranker earns its place only if it fits the latency budget: here p95 &lt; 300 ms end to end, of which about "
  "100 ms for re-ranking. We measured C2's architecture on a laptop CPU (ONNX fp32 and dynamically quantized int8) "
  "and on an NVIDIA T4 (fp32 and fp16), with real passages, warm-up excluded (Figure 10).")
bullets([
    f"<b>GPU fp16 fits with margin</b>: {mini16[20]['p95']:.0f} ms p95 at depth 20, {mini16[50]['p95']:.0f} ms at "
    f"50; fp32 needs {mini32[20]['p95']:.0f} ms at 20. The larger bge-reranker-base fits only at depth 20 "
    f"({bge16[20]['p95']:.0f} ms).",
    f"<b>Laptop CPU</b>: {cpu_int8_20['p95_ms']:.0f} ms p95 at depth 20 in int8 ({cpu_fp32_20['p95_ms']:.0f} ms in "
    "fp32): a fallback, not a serving target.",
    f"<b>Tokenization becomes the bottleneck on GPU</b>: {mini16[20]['tok']:.0f} of {mini16[20]['p50']:.0f} ms at "
    "depth 20. Passages can be tokenized at indexing time and stored with the chunk.",
])
figure("f9_latency", "Left: p95 latency per query against the number of passages re-ranked (log scale). Right: on "
       "the GPU in fp16, tokenization on the CPU is as large as the model itself.")
figure("f10_quality_latency", f"Quality against latency (test). Filled points are measured on the laptop; the hollow "
       f"point is an estimate for C2 served on a T4 (first-stage p50 + GPU re-ranking p50 at depth 20 = "
       f"{N['c2_t4_estimate_p50_ms'][nf]:.0f} ms on NFCorpus, {N['c2_t4_estimate_p50_ms'][sf]:.0f} ms on SciFact). "
       "On a GPU the re-ranker fits the target; it does not move quality off the hybrid point.")

# ============================================================================= 8
H1("8. Risks: what a re-ranker can get wrong")
P("A re-ranker does not generate text, but it decides which sources the language model reads and cites first. It "
  "optimises topical relevance, not truth: it cannot tell whether a passage supports or refutes the claim, and "
  "nothing in its training asks it to. Table 4 lists what that can cause, what this study measured, and how to "
  "contain it.")
tab_no[0] += 1
risk_rows = [
    ("<b>Clinical safety</b>", "", "", ""),
    ("Stance confusion", "A refuting passage is ranked first because it is on topic; the generator reads it as "
     "support.", f"Measured: the macrolides study gets the same rank for claim 742 and its negation 743; contradicting "
     f"evidence is at rank 1 for {r1['hybrid']['contradict']['share'] * 100:.0f}–"
     f"{max(r1[k]['contradict']['share'] for k in r1) * 100:.0f}% of claims.",
     "Stance check after generation; evaluate with SUPPORT / CONTRADICT labels."),
    ("Negation", "\"does not\", \"no effect\" read as the positive claim.",
     f"Indirect: MiniLM vs hybrid {sg(neg['minilm-hybrid/negated']['delta'])} on negated claims (n = "
     f"{neg['minilm-hybrid/negated']['n']}, p = {neg['minilm-hybrid/negated']['p']:.2f}) vs "
     f"{sg(neg['minilm-hybrid/other']['delta'])} on the others (regex selection, small sample).",
     "Negation-aware test sets; keep negation in the grounding check."),
    ("Wrong population", "A paediatric or animal study for an adult question.", "Not measured.",
     "Population check in the grounding guard; metadata filters."),
    ("Weak evidence first", "A case report ranked above a grade A recommendation.", "Not measured.",
     "Evidence grade as a filter or a ranking feature."),
    ("False confidence", "A high score read as certainty.", "Not measured.",
     "Calibration; answer / qualify / abstain decided in a separate layer."),
    ("<b>Quality</b>", "", "", ""),
    ("No gain over the first stage", "Added latency and complexity for nothing.",
     f"Measured: C2 vs hybrid {c('nfcorpus/c2-hybrid')} and {c('scifact/c2-hybrid')}.",
     "Keep re-ranking behind a flag; ship only on a significant gain."),
    ("Errors moved, not removed", "Some queries improve, others break.",
     f"Measured: {cm['scifact/c2-hybrid']['wins']} wins, {cm['scifact/c2-hybrid']['losses']} losses, "
     f"{cm['scifact/c2-hybrid']['ties']} ties (C2 vs hybrid, SciFact).",
     "Per-query regression review before release."),
    ("Fragile fine-tuning", "A training run silently degrades the model.",
     f"Measured: SciFact validation fell from {f3(KR['base']['scifact'])} to {f3(N['run1_scifact_min'])} in run 1.",
     "Evaluate during training; keep the base model unless beaten."),
    ("<b>Evaluation</b>", "", "", ""),
    ("Label convention taken for quality", "The metric rewards finding the cited document, not the one that settles "
     "the claim.", f"Measured: {pct(lb['pairs_without_rationale'], lb['relevant_pairs'])} of BEIR SciFact relevant "
     f"pairs hold no evidence; both the MiniLM loss and the C2 gain sit on those claims.",
     "Evaluate with stance labels; report evidence-only metrics next to BEIR ones."),
    ("Noise taken for gain", "A lucky checkpoint or a small test set.",
     f"Measured: C2 chosen after {N['c2_validation_evaluations']} evaluations of the same validation queries; test "
     f"sets of {N['n_queries'][nf]} and {N['n_queries'][sf]} queries; CIs of ±0.03 on SciFact.",
     "Fixed decision rules; one test evaluation; paired tests and CIs."),
    ("<b>Operations</b>", "", "", ""),
    ("Cost and failure", "Timeouts, GPU unavailable.",
     f"Measured: {cpu_int8_20['p95_ms']:.0f} ms (CPU int8) vs {mini16[20]['p95']:.0f} ms (T4 fp16) p95 at depth 20.",
     "GPU fp16; conditional re-ranking; fall back to the hybrid ranking."),
    ("Silent drift", "Quantization or a new model changes rankings with the same mean.",
     f"Measured: int8 keeps nDCG ({f3(i8[nf]['ndcg_fp32'])} → {f3(i8[nf]['ndcg_int8'])}) but the same top-10 set on "
     f"only {i8[nf]['same_top10_set']}/{i8[nf]['queries']} NFCorpus and {i8[sf]['same_top10_set']}/"
     f"{i8[sf]['queries']} SciFact validation queries.", "Version models; regression tests on rankings, not only means."),
    ("Traceability", "No way to explain why a passage came first.", "Not measured (design).",
     "Log first-stage ranks and re-ranker scores with every answer."),
]
data = [[Paragraph(h, S["cellh"]) for h in ("Risk", "What can happen", "Evidence in this study", "Mitigation")]]
st = [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
      ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2.8),
      ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LEFTPADDING", (0, 0), (-1, -1), 4),
      ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
for row in risk_rows:
    i = len(data)
    data.append([Paragraph(x, S["cell"]) for x in row])
    if not row[1]:
        st += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), SOFT)]
rt = Table(data, colWidths=[TW * w for w in (0.16, 0.25, 0.33, 0.26)], repeatRows=1)
rt.setStyle(TableStyle(st))
story.append(Paragraph(f"<b>Table {tab_no[0]}.</b> Risks of a re-ranking stage in medical retrieval.", S["caption"]))
story.append(rt)
P("Rows marked \"Not measured\" are identified risks, not results of this study.", "caption")
callout("Relevance is not evidence", [
    "Neither the re-ranker nor the first stage knows whether a passage <i>supports</i> the claim; the BEIR metric does "
    "not know either. Safety therefore cannot be delegated to ranking. It belongs to a layer after generation that "
    "checks that each citation exists, quotes its source exactly, keeps negation, population and figures, and decides "
    "whether to answer, qualify or abstain."])

# ============================================================================= 9
H1("9. Lessons")
bullets([
    "<b>Measure the re-ranker against the first stage, not against nothing.</b> A strong hybrid retriever leaves "
    "little room.",
    "<b>Know what your labels reward.</b> Decompose a benchmark gain by label type before concluding: here it changed "
    "\"domain mismatch, fixed by fine-tuning\" into \"citation convention, learned by fine-tuning\".",
    "<b>Label quality beats label quantity</b>, and the training labels carry the evaluation's conventions with them.",
    "<b>Use a ranking objective.</b> A listwise loss over a query's own hard negatives fits what a re-ranker does.",
    "<b>Ablate one factor at a time and keep the failed runs.</b> Without them the conclusion would have been "
    "\"lower the learning rate\".",
    "<b>Recall is the ceiling.</b> When the first stage misses most relevant documents, gains come from candidate "
    "generation and query processing.",
    "<b>Do not delegate safety to ranking</b>: measure what sits at rank 1 with stance labels, and verify citations "
    "after generation.",
    "<b>A GPU makes a re-ranker affordable; it does not make it useful.</b>",
])
H2("Limitations and next steps")
P("Two small English benchmarks; labels from benchmark annotations, not clinicians; one training seed per run; "
  "laptop-scale indexes, so first-stage latency says nothing yet about 100M documents; GPU throughput is "
  "estimated, not load-tested. Next: train and select with stance-aware labels (evidence only), measure stance "
  "errors of a generator on the retrieved passages, a load test at 50 req/s, French queries and guideline "
  "documents.", "body")
H2("Reproducibility")
P("Every number and figure in this note is generated from run files by <font face='Mono'>docs/article/"
  "make_figures.py</font>, which also writes <font face='Mono'>post-numbers.md</font>. Commands: "
  "<font face='Mono'>cer candidates</font>, <font face='Mono'>cer rerank</font>, "
  "<font face='Mono'>scripts/train_reranker.py</font>, <font face='Mono'>cer export-onnx</font>, "
  "<font face='Mono'>scripts/bench_reranker*.py</font>; the official SciFact labels are read with "
  "<font face='Mono'>scripts/extract_scifact_labels.py</font>.", "body_left")

H1("References")
for ref in [
    "Boteva, V., Gholipour, D., Sokolov, A., Riezler, S. (2016). A Full-Text Learning to Rank Dataset for Medical "
    "Information Retrieval. <i>ECIR</i>.",
    "Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and "
    "Individual Rank Learning Methods. <i>SIGIR</i>.",
    "Gao, L., Dai, Z., Callan, J. (2021). Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline. "
    "<i>ECIR</i>.",
    "Smucker, M. D., Allan, J., Carterette, B. (2007). A Comparison of Statistical Significance Tests for "
    "Information Retrieval Evaluation. <i>CIKM</i>.",
    "Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for "
    "Zero-shot Evaluation of Information Retrieval Models. <i>NeurIPS Datasets and Benchmarks</i>.",
    "Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., Hajishirzi, H. (2020). Fact or Fiction: "
    "Verifying Scientific Claims. <i>EMNLP</i>. Annotations: SciFact release, AllenAI.",
    "Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F. (2024). Multilingual E5 Text Embeddings: A "
    "Technical Report. <i>arXiv:2402.05672</i>.",
]:
    P(ref, "ref")


# ============================================================================= page template
def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(M, 13 * mm, W - M, 13 * mm)
    canvas.setFont("Sans", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(M, 9 * mm, "Re-ranking: little gain, real risks  ·  M. Hamdi")
    canvas.drawRightString(W - M, 9 * mm, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=17 * mm, bottomMargin=19 * mm,
                      title="Re-ranking: little gain, real risks", author="Mustapha HAMDI",
                      subject="Cross-encoder re-ranking on top of hybrid retrieval in biomedical search")
doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(M, 19 * mm, TW, H - 36 * mm, id="f")],
                                   onPage=on_page)])
doc.build(story)
print(f"wrote {OUT}")
