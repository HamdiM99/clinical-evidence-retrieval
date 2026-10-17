"""Build the article PDF from docs/article/figures (run make_figures.py first). Every number comes from
figures/numbers.json, itself computed from the experiment artifacts.

Usage: python docs/article/build_pdf.py  ->  docs/article/reranker-article.pdf
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
OUT = HERE / "reranker-article.pdf"

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
hyb = {d: N[f"{d}/hybrid"] for d in (nf, sf)}
bm = {d: N[f"{d}/bm25"] for d in (nf, sf)}
den = {d: N[f"{d}/dense"] for d in (nf, sf)}
mini = {d: N[f"{d}/hybrid-minilm-top20"] for d in (nf, sf)}
c2 = {d: N[f"{d}/hybrid-c2-top20"] for d in (nf, sf)}
fo = N["forest"]
ts = N["train_stats"]
assert ts["datasets"]["scifact"]["queries"] == 608
gpu = {(r["model"], r["precision"], r["passages"], r["batch"]): r for r in N["gpu"]}
mini16 = {n: gpu[("ms-marco-MiniLM-L-6-v2", "fp16", n, n)] for n in (20, 50, 100)}
mini32 = {n: gpu[("ms-marco-MiniLM-L-6-v2", "fp32", n, n)] for n in (20, 50, 100)}
bge16 = {n: gpu[("bge-reranker-base", "fp16", n, n)] for n in (20, 50, 100)}
cpu = {(r["backend"], r["passages"], r["batch_size"]): r for r in N["cpu"]}
cpu_int8_20 = cpu[("ONNX Runtime int8 (dynamic)", 20, 20)]
cpu_fp32_20 = cpu[("PyTorch fp32", 20, 20)]
ab = N["ablation"]
tn = N["topn"]
ra = N["recall_at"]

# ============================================================================= front matter
P("TECHNICAL NOTE  ·  MEDICAL INFORMATION RETRIEVAL  ·  RE-RANKING", "kicker")
P("When the Re-ranker Makes Things Worse", "title")
P("Fine-tuning a cross-encoder for medical evidence retrieval: an ablation showing that label quality, "
  "not model or learning rate, decides the outcome — and what it costs to serve.", "subtitle")
P("Mustapha HAMDI  ·  October 2026  ·  Code, configs and run files: <i>clinical-evidence-retrieval</i> "
  "(GitHub link added on publication)", "byline")

tldr = [
    f"An off-the-shelf MS MARCO cross-encoder placed after a strong hybrid retriever <b>lowers</b> nDCG@10 on "
    f"SciFact ({sg(fo[f'{sf}/minilm-hybrid']['delta'])}, {pv(fo[f'{sf}/minilm-hybrid']['p'])}).",
    "A first fine-tuning on our own hard negatives never beats the base model. Changing the learning rate does not "
    "help either; switching to a listwise loss helps a little.",
    f"The cause is the training labels. Trained on {ts['datasets']['scifact']['queries']} clean in-domain claims "
    "only, with a "
    f"listwise loss, the re-ranker fixes the regression: {sg(fo[f'{sf}/c2-minilm']['delta'])} nDCG@10 vs the "
    f"off-the-shelf model on test ({pv(fo[f'{sf}/c2-minilm']['p'])}), in 3 minutes of GPU training.",
    "No re-ranker beats the hybrid first stage significantly on these benchmarks; on NFCorpus the first stage "
    f"caps what any re-ranker can do (Recall@100 = {f3(hyb[nf]['recall@100'])}).",
    f"Serving: in fp16 on a T4 the model re-ranks 20 passages in {mini16[20]['p95']:.0f} ms p95, inside a 100 ms "
    f"budget; a laptop CPU needs ≈ {cpu_int8_20['p95_ms'] / 1000:.1f} s even in int8. At GPU speed, tokenization "
    "becomes half of the cost.",
]
tl = [Paragraph("TL;DR", S["box_title"])] + [Paragraph(x, S["tldr"], bulletText="•") for x in tldr]
t = Table([[tl]], colWidths=[TW])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
                       ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                       ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
story.append(t)
story.append(Spacer(1, 6))

# ============================================================================= 1
H1("1. Why re-ranking matters in medical retrieval")
P("A clinical assistant that answers from guidelines and studies is only as good as the passages it retrieves: "
  "if the right recommendation is not in the context, no language model can recover it, and if a weak source is "
  "ranked first, it will be cited confidently. Production retrieval is therefore usually a <b>two-stage</b> "
  "pipeline (Figure 1). A fast first stage searches the whole collection and returns about a hundred candidates; "
  "a slower, more accurate model then <i>re-ranks</i> the best of them.")
figure("f1_pipeline", "The pipeline studied here. Offline, documents are chunked, embedded and indexed once. "
       "Online, BM25 and vector search run in parallel, their lists are merged with Reciprocal Rank Fusion, and a "
       "cross-encoder re-orders the top 20. For evaluation, the fused top 100 is exported once and every re-ranker "
       "is replayed on exactly the same candidates.")
callout("Bi-encoder vs cross-encoder, in one paragraph", [
    "A <b>bi-encoder</b> (the embedding model of the first stage) turns the query and each passage into vectors "
    "<i>separately</i>; passages are encoded once at indexing time and search is a nearest-neighbour lookup, which "
    "scales to millions of documents. A <b>cross-encoder</b> reads the query and one passage <i>together</i>, so "
    "every word of the query can attend to every word of the passage. It is much more accurate at judging "
    "relevance, but it must run once per (query, passage) pair at query time: affordable for 20 candidates, not "
    "for a whole corpus. That is why it sits in second position."])
P("Off-the-shelf cross-encoders are typically trained on MS MARCO, a large set of web search questions. The "
  "question this note answers is practical: <i>does such a model help on medical evidence, can we make it better "
  "with our own data, and what does it cost to run?</i>")

# ============================================================================= 2
H1("2. Setup: data, first stage and an evaluation you can trust")
H2("Two public benchmarks")
P(f"<b>NFCorpus</b> (Boteva et al., 2016): nutrition and medical questions written for the public, linked to "
  f"PubMed articles; about 40 relevant documents per query, with graded relevance. <b>SciFact</b> (Wadden et al., "
  f"2020): scientific <i>claims</i> (\"Vitamin D supplementation reduces fracture risk\") paired with the abstracts "
  f"that support or refute them; usually a single relevant abstract. Both come from the BEIR benchmark "
  f"(Thakur et al., 2021) with official train/dev/test splits. Test sets have {N['n_queries'][nf]} and "
  f"{N['n_queries'][sf]} queries; indexes hold {N['chunks'][nf]:,} and {N['chunks'][sf]:,} chunks.")
H2("A hybrid first stage")
P("Documents are split into chunks of about 250 words and indexed in Elasticsearch twice over: as text for BM25 "
  "(keyword matching) and as vectors from <i>multilingual-e5-base</i> (semantic matching, Wang et al., 2024). For "
  "each query both lists are retrieved in one round trip, each chunk is mapped to its document, and the two "
  "rankings are merged with Reciprocal Rank Fusion (Cormack et al., 2009), which only uses ranks and therefore "
  "needs no score calibration:")
equation("eq_rrf", TW * 0.62)
figure("f2_baselines", "First-stage results on the test sets. Hybrid fusion beats both of its components on "
       "both datasets and both metrics; it is the baseline every re-ranker has to beat.")
callout("Reading the metrics", [
    "<b>nDCG@10</b> scores the first ten results: a relevant document counts more the higher it is ranked "
    "(logarithmic discount), and the score is normalised by the best possible ranking, so 1.0 is perfect.",
    "<b>Recall@100</b> is the share of relevant documents present anywhere in the top 100: what the second stage "
    "could possibly promote. <b>MRR@10</b> looks only at the rank of the first relevant result."])
equation("eq_ndcg", TW * 0.5)
H2("Protocol")
bullets([
    "<b>Replay, not re-run.</b> The fused top 100 of every query is exported once; every re-ranker, depth and "
    "checkpoint is then scored on identical candidates, without Elasticsearch. Replaying the export without "
    "re-ranking reproduces the first-stage runs exactly, a built-in consistency check.",
    "<b>Validation for decisions, test once.</b> Depths, checkpoints and model choices are made on NFCorpus dev and "
    "on 200 held-out SciFact training claims; each test set is evaluated once per final system, with the decision "
    "rule written down before looking.",
    "<b>Paired significance.</b> Two systems are compared query by query with a paired permutation test "
    "(10,000 permutations; Smucker et al., 2007) and 95% bootstrap confidence intervals, not by comparing means.",
])
callout("Why a permutation test?", [
    "With 300 queries, a difference of 0.01 in mean nDCG@10 can easily be luck: a handful of queries flipping "
    "decides it. The permutation test asks how often a difference at least this large appears if we randomly swap "
    "the two systems' scores within each query. A small p-value means the observed gap is unlikely under "
    "\"no difference\". It is paired, so the large variance between easy and hard queries cancels out."])

# ============================================================================= 3
H1("3. Surprise #1: the off-the-shelf re-ranker hurts")
P(f"We first re-rank with <i>cross-encoder/ms-marco-MiniLM-L-6-v2</i>, a popular 22M-parameter model. The depth "
  f"(how many fused candidates to re-score) is tuned on validation (Figure 3, right): re-ranking more never helps, "
  f"and on SciFact every depth is <i>below</i> the first stage ({f3(tn[sf]['20'])} at top 20 vs "
  f"{f3(tn[sf]['fused'])} without re-ranking). We keep top 20 and evaluate once on test (Table 1).")
figures_side(("f3_recall_ceiling", 0.47), ("f4_topn", 0.47),
             f"Left: share of relevant documents found within the first k candidates (test). On NFCorpus only "
             f"{ra[nf]['20'] * 100:.0f}% of relevant documents are in the re-ranked window and "
             f"{ra[nf]['100'] * 100:.0f}% in the top 100: a hard ceiling for any re-ranker. SciFact reaches "
             f"{ra[sf]['20'] * 100:.0f}% at 20. Right: effect of the re-ranking depth for the off-the-shelf model "
             f"on the validation sets.")
table(["Test, nDCG@10", "BM25", "Dense", "Hybrid RRF", "+ MiniLM (MS MARCO)", "Δ vs hybrid"],
      [["NFCorpus", f3(bm[nf]["ndcg@10"]), f3(den[nf]["ndcg@10"]), f3(hyb[nf]["ndcg@10"]), f3(mini[nf]["ndcg@10"]),
        f"{sg(fo[f'{nf}/minilm-hybrid']['delta'])} ({pv(fo[f'{nf}/minilm-hybrid']['p'])})"],
       ["SciFact", f3(bm[sf]["ndcg@10"]), f3(den[sf]["ndcg@10"]), f3(hyb[sf]["ndcg@10"]), f3(mini[sf]["ndcg@10"]),
        f"{sg(fo[f'{sf}/minilm-hybrid']['delta'])} ({pv(fo[f'{sf}/minilm-hybrid']['p'])})"]],
      [0.17, 0.11, 0.11, 0.15, 0.2, 0.26], "Off-the-shelf re-ranking on test (top 20 re-ranked).")
P("Why would a strong re-ranker make things worse? Two reasons that compound. First, the baseline is already "
  "high: the BEIR authors report gains of about +0.02 on these two datasets for an MS MARCO cross-encoder "
  "re-ranking <i>BM25</i>, but our hybrid "
  f"first stage is already {sg(hyb[sf]['ndcg@10'] - bm[sf]['ndcg@10'])} above BM25 on SciFact. Second, the "
  "task is different. MS MARCO teaches \"does this passage answer the question?\"; SciFact asks \"is this the "
  "evidence for or against this claim?\". Claims are declarative, often negated (\"… does not cause …\"), and the "
  "relevant abstract may well <i>contradict</i> them, something a question-answering model is not trained to "
  "reward. Negated claims indeed lose the most in our per-query analysis.")

# ============================================================================= 4
H1("4. Surprise #2: fine-tuning on our own data does not help either")
P("The natural fix is to fine-tune the re-ranker on the training splits of the two datasets, with <b>hard "
  "negatives</b> mined by our own first stage: non-relevant documents that the retriever ranks high, precisely "
  "the mistakes the re-ranker must learn to correct (Figure 4). On NFCorpus, whose judgments are incomplete, the "
  "top three unjudged documents are not used as negatives, because they are often relevant but unlisted.")
figure("f5_pairs", "Building training pairs for one query from the first-stage ranking (illustrative).")
P(f"The resulting set has {ts['train_pairs']:,} (query, passage, label) pairs, {ts['positives_share'] * 100:.0f}% "
  "positive. Training queries whose text is identical to a test or validation query were removed (a few exist "
  "across BEIR splits). We train with the standard pointwise binary cross-entropy:")
equation("eq_bce", TW * 0.66)
P("where s is the model's score for the pair and y its 0/1 label. Figure 5 shows what happens on validation: the "
  "model first <i>forgets</i> (both datasets drop sharply), then NFCorpus slowly recovers to the base level while "
  "SciFact never does. No checkpoint beats the starting model, so the training script, which keeps the best "
  "validation checkpoint and starts from the base model, simply returns the base model.")
figure("f6_learning_curves", "Validation nDCG@10 during fine-tuning (Kaggle T4). Dotted: the off-the-shelf model "
       "(step 0). Dash-dotted: the first stage without re-ranking. Pointwise runs (orange) degrade, with a high or a "
       "low learning rate; the listwise run on SciFact only (green) improves and stays there.")

# ============================================================================= 5
H1("5. The ablation: what actually matters")
P("A failed run has many possible causes. We changed one factor at a time, with the same data, validation sets and "
  "selection rule (Figure 6).")
bullets([
    "<b>Learning rate (run A, 4× lower).</b> The collapse on SciFact is only delayed, from step 500 to about 750. "
    "Not the cause.",
    "<b>Loss (run B).</b> A re-ranker only needs to <i>order</i> the documents of one query, not to predict an "
    "absolute probability. The listwise (\"localised contrastive\") loss of Gao et al. (2021) compares one positive "
    "against seven hard negatives of the same query with a softmax:",
])
equation("eq_listwise", TW * 0.6)
bullets([
    f"With the same data, run B is better than the pointwise runs (SciFact {sg(ab[sf]['B'])} over the base model) "
    f"but still loses on NFCorpus ({sg(ab[nf]['B'])}). Helps, but not enough.",
    "<b>Data (runs C1, C2).</b> Trained on NFCorpus alone, the listwise model never improves, not even on "
    f"NFCorpus. Trained on SciFact alone (608 claims, 516 steps, 3 minutes), it improves SciFact by "
    f"{sg(ab[sf]['C2'])} and leaves NFCorpus untouched ({sg(ab[nf]['C2'])}).",
])
figure("f7_ablation", "Best validation checkpoint of each fine-tuning run, relative to the off-the-shelf model. "
       "Runs 1, A and C1 never beat it. Only the listwise loss on clean in-domain labels (C2) improves without "
       "hurting the other dataset.")
P("The diagnosis is the labels. NFCorpus judgments are derived from citation links between a health website and "
  "PubMed: about forty \"relevant\" documents per query, many only loosely on topic. For a model that already "
  "knows how to rank, these broad labels contain little new information, and they pull it away from the precise "
  "evidence matching SciFact needs. Mixing them in degrades SciFact; removing them is what works. More data was "
  "not better: 83,604 weak pairs did harm, 608 clean ones helped.")

# ============================================================================= 6
H1("6. Test results, with uncertainty")
P("The fine-tuned model, called C2, was then evaluated once on both test sets with the rule fixed beforehand: keep "
  "it if it is not significantly worse than the first stage on either dataset and it beats the off-the-shelf model "
  "on SciFact. Figure 7 shows each difference with its 95% confidence interval.")
figure("f8_forest", "Paired differences in nDCG@10 on test. An interval that crosses zero means the data cannot "
       "rule out \"no difference\". p-values from paired permutation tests.")
table(["Test", "Hybrid RRF", "+ MiniLM", "+ C2 (fine-tuned)", "C2 vs MiniLM", "C2 vs hybrid"],
      [["NFCorpus nDCG@10", f3(hyb[nf]["ndcg@10"]), f3(mini[nf]["ndcg@10"]), f3(c2[nf]["ndcg@10"]),
        f"{sg(fo[f'{nf}/c2-minilm']['delta'])} ({pv(fo[f'{nf}/c2-minilm']['p'])})",
        f"{sg(fo[f'{nf}/c2-hybrid']['delta'])} ({pv(fo[f'{nf}/c2-hybrid']['p'])})"],
       ["SciFact nDCG@10", f3(hyb[sf]["ndcg@10"]), f3(mini[sf]["ndcg@10"]), f3(c2[sf]["ndcg@10"]),
        f"<b>{sg(fo[f'{sf}/c2-minilm']['delta'])} ({pv(fo[f'{sf}/c2-minilm']['p'])})</b>",
        f"{sg(fo[f'{sf}/c2-hybrid']['delta'])} ({pv(fo[f'{sf}/c2-hybrid']['p'])})"],
       ["NFCorpus MRR@10", f3(hyb[nf]["mrr@10"]), f3(mini[nf]["mrr@10"]), f3(c2[nf]["mrr@10"]), "", ""],
       ["SciFact MRR@10", f3(hyb[sf]["mrr@10"]), f3(mini[sf]["mrr@10"]), f3(c2[sf]["mrr@10"]), "", ""]],
      [0.2, 0.13, 0.12, 0.17, 0.19, 0.19], "Final comparison on test (top 20 re-ranked).", highlight=[2])
P("Two conclusions, and it matters to state both. Fine-tuning on a small clean in-domain set <b>does fix</b> the "
  "domain mismatch of the off-the-shelf model, with an interval well clear of zero. But <b>no re-ranker beats the "
  "hybrid first stage significantly</b>: on NFCorpus the gain is about +0.01 with an interval touching zero, on "
  "SciFact C2 is level with the first stage. With 300 queries, effects of this size are below what the "
  "protocol can detect, and on NFCorpus the re-ranker can only reorder the "
  f"{ra[nf]['20'] * 100:.0f}% of relevant documents that the first stage put in its top 20.")

# ============================================================================= 7
H1("7. What it costs to serve")
P("A re-ranker earns its place only if it fits the latency budget: here p95 &lt; 300 ms end to end, of which about "
  "100 ms for re-ranking. We measured C2's architecture on a laptop CPU (exported to ONNX, fp32 and dynamically "
  "quantized int8) and on an NVIDIA T4 GPU (fp32 and fp16), with real passages, warm-up excluded (Figure 8).")
figures_side(("f9_latency", 0.62), ("f10_breakdown", 0.34),
             "Left: p95 latency per query against the number of passages re-ranked (log scale). Right: on the GPU "
             "in fp16, tokenization on the CPU is as large as the model itself.")
bullets([
    f"<b>GPU fp16 fits with margin</b>: {mini16[20]['p95']:.0f} ms p95 for 20 passages, {mini16[50]['p95']:.0f} ms "
    f"for 50; fp16 is 2–4× faster than fp32 on the T4 ({mini32[20]['p95']:.0f} ms at 20).",
    f"<b>Int8 on CPU</b> keeps ranking quality on validation and cuts the model 4× in size, but a 15 W laptop still "
    f"needs ≈ {cpu_int8_20['p95_ms']:.0f} ms p95 for 20 passages ({cpu_fp32_20['p95_ms']:.0f} ms in fp32): a "
    "fallback, not a serving target.",
    f"<b>The larger bge-reranker-base</b> fits only at depth 20 in fp16 ({bge16[20]['p95']:.0f} ms) and costs about "
    "3× MiniLM: it would need a clear quality gain to justify it.",
    f"<b>Tokenization becomes the bottleneck</b>: {mini16[20]['tok']:.0f} of {mini16[20]['p50']:.0f} ms at depth 20. "
    "Passages can be tokenized once at indexing time and stored with the chunk, so only the query is tokenized "
    "online.",
])

# ============================================================================= 8
H1("8. Lessons")
bullets([
    "<b>Measure the re-ranker against the first stage, not against nothing.</b> A strong hybrid retriever leaves "
    "little room, and an off-the-shelf model can lose it.",
    "<b>Label quality beats label quantity.</b> 608 clean, task-matched examples helped; 83,604 broad ones hurt. In "
    "a clinical setting this argues for small clinician-validated sets over large weak ones.",
    "<b>Use a ranking objective.</b> A listwise loss over a query's own hard negatives fits what a re-ranker does.",
    "<b>Ablate one factor at a time and keep failed runs.</b> Without runs A, B and C1, the conclusion would have "
    "been \"lower the learning rate\", which is wrong.",
    "<b>Recall is the ceiling.</b> When the first stage misses 70% of the relevant documents, the next gains come "
    "from query processing and candidate generation, not from a better re-ranker.",
    "<b>Precision and tokenization are serving levers.</b> fp16 on GPU, int8 on CPU, and pre-tokenized passages.",
])
H2("Limitations")
P("Two small English benchmarks; labels from benchmark judgments, not clinicians; one training seed per run; "
  "laptop-scale indexes (thousands of chunks), so first-stage latency here says nothing about 100M-document "
  "behaviour; GPU throughput is extrapolated, not load-tested. French queries and guideline documents are the "
  "next steps of the project.")
H2("Reproducibility")
P("Every number and figure in this note is generated from run files by <font face='Mono'>docs/article/"
  "make_figures.py</font>. Commands: <font face='Mono'>cer candidates</font> (export first-stage candidates), "
  "<font face='Mono'>cer rerank</font> (replay a re-ranker), <font face='Mono'>scripts/train_reranker.py</font> "
  "(fine-tuning, listwise or pointwise), <font face='Mono'>cer export-onnx</font> and "
  "<font face='Mono'>scripts/bench_reranker*.py</font> (serving cost).", "body_left")

H1("References")
for r in [
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
    "Verifying Scientific Claims. <i>EMNLP</i>.",
    "Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F. (2024). Multilingual E5 Text Embeddings: A "
    "Technical Report. <i>arXiv:2402.05672</i>.",
]:
    P(r, "ref")


# ============================================================================= page template
def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(M, 13 * mm, W - M, 13 * mm)
    canvas.setFont("Sans", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(M, 9 * mm, "When the Re-ranker Makes Things Worse  ·  M. Hamdi")
    canvas.drawRightString(W - M, 9 * mm, str(doc.page))
    if doc.page == 1:
        canvas.setFillColor(ACCENT)
        canvas.rect(0, H - 6 * mm, W, 6 * mm, stroke=0, fill=1)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=17 * mm, bottomMargin=19 * mm,
                      title="When the Re-ranker Makes Things Worse", author="Mustapha HAMDI",
                      subject="Fine-tuning a cross-encoder for medical evidence retrieval")
doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(M, 19 * mm, TW, H - 36 * mm, id="f")],
                                   onPage=on_page)])
doc.build(story)
print(f"wrote {OUT}")
