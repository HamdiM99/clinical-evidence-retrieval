import json

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cer.config import RerankConfig
from cer.rerank import build_reranker
from cer.rerank.onnx import export_onnx, quantize_onnx

WORDS = ["vitamin", "d", "fracture", "risk", "older", "adults", "fiber", "colon", "cancer", "does", "prevent",
         "aspirin", "stroke", "trial", "evidence"]


@pytest.fixture
def tiny_model(tmp_path):
    """A 2-layer random BERT cross-encoder with a local vocabulary: no download."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    cfg = transformers.BertConfig(vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=64, num_labels=1)
    torch.manual_seed(0)
    model = transformers.BertForSequenceClassification(cfg).eval()
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out, model, tok


def test_onnx_fp32_matches_torch_and_int8_runs(tmp_path, tiny_model):
    model_dir, model, tok = tiny_model
    fp32 = export_onnx(model_dir, tmp_path / "onnx")
    int8 = quantize_onnx(fp32, tmp_path / "onnx-int8")
    assert json.loads((int8 / "export_info.json").read_text(encoding="utf-8"))["precision"] == "int8-dynamic"
    assert (int8 / "model.onnx").stat().st_size < (fp32 / "model.onnx").stat().st_size

    query = "does vitamin d prevent fracture"
    passages = ["vitamin d fracture risk older adults", "fiber colon cancer", "aspirin stroke trial evidence",
                "d"]  # different lengths: exercises padding and dynamic axes
    with torch.no_grad():
        enc = tok([query] * len(passages), passages, padding=True, return_tensors="pt")
        expected = model(**enc).logits.view(-1).numpy()
    got = build_reranker(RerankConfig(type="onnx", model=str(fp32), batch_size=3, max_length=32)).score(query, passages)
    np.testing.assert_allclose(got, expected, atol=1e-4)

    q8 = build_reranker(RerankConfig(type="onnx", model=str(int8), batch_size=2, threads=1)).score(query, passages)
    assert len(q8) == len(passages) and np.all(np.isfinite(q8))
