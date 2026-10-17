"""ONNX serving path for cross-encoders: export (fp32), dynamic int8 quantization, ONNX Runtime reranker.

A model directory produced here holds `model.onnx` + the tokenizer files, and is used with
`RerankConfig(type="onnx", model=<dir>)` (or `cer rerank --type onnx --model <dir>`).
Dynamic quantization stores weights as int8 and quantizes activations on the fly: no calibration data,
≈ 4× smaller weights, faster matrix products on CPUs with int8 instructions; its cost in ranking quality is
measured, not assumed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from cer.config import RerankConfig


def export_onnx(model_dir: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    """Export a Hugging Face sequence-classification cross-encoder to `out_dir/model.onnx` (fp32)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    dummy = tok(["what is the evidence"], ["a short passage about the evidence"], return_tensors="pt")
    names = list(dummy.keys())  # input_ids, token_type_ids (BERT), attention_mask

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *tensors):
            return self.m(**dict(zip(names, tensors))).logits

    axes = {n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(Logits(model), tuple(dummy[n] for n in names), str(out / "model.onnx"),
                          input_names=names, output_names=["logits"], dynamic_axes=axes,
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    tok.save_pretrained(out)
    (out / "export_info.json").write_text(json.dumps({"source": str(model_dir), "opset": opset,
                                                      "precision": "fp32", "inputs": names}, indent=2),
                                          encoding="utf-8")
    return out


def quantize_onnx(fp32_dir: str | Path, out_dir: str | Path) -> Path:
    """Dynamic int8 quantization (weights int8, activations quantized at run time)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, out = Path(fp32_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src / "model.onnx"), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    for f in src.iterdir():
        if f.name not in ("model.onnx", "export_info.json"):
            shutil.copy2(f, out / f.name)
    info = json.loads((src / "export_info.json").read_text(encoding="utf-8"))
    (out / "export_info.json").write_text(json.dumps({**info, "precision": "int8-dynamic", "fp32": str(src)},
                                                     indent=2), encoding="utf-8")
    return out


class OnnxReranker:
    def __init__(self, cfg: RerankConfig):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.name = cfg.model
        opts = ort.SessionOptions()
        if cfg.threads:
            opts.intra_op_num_threads = cfg.threads
        self.session = ort.InferenceSession(str(Path(cfg.model) / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.tok = AutoTokenizer.from_pretrained(cfg.model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), self.cfg.batch_size):
            batch = passages[i: i + self.cfg.batch_size]
            enc = self.tok([query] * len(batch), batch, truncation="longest_first", max_length=self.cfg.max_length,
                           padding=True, return_tensors="np")
            feeds = {n: enc[n].astype(np.int64) for n in self.inputs}
            out += self.session.run(["logits"], feeds)[0].reshape(-1).astype(float).tolist()
        return out
