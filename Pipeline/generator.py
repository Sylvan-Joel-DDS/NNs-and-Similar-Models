"""flan-t5 wrapper. PyTorch / transformers are imported lazily, so merely importing the Pipeline never needs them."""
from __future__ import annotations
from pathlib import Path

from . import config
from .policy_rag import format_instructions

DEFAULT_GEN_KWARGS = dict(max_new_tokens=256, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)


class Generator:
    """`base_model` defaults to config.GEN_BASE_MODEL (flan-t5-small). Each base model has its own fine-tuned folder
    (Models/flan_t5_recycling_small, ..._base) so they never overwrite each other."""
    def __init__(self, base_model: str | None = None, path=None):
        self.base_model = base_model or config.GEN_BASE_MODEL
        short = self.base_model.split("/")[-1].replace("flan-t5-", "")
        self.path = Path(path) if path else config.MODELS_DIR / f"flan_t5_recycling_{short}"
        self.tok = self.model = None
        self._ok = None

    @property
    def available(self) -> bool:
        if self._ok is None:
            try:
                import torch, transformers  # noqa: F401
                self._ok = True
            except Exception as e:
                print(f"[Generator] PyTorch/transformers unavailable ({type(e).__name__}) - extractive answers only.")
                self._ok = False
        return self._ok

    @property
    def signature(self) -> str:
        """Changes whenever the fine-tuned weights change -> invalidates the cached answers."""
        files = (list(self.path.glob("*.safetensors")) + list(self.path.glob("pytorch_model*.bin"))) if self.path.exists() else []
        return f"finetuned:{self.base_model}:{max(f.stat().st_mtime_ns for f in files)}" if files else f"base:{self.base_model}"

    @property
    def is_finetuned(self) -> bool:
        return self.signature.startswith("finetuned")

    def load(self, force_base: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        src = self.base_model if (force_base or not self.is_finetuned) else str(self.path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(src).to(self.device).eval()
        return self

    def _ensure(self):
        if self.model is None:
            self.load()

    def generate(self, prompt: str, model=None, **gen_kwargs) -> str:
        import torch
        self._ensure()
        model = model or self.model
        # a decoding config REPLACES the defaults (merging would leak beam-search options into sampling configs)
        kwargs = dict(gen_kwargs) if gen_kwargs else dict(DEFAULT_GEN_KWARGS)
        kwargs.setdefault("max_new_tokens", 256)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        return format_instructions(self.tok.decode(out[0], skip_special_tokens=True))
