"""Fine-tuning data and loop for the instruction generator (flan-t5-base)."""
from __future__ import annotations
import random

import numpy as np
import pandas as pd

from . import config
from .config import CATEGORIES
from .policy_rag import (build_prompt, compose_instructions, grounding_score, label_coverage, readability, pretty)

TRAIN_TEMPLATES = ["How to recycle {c}", "How do I recycle {c}?", "What is the correct way to dispose of {c}?",
                   "Recycling instructions for {c}", "Can you explain how to recycle {c} in Metro City?",
                   "I have {c} to throw away. What should I do?"]
VAL_TEMPLATES = ["Where does {c} go?", "Tell me the rules for {c} recycling"]


def make_examples(retriever, templates, n_per_cat, seed):
    """Simulate inference: several phrasings x several retrieval contexts. The TARGET is composed only from the
    chunks in that very context, so the model learns 'faithful rewrite of the retrieved policy'."""
    rng = random.Random(seed); out = []
    for cat in CATEGORIES:
        for _ in range(n_per_cat):
            q = rng.choice(templates).format(c=pretty(cat))
            ctx = retriever.retrieve_for_category(cat, q, k=rng.choice([7, 8, 9]))
            n_core = sum(c["authoritative"] for c in ctx)
            core, extra = ctx[:n_core], ctx[n_core:]
            rng.shuffle(extra); ctx = core + extra
            out.append(dict(category=cat, query=q, prompt=build_prompt(q, ctx), target=compose_instructions(cat, ctx, rng)))
    return out


def finetune_generator(gen, train_ex, val_ex, epochs=3, lr=2e-4, batch_size=4, verbose=True):
    """Fine-tune ALL weights. Keeps the best-validation-loss weights and saves them to Models/flan_t5_recycling."""
    import torch
    from torch.utils.data import DataLoader
    gen._ensure(); model, tok, device = gen.model, gen.tok, gen.device

    def collate(batch):
        enc = tok([b["prompt"] for b in batch], max_length=512, truncation=True, padding=True, return_tensors="pt")
        lab = tok([b["target"] for b in batch], max_length=256, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    tr = DataLoader(train_ex, batch_size=batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(val_ex, batch_size=8, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / total))
    best, best_state, log = float("inf"), None, []
    for ep in range(epochs):
        model.train(); losses = []
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); losses.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for b in va:
                vl.append(model(**{k: v.to(device) for k, v in b.items()}).loss.item())
        log.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), val_loss=float(np.mean(vl))))
        if verbose: print(log[-1])
        if np.mean(vl) < best:
            best = np.mean(vl); best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    config.GENERATOR_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.GENERATOR_DIR); tok.save_pretrained(config.GENERATOR_DIR)
    config.write_meta(config.GENERATOR_DIR / "meta", fingerprint=config.fingerprint([e["prompt"] for e in train_ex], epochs, lr))
    return pd.DataFrame(log)


def generator_is_current(train_ex, epochs=3, lr=2e-4) -> bool:
    """True if Models/flan_t5_recycling was trained on exactly this data/settings (skip re-training)."""
    meta = config.read_meta(config.GENERATOR_DIR / "meta")
    return meta.get("fingerprint") == config.fingerprint([e["prompt"] for e in train_ex], epochs, lr)


def evaluate_generator(gen, retriever, kb, gen_kwargs, label, model=None, templates=VAL_TEMPLATES):
    rows = []
    for cat in CATEGORIES:
        q = templates[0].format(c=pretty(cat))
        ctx = retriever.retrieve_for_category(cat, q)
        text = gen.generate(build_prompt(q, ctx), model=model, **gen_kwargs)
        g = grounding_score(text, ctx)
        rows.append(dict(category=cat, grounding=g["score"], label_coverage=label_coverage(text, cat, kb.auth),
                         words=len(text.split()), **readability(text), output=text))
    df = pd.DataFrame(rows); df.insert(0, "setting", label)
    return df
