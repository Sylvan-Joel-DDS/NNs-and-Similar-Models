"""Fine-tuning data and loop for the instruction generator (flan-t5-base)."""
from __future__ import annotations
import random, time

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


def finetune_generator(gen, train_ex, val_ex, epochs=2, lr=None, batch_size=2, grad_accum=4, optimizer="adafactor",
                       freeze_encoder=True, time_budget_min=None, log_every=10, verbose=True):
    """Memory-lean fine-tuning for a laptop CPU.

    * optimizer='adafactor' keeps almost no optimizer state (AdamW needs 2 extra copies of every weight)
    * freeze_encoder=True trains only the decoder (about half the backward work and memory)
    * batch_size x grad_accum gives the same effective batch with a fraction of the activation memory
    * the best weights are saved to disk instead of being cloned in RAM every epoch
    * progress is printed every `log_every` batches, and `time_budget_min` stops training cleanly
      (keeping the best weights so far) instead of running for hours
    """
    import gc, torch
    from torch.utils.data import DataLoader
    from transformers.optimization import Adafactor
    gen._ensure(); model, tok, device = gen.model, gen.tok, gen.device
    lr = lr or (1e-3 if optimizer == "adafactor" else 3e-4)

    if freeze_encoder:
        for p in list(model.shared.parameters()) + list(model.encoder.parameters()):
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"Training {sum(p.numel() for p in params) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters "
              f"| optimizer {optimizer} | effective batch {batch_size * grad_accum} | device {device}")

    def collate(batch):
        enc = tok([b["prompt"] for b in batch], max_length=512, truncation=True, padding=True, return_tensors="pt")
        lab = tok([b["target"] for b in batch], max_length=256, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    tr = DataLoader(train_ex, batch_size=batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(val_ex, batch_size=4, collate_fn=collate)
    opt = (Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
           if optimizer == "adafactor" else torch.optim.AdamW(params, lr=lr, weight_decay=0.01))
    updates = epochs * -(-len(tr) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.1, 1 - s / max(1, updates)))

    t0, best, log, saved, stop = time.time(), float("inf"), [], False, False
    for ep in range(epochs):
        model.train(); losses = []
        for i, b in enumerate(tr):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            (loss / grad_accum).backward(); losses.append(loss.item())
            if (i + 1) % grad_accum == 0 or i + 1 == len(tr):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if verbose and (i + 1) % log_every == 0:
                print(f"  epoch {ep + 1}/{epochs} batch {i + 1}/{len(tr)} | loss {np.mean(losses[-log_every:]):.3f} | {time.time() - t0:.0f}s")
            if time_budget_min and time.time() - t0 > time_budget_min * 60:
                stop = True; print(f"  time budget of {time_budget_min} min reached - stopping."); break
        model.eval(); vl = []
        with torch.no_grad():
            for b in va:
                vl.append(model(**{k: v.to(device) for k, v in b.items()}).loss.item())
        log.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), val_loss=float(np.mean(vl)), seconds=round(time.time() - t0)))
        if verbose: print(log[-1])
        if np.mean(vl) < best or not saved:            # save to DISK (no extra copy of the weights in RAM)
            best = float(np.mean(vl)); gen.path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(gen.path); tok.save_pretrained(gen.path); saved = True
        if stop:
            break
    del opt, sched; gc.collect()
    config.write_meta(gen.path / "meta", fingerprint=_fp(gen, train_ex, epochs))
    gen.load()                                         # re-load the best saved weights
    return pd.DataFrame(log)


def _fp(gen, train_ex, epochs):
    return config.fingerprint([e["prompt"] for e in train_ex], epochs, gen.base_model)


def generator_is_current(gen, train_ex, epochs=2) -> bool:
    """True if the fine-tuned folder was trained on exactly this data / base model / epochs (skip re-training)."""
    return gen.is_finetuned and config.read_meta(gen.path / "meta").get("fingerprint") == _fp(gen, train_ex, epochs)


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
