"""Build every cached artefact from the command line (no notebook needed):

    python -m Pipeline.run_pipeline --stage all          # images -> cnn -> text -> rag
    python -m Pipeline.run_pipeline --stage text         # one stage
    python -m Pipeline.run_pipeline --stage cnn --full   # --full = the complete (slower) search / fine-tune schedule
    python -m Pipeline.run_pipeline --stage rag --skip-generator   # policy index + extractive cache only (no PyTorch)

Run from the project root. Re-running is cheap: finished stages are loaded from cache.
"""
from __future__ import annotations
import argparse, time

from . import config


def stage_images(args):
    from .images import get_image_data
    data = get_image_data(force=args.force)
    print(data.summary()); return data


def stage_cnn(args):
    from . import cnn
    data = stage_images(args)
    feats = cnn.get_feature_set(data, n_aug=1)
    space = [dict(units=u, dropout=d, l2=l2, lr=lr) for u in (128, 256, 512) for d in (0.3, 0.5) for l2 in (1e-4, 1e-3) for lr in (1e-3, 5e-4)]
    df, best = cnn.head_search(feats, space if args.full else space[:6])
    print(df.head()); print("best head:", best)
    head, _ = cnn.train_head(feats, best, epochs=40)
    cnn_head = cnn.build_cnn(**best, head=head)
    cnn.save_cnn(cnn_head, config.MODELS_DIR / "cnn_head.keras", data, stage="head")
    depths, epochs = ([30, 60], 12) if args.full else ([30], 6)
    results = []
    for d in depths:
        m, h, how = cnn.finetune_or_load(best, head, data, d, epochs, tag="fine-tune", force=args.force)
        pred = cnn.predict_probs(m, data.X_val).argmax(1)
        acc = float((pred == data.y_val).mean()); results.append((acc, d, m)); print(f"fine-tune top {d}: val acc {acc:.4f} ({how})")
    acc, d, m = max(results, key=lambda r: r[0])
    cnn.save_cnn(m, config.CNN_FINAL, data, depth=d, val_acc=acc)
    print("final CNN saved ->", config.CNN_FINAL)


def stage_text(args):
    from . import text_clf
    sp = text_clf.get_text_splits()
    art = text_clf.train_text_model(sp, force=args.force)
    from sklearn.metrics import accuracy_score
    print("test accuracy:", round(accuracy_score(sp["y_test"], art["model"].predict(sp["X_test"])), 4))


def stage_rag(args):
    from .policy_rag import build_knowledge_base, build_retriever, RecyclingAdvisor
    from .generator import Generator
    kb = build_knowledge_base(); print(f"{len(kb.chunks)} chunks | {len(kb.conflicts)} policy conflicts quarantined")
    retriever, emb = build_retriever(kb); print("embedder:", emb.name)
    gen = None if args.skip_generator else Generator()
    adv = RecyclingAdvisor(kb, retriever, gen, gen_kwargs=dict(num_beams=4, no_repeat_ngram_size=3, early_stopping=True, max_new_tokens=256))
    adv.warm_cache(); print("instruction cache written ->", config.INSTRUCTION_CACHE)


STAGES = dict(images=stage_images, cnn=stage_cnn, text=stage_text, rag=stage_rag)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--force", action="store_true", help="ignore caches and rebuild")
    ap.add_argument("--full", action="store_true", help="full (slower) CNN search / fine-tune schedule")
    ap.add_argument("--skip-generator", action="store_true", help="RAG stage without PyTorch (extractive answers only)")
    a = ap.parse_args()
    t0 = time.time()
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"\n===== {name} =====")
        STAGES[name](a)
    print(f"\nDone in {time.time() - t0:.0f}s")
