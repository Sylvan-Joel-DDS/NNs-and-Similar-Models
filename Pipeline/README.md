# Pipeline

Reusable code for the EcoSort notebook (`../Notebook/waste_management_summative_completed.ipynb`).

| Module | Purpose |
|---|---|
| `config.py` | paths (`Data/`, `Models/`, `Data/cache/`), constants, cache fingerprints |
| `images.py` | RealWaste discovery, stratified 80/10/10 split, cached uint8 arrays |
| `cnn.py` | MobileNetV2 + head; cached frozen features; staged fine-tuning; evaluation |
| `text_clf.py` | cleaning, leak-free splits, TF-IDF classifier (joblib-cached) |
| `policy_rag.py` | policy parsing, conflict detection, retriever, grounded answers, `RecyclingAdvisor` |
| `generator.py` / `rag_train.py` | flan-t5 wrapper (lazy PyTorch) and its fine-tuning |
| `assistant.py` | `EcoSortAssistant`: image or text -> category -> instructions + policy documents |
| `run_pipeline.py` | build every cache from the terminal |

Run from the project root:

```
python -m Pipeline.run_pipeline --stage all                    # images -> cnn -> text -> rag
python -m Pipeline.run_pipeline --stage cnn --full             # long schedule (GPU recommended)
python -m Pipeline.run_pipeline --stage rag --skip-generator   # extractive answers, no PyTorch
```

Caches are keyed by a fingerprint of the data they came from; a model trained on a different split is
re-trained rather than silently re-used. Delete `Data/cache/` to force a rebuild.
