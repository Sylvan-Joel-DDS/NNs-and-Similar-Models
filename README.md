# EcoSort Waste Management Assistant

An integrated assistant for Metro City's waste department. A resident uploads a **photo** or types a **description**
of a waste item, and the assistant returns the waste category, how confident it is, step-by-step **recycling
instructions** grounded in the city's policy documents, and the policy documents that apply.

Summative lab: *NNs and Similar Models* (CNN, text classification, transformer + retrieval-augmented generation).

| Input | Model | Output |
|---|---|---|
| Image | MobileNetV2 transfer learning (CNN) | category + confidence |
| Text description | TF-IDF (word + character n-grams) + logistic regression | category + confidence |
| Category | MiniLM retrieval over policy chunks + fine-tuned `flan-t5-small` | instructions + policy documents |

---

## 1. Project layout

```
NNs and Similar Models/
├── README.md                 this file
├── requirements.txt          pip dependencies
├── requirements-lock.txt     exact versions of the environment the results came from
├── environment.yml           conda environment (wraps requirements.txt)
├── Data/
│   ├── RealWaste/            image dataset, one folder per class (download separately, see section 3)
│   ├── waste_descriptions.csv        5,000 generated waste descriptions with disposal notes
│   ├── waste_policy_documents.json   14 generated Metro City policy documents
│   ├── README.md             notes on the data
│   └── cache/                (auto-created) decoded image arrays, CNN features, embeddings - safe to delete
├── Models/                   (auto-created) trained artefacts - rebuilt by the pipeline if missing
│   ├── cnn_head.keras, cnn_ft30.keras, ecosort_cnn_final.keras
│   ├── text_model.joblib
│   ├── flan_t5_recycling_small/
│   └── instruction_cache.json
├── Notebook/
│   └── waste_management_summative_completed.ipynb     the report: explanations, plots, evaluation
└── Pipeline/                 reusable code imported by the notebook (must be submitted with it)
    ├── config.py             paths, constants, cache fingerprints
    ├── images.py             RealWaste discovery, stratified split, cached image arrays
    ├── cnn.py                MobileNetV2 model, cached frozen features, fine-tuning, evaluation
    ├── text_clf.py           text cleaning, leak-free splits, classifier (cached)
    ├── policy_rag.py         policy knowledge base, conflict detection, retriever, grounded answers
    ├── generator.py          flan-t5 wrapper (PyTorch is imported lazily)
    ├── rag_train.py          fine-tuning data and loop for the generator (low-memory)
    ├── assistant.py          EcoSortAssistant: image | text -> category -> instructions
    ├── make_lock.py          writes a clean requirements-lock.txt from the current environment
    └── run_pipeline.py       build every cache from the command line
```

The notebook is the readable report; the heavy logic lives in `Pipeline/` and is printed inside the notebook
with `show_source(...)`, so everything is visible in one place.

---

## 2. Setup

Requires **Python 3.10 - 3.12**. A GPU is *not* needed: the project is designed to run on a CPU-only laptop.

**Option A - conda (recommended)**

```bash
conda env create -f environment.yml
conda activate ecosort
```

**Option B - pip**

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows   (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only PyTorch first
pip install -r requirements.txt
```

Check the two deep-learning frameworks load (both must import cleanly in the same environment):

```bash
python -c "import torch, tensorflow as tf; print(torch.__version__, tf.__version__)"
```

`requirements.txt` gives minimum versions. `requirements-lock.txt` pins the versions of the environment in which the notebook
ran end to end (Python 3.11, Windows, CPU-only PyTorch); install it instead for an identical setup. To regenerate it from your
own working environment run `python -m Pipeline.make_lock` (plain `pip freeze` is not portable in a conda environment).

---

## 3. Data

* `waste_descriptions.csv` and `waste_policy_documents.json` are provided with the lab (note: the brief calls the
  policy file a CSV; it is JSON).
* **RealWaste images:** download from <https://archive.ics.uci.edu/dataset/908/realwaste>, unzip, and put the
  class folders under `Data/RealWaste/`:

```
Data/RealWaste/
├── Cardboard/   Food Organics/   Glass/   Metal/   Miscellaneous Trash/
└── Paper/       Plastic/         Textile Trash/   Vegetation/
```

The dataset has **9** classes (the brief's "8 classes" lists glass colours separately; here they are one `Glass`
class). A zip that nests the folders one or two levels deeper (e.g. `Data/RealWaste/RealWaste/Cardboard`) is found
automatically.

---

## 4. Running it

### In the notebook

1. Open `Notebook/waste_management_summative_completed.ipynb` and select the `ecosort` environment as the kernel.
2. **Restart the kernel and Run All.** The first run builds caches; later runs reload them.

Cells depend on earlier ones (for example Part 5 needs `img`, `head` and `cnn_model` from Parts 1-2). After a kernel
restart, run from the top or use *Run All Above* on the cell you want.

### From the command line (project root)

```bash
python -m Pipeline.run_pipeline --stage all                      # images -> cnn -> text -> lm -> rag
python -m Pipeline.run_pipeline --stage text                     # one stage
python -m Pipeline.run_pipeline --stage cnn --full               # long CNN schedule (GPU recommended)
python -m Pipeline.run_pipeline --stage lm --minutes 20          # fine-tune flan-t5-small, time-boxed
python -m Pipeline.run_pipeline --stage rag --skip-generator     # policy index + extractive answers, no PyTorch
```

Add `--force` to ignore caches and rebuild. `lm` is best run in a terminal (not the notebook) on a low-memory
machine, because it keeps TensorFlow and the image arrays out of RAM; the notebook then simply loads the saved model.

### Use the trained assistant in your own script

```python
from Pipeline.assistant import EcoSortAssistant, show_result

bot = EcoSortAssistant()          # loads Models/ecosort_cnn_final.keras, text_model.joblib, instruction_cache.json
show_result(bot("empty tin of beans", "text"))
show_result(bot("Data/RealWaste/Glass/Glass_12.jpg", "image"))     # any image path, PIL image or array
```

Every call returns the same dictionary: `status` (`ok` / `low_confidence` / `error`), `category`, `confidence`,
`top_predictions`, `instructions`, `policy_documents`, `grounding`, `generation_mode`, `warnings`, `latency_sec`.
Failures are returned as `status = error`, never raised. Components load lazily: the text path needs only
scikit-learn, the image path imports TensorFlow on first use, and PyTorch is only needed for answers that are not
already cached.

---

## 5. What makes it fast (and robust)

| Slow step | How it is handled |
|---|---|
| Decoding ~4.7k JPEGs | decoded once (multi-threaded) and cached as `.npy` in `Data/cache` |
| CNN head search | MobileNetV2 features are computed once and cached; the search then runs on 1280-d vectors (seconds per configuration) |
| Text grid search | trained once, cached in `Models/text_model.joblib` |
| CNN fine-tuning, generator fine-tuning | trained once, reloaded afterwards |
| Generating the 9 answers | cached in `Models/instruction_cache.json`; later runs are instant and need no PyTorch |

Every saved model stores a **fingerprint** of the data it was trained on and is reloaded only if it still matches,
so a stale model is retrained instead of being silently evaluated on a different split.

Low-resource switches: `FAST_MODE` (CNN fine-tune: one depth, few epochs) and `QUICK_SEARCH` (4 of 12 head
configurations) in the notebook; set both to `False` on a faster machine. The generator is `flan-t5-small`; set the
environment variable `ECOSORT_GEN_MODEL=google/flan-t5-base` to use the larger model (needs a GPU or 16 GB+ RAM).

---

## 6. Results (from the saved notebook run)

| Component | Metric | Result |
|---|---|---|
| CNN, frozen backbone | validation accuracy / macro-F1 | 0.848 / 0.847 |
| CNN, fine-tuned (top 30 layers) | validation accuracy | 0.865 |
| CNN, fine-tuned | **test** accuracy / macro-F1 | **0.851 / 0.851** (weakest class: Miscellaneous Trash, F1 0.705) |
| CNN, confidence threshold 0.60 | validation | 0.918 accuracy on the 87% of requests answered |
| Text classifier | **test** accuracy / macro-F1 | **0.9987 / 0.9987** (1 error in 741) |
| Retrieval, free text only | top-1 category accuracy | 0.43 - 0.49 (the assistant retrieves within the predicted category instead) |
| Generator, zero-shot to fine-tuned | section coverage / grounding | 0.037 to 0.97+ / 0.876 to 0.998 |
| Final instructions (9 categories) | mean grounding / fallback used | 0.999 / 0 of 9 |
| End-to-end, 100 test images | accuracy | 0.850 (0.963 when confident, 0.333 when flagged low-confidence) |
| End-to-end, 200 test descriptions | accuracy | 1.000; median latency 0.20 s (cached answers) |

The text scores are inflated by template-generated descriptions; see the notebook's error analysis and limitations.

### Findings about the provided data that shaped the design

1. **Conflicting policies.** Bundled programme documents list *plastic bags and film* and *window glass or mirrors*
   as accepted, while the category-specific guidelines reject them, and the bundled documents are newer. The system
   treats the category-specific guideline as authoritative, ignores the conflicting items and flags them in the
   returned policy documents. Metro City should resolve this before deployment.
2. **Starter split leaked.** The starter's shuffled `take`/`skip` validation/test split overlaps. `Pipeline/images.py`
   makes a stratified, deterministic 80/10/10 split by image with an integrity check.
3. **Factual safeguard.** Generated instructions are verified against the retrieved policy text; if unsupported,
   the policy text itself is returned (demonstrated in notebook section 4.4).

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `OSError: [WinError 127] ... torch\lib\shm.dll` | Import PyTorch before TensorFlow (the notebook does), install the Microsoft Visual C++ Redistributable (x64), or reinstall CPU PyTorch: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu` |
| Part 4 runs without a language model | PyTorch failed to load; the notebook falls back to extractive answers (policy text). Fix PyTorch and re-run Part 4 |
| `NameError` for `kb`, `img`, `sp`, ... | A cell was run out of order or after a kernel restart; run from the top or *Run All Above* |
| Hugging Face "Retrying" / "unauthenticated requests" messages | Harmless. After the first download, set `HF_HUB_OFFLINE=1` to stop the retries |
| "Could not render content for application/vnd.jupyter.widget-view+json" | Cosmetic (download progress bar); `pip install ipywidgets` |
| Generator fine-tuning is slow or runs out of memory | Run `python -m Pipeline.run_pipeline --stage lm --minutes 20` in a terminal with other apps closed |
| A model is retrained although it exists | Its fingerprint no longer matches the current data (by design). Use `--force` to rebuild everything, or delete `Data/cache/` |

---

## 8. Limitations and next steps

* RealWaste photos share one camera and background; resident phone photos will be harder. Collect real uploads
  through the assistant's feedback loop (`submit_feedback`) and retrain.
* Miscellaneous Trash is the weakest image class (no visual signature) and is often confused with Textile Trash.
* The generated text data is template-based, so text accuracy overstates real-world performance.
* Free-text retrieval is weak; a hybrid sparse + dense retriever or a re-ranker is the next step.
* The generator is fine-tuned on only 180 synthetic examples and works as a policy-grounded formatter; the grounding
  check measures word-level support, not logical correctness, so policy sources still need human review.
* Output is single-label; items made of several materials (a glass jar with a metal lid) get one category.

---

## 9. Repository hygiene

Large, rebuildable folders should not be committed. Suggested `.gitignore` entries:

```
Data/cache/
Data/RealWaste/
Models/
__pycache__/
.ipynb_checkpoints/
.venv/
```

If the grader needs the trained models, leave `Models/` out of the ignore list (or share it separately) and submit
the notebook together with the whole `Pipeline/` folder.

---

## 10. Credits

* **RealWaste** image dataset: UCI Machine Learning Repository (link above) - please cite it as instructed on that page.
* Pre-trained models: MobileNetV2 (ImageNet weights via Keras), `google/flan-t5-small` and
  `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face). Check each model's licence before redistribution.
* `waste_descriptions.csv` and `waste_policy_documents.json` are synthetic data provided for the course.
