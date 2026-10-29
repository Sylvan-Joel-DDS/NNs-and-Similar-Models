"""Project paths, constants and cache fingerprints.

Layout (this file lives in <project>/Pipeline):
    Data/      RealWaste/, waste_descriptions.csv, waste_policy_documents.json, cache/  (cache/ is created here)
    Models/    trained artefacts (*.keras, text_model.joblib, flan_t5_recycling/, instruction_cache.json)
    Notebook/  notebooks
    Pipeline/  this package
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "Models"
CACHE_DIR = DATA_DIR / "cache"

DESCRIPTIONS_CSV = DATA_DIR / "waste_descriptions.csv"
POLICY_JSON = DATA_DIR / "waste_policy_documents.json"

CATEGORIES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',
              'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

CNN_FINAL = MODELS_DIR / "ecosort_cnn_final.keras"
TEXT_MODEL = MODELS_DIR / "text_model.joblib"
# flan-t5-small (77M params) trains in ~2 GB of RAM on a CPU; use google/flan-t5-base (250M) only with a GPU / 16 GB+.
GEN_BASE_MODEL = os.environ.get("ECOSORT_GEN_MODEL", "google/flan-t5-small")
INSTRUCTION_CACHE = MODELS_DIR / "instruction_cache.json"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def fingerprint(*parts) -> str:
    """Short stable hash of arbitrary JSON-able parts; used to invalidate caches when inputs change."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def find_image_root() -> Path:
    """Folder that directly contains the 9 class folders (the UCI/GitHub zip nests it one or two levels deep)."""
    base = DATA_DIR / "RealWaste"
    candidates = [base]
    if base.exists():
        level1 = [p for p in base.iterdir() if p.is_dir()]
        candidates += level1 + [q for p in level1 for q in p.iterdir() if q.is_dir()]
    for d in candidates:
        if all((d / c).is_dir() for c in CATEGORIES):
            return d
    raise FileNotFoundError(f"RealWaste class folders not found under {base}. Expected {CATEGORIES}.")


def read_meta(path: Path) -> dict:
    p = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    return json.loads(p.read_text()) if p.exists() else {}


def write_meta(path: Path, **meta) -> None:
    p = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    p.write_text(json.dumps(meta, indent=2, default=str))
