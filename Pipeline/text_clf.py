"""Waste-description classifier: cleaning, leak-free splits, TF-IDF + linear model (trained once, cached)."""
from __future__ import annotations
import re, unicodedata, time

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .config import CATEGORIES

LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def clean_text(text) -> str:
    """Unicode-normalise, lowercase, split hyphenated words (t-shirt -> t shirt), drop punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_text_splits(seed: int = config.SEED) -> dict:
    """De-duplicate BEFORE splitting (no identical string in train and test), stratified 70/15/15."""
    df = pd.read_csv(config.DESCRIPTIONS_CSV)
    df["clean"] = df.description.map(clean_text)
    full = df.drop_duplicates("clean").reset_index(drop=True)
    train, tmp = train_test_split(full, test_size=0.30, stratify=full.category, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.category, random_state=seed)
    out = dict(raw=df, full=full, train=train, val=val, test=test)
    for k in ("train", "val", "test"):
        out[f"X_{k}"] = out[k].clean.values
        out[f"y_{k}"] = out[k].category.map(LABEL2ID).values
    return out


def make_features() -> FeatureUnion:
    """Word 1-2-grams + character 2-5-grams (the latter make the model robust to typos / plurals / compounds)."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


def candidate_models() -> dict:
    return {
        "ComplementNB": ComplementNB(alpha=0.3),
        "LogisticRegression": LogisticRegression(C=10, max_iter=2000),
        "LinearSVC": LinearSVC(C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=config.SEED),
    }


def compare_models(splits: dict) -> tuple[pd.DataFrame, dict]:
    """Fit every candidate on TRAIN, score on VALIDATION."""
    rows, fitted = [], {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("features", make_features()), ("clf", clf)])
        t0 = time.time(); pipe.fit(splits["X_train"], splits["y_train"]); dt = time.time() - t0
        pred = pipe.predict(splits["X_val"])
        rows.append(dict(model=name, val_acc=accuracy_score(splits["y_val"], pred),
                         val_macro_f1=f1_score(splits["y_val"], pred, average="macro"), train_sec=round(dt, 1)))
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


PARAM_GRID = {
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "features__char__ngram_range": [(2, 4), (2, 5), (3, 6)],
    "clf__C": [1, 10, 50],
}


def train_text_model(splits: dict, force: bool = False, verbose: bool = True) -> dict:
    """5-fold grid search on TRAIN. Result is cached in Models/text_model.joblib and re-used while the training
    data and grid are unchanged (the grid search is the slowest step of Part 3)."""
    config.ensure_dirs()
    fp = config.fingerprint(list(splits["X_train"]), list(splits["y_train"]), PARAM_GRID)
    if config.TEXT_MODEL.exists() and not force:
        art = joblib.load(config.TEXT_MODEL)
        if art.get("fingerprint") == fp:
            if verbose: print("Text model loaded from cache:", config.TEXT_MODEL.name, "| params:", art["best_params"])
            return art
    base = Pipeline([("features", make_features()), ("clf", LogisticRegression(max_iter=3000))])
    gs = GridSearchCV(base, PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1)
    gs.fit(splits["X_train"], splits["y_train"])
    art = dict(model=gs.best_estimator_, best_params=gs.best_params_, cv_macro_f1=gs.best_score_, fingerprint=fp)
    joblib.dump(art, config.TEXT_MODEL)
    if verbose: print("Text model trained and cached | params:", gs.best_params_, "| CV macro-F1:", round(gs.best_score_, 4))
    return art


class TextClassifier:
    """Thin wrapper used by the assistant: probabilities over CATEGORIES + a vocabulary-coverage signal."""
    def __init__(self, model=None):
        if model is None:
            if not config.TEXT_MODEL.exists():
                raise FileNotFoundError("Text model not trained yet - run Part 3 or `python -m Pipeline.run_pipeline --stage text`.")
            model = joblib.load(config.TEXT_MODEL)["model"]
        self.model = model
        self.vocab = set(model.named_steps["features"].transformer_list[0][1].vocabulary_)

    def predict_proba(self, description: str):
        if not isinstance(description, str):
            raise TypeError("Text input must be a string")
        clean = clean_text(description)
        if not clean:
            raise ValueError("Description is empty")
        known = [w for w in clean.split() if w in self.vocab]
        return self.model.predict_proba([clean])[0], len(known) / len(clean.split())

    def classify(self, description: str, top_k: int = 3):
        probs, _ = self.predict_proba(description)
        order = probs.argsort()[::-1][:top_k]
        return ID2LABEL[order[0]], float(probs[order[0]]), [(ID2LABEL[i], float(probs[i])) for i in order]
