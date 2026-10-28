"""Integrated EcoSort assistant: image or text -> category -> recycling instructions + policy documents.

Components are loaded lazily and independently:
  * text path   : scikit-learn only
  * image path  : TensorFlow, imported on first image
  * generation  : PyTorch, only if an answer is not already cached (see RecyclingAdvisor.warm_cache)
so a broken PyTorch/TensorFlow install in one path never takes down the other.
"""
from __future__ import annotations
import os, time, uuid
import numpy as np
import pandas as pd

from . import config, images
from .config import CATEGORIES
from .policy_rag import RecyclingAdvisor, build_knowledge_base, build_retriever
from .text_clf import TextClassifier

IMG_CONF_MIN = 0.60
TXT_CONF_MIN = 0.50


class EcoSortAssistant:
    def __init__(self, cnn_model=None, text_clf: TextClassifier | None = None, advisor: RecyclingAdvisor | None = None,
                 img_conf_min=IMG_CONF_MIN, txt_conf_min=TXT_CONF_MIN):
        self._cnn, self._text, self._advisor = cnn_model, text_clf, advisor
        self.img_conf_min, self.txt_conf_min = img_conf_min, txt_conf_min
        self.feedback_log: list[dict] = []

    # --- lazy components
    @property
    def cnn(self):
        if self._cnn is None:
            from . import cnn as cnn_mod
            self._cnn = cnn_mod.load_cnn()
            if self._cnn is None:
                raise FileNotFoundError(f"No trained CNN at {config.CNN_FINAL}. Run Part 2 or `python -m Pipeline.run_pipeline --stage cnn`.")
        return self._cnn

    @property
    def text(self):
        if self._text is None:
            self._text = TextClassifier()
        return self._text

    @property
    def advisor(self):
        if self._advisor is None:
            from .generator import Generator
            kb = build_knowledge_base()
            retriever, _ = build_retriever(kb)
            self._advisor = RecyclingAdvisor(kb, retriever, Generator())
        return self._advisor

    # --- classification branches
    def _classify_image(self, x):
        arr = images.load_image(x).astype("float32")
        return self.cnn.predict(arr[None], verbose=0)[0]

    def _classify_text(self, description):
        return self.text.predict_proba(description)

    # --- main entry
    def __call__(self, input_data, input_type="image", top_k=3) -> dict:
        t0 = time.time()
        res = dict(request_id=uuid.uuid4().hex[:8], input_type=input_type, status="ok", category=None, confidence=None,
                   top_predictions=[], instructions=None, policy_documents=[], grounding=None, generation_mode=None,
                   warnings=[], latency_sec=None)
        try:
            if input_type == "image":
                probs, min_conf = self._classify_image(input_data), self.img_conf_min
            elif input_type == "text":
                probs, known_frac = self._classify_text(input_data)
                min_conf = self.txt_conf_min
                if known_frac == 0:
                    res["warnings"].append("None of the words were recognised, so this guess is unreliable.")
                    res["status"] = "low_confidence"
                elif known_frac < 0.5:
                    res["warnings"].append("Few words were recognised; try describing the main object (e.g. 'glass jar').")
            else:
                raise ValueError("input_type must be 'image' or 'text'")

            order = probs.argsort()[::-1][:top_k]
            res["category"], res["confidence"] = CATEGORIES[order[0]], float(probs[order[0]])
            res["top_predictions"] = [(CATEGORIES[i], round(float(probs[i]), 3)) for i in order]
            if res["confidence"] < min_conf:
                res["status"] = "low_confidence"
                alt = "send a description of the item" if input_type == "image" else "add more detail (material, object)"
                res["warnings"].append(f"Confidence {res['confidence']:.0%} is below {min_conf:.0%}; please {alt}. "
                                       f"Other possibilities: {', '.join(c for c, _ in res['top_predictions'][1:])}.")
            gen = self.advisor.details(res["category"])
            res.update(instructions=gen["instructions"], policy_documents=gen["policy_documents"],
                       grounding=gen["grounding"], generation_mode=gen["mode"])
            if gen["mode"] == "extractive_only":
                res["warnings"].append("Language model unavailable - showing the policy text directly.")
        except (FileNotFoundError, ValueError, TypeError) as e:
            res.update(status="error", instructions=None)
            res["warnings"].append(f"{type(e).__name__}: {e}")
        except Exception as e:                                   # never crash the user-facing app
            res.update(status="error")
            res["warnings"].append(f"Unexpected error ({type(e).__name__}): {e}")
        res["latency_sec"] = round(time.time() - t0, 2)
        return res

    def image_and_text(self, image, description, w_img=0.5) -> dict:
        """Optional multimodal mode: average the two probability vectors; report whether the inputs agree."""
        p_img, (p_txt, _) = self._classify_image(image), self._classify_text(description)
        probs = w_img * p_img + (1 - w_img) * p_txt
        cat, agree = CATEGORIES[int(probs.argmax())], p_img.argmax() == p_txt.argmax()
        out = self(description, "text"); gen = self.advisor.details(cat)
        out.update(category=cat, confidence=float(probs.max()), instructions=gen["instructions"],
                   policy_documents=gen["policy_documents"], input_type="image+text")
        if not agree:
            out["warnings"].append(f"Image says {CATEGORIES[p_img.argmax()]}, text says {CATEGORIES[p_txt.argmax()]}.")
        return out

    # --- feedback loop
    def submit_feedback(self, result, correct_category=None, helpful=None, comment=""):
        if correct_category is not None and correct_category not in CATEGORIES:
            raise ValueError(f"correct_category must be one of {CATEGORIES}")
        self.feedback_log.append(dict(request_id=result["request_id"], input_type=result["input_type"],
                                      predicted=result["category"], correct_category=correct_category, helpful=helpful,
                                      comment=comment, confidence=result["confidence"],
                                      timestamp=pd.Timestamp.now().isoformat(timespec="seconds")))

    def feedback_report(self):
        df = pd.DataFrame(self.feedback_log)
        if df.empty:
            return "No feedback yet."
        lab = df.dropna(subset=["correct_category"])
        acc = (lab.predicted == lab.correct_category).mean() if len(lab) else float("nan")
        return dict(n_feedback=len(df), accuracy_from_feedback=acc,
                    retraining_rows=lab[lab.predicted != lab.correct_category][["input_type", "predicted", "correct_category"]])


def show_result(res):
    print(f"[{res['status'].upper()}] request {res['request_id']} | input: {res['input_type']} | {res['latency_sec']}s")
    if res["category"]:
        print(f"Category: {res['category']} ({res['confidence']:.1%})   top-{len(res['top_predictions'])}: {res['top_predictions']}")
    for w in res["warnings"]:
        print("  ! ", w)
    if res["instructions"]:
        print(f"\n{res['instructions']}\n\n(generation: {res['generation_mode']}, grounding {res['grounding']:.2f})")
        print("Relevant policy documents:", [f"#{d['policy_id']} {d['policy_type']}" for d in res["policy_documents"]])
