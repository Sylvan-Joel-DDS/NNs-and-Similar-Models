"""Policy knowledge base, retrieval and grounded instruction composition (no PyTorch needed at import time)."""
from __future__ import annotations
import json, re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .config import CATEGORIES

pretty = lambda c: c.lower()          # how a category is written inside a question
CANONICAL = {c.lower(): c for c in CATEGORIES}
_CAT_BY_UPPER = {c.upper(): c for c in CATEGORIES}

SECTION_LABELS = {
    "Acceptable Items": "Accepted", "Non-Acceptable Items": "Not accepted",
    "Preparation Instructions": "Prepare", "Collection Method": "Where to put it",
    "Items That Cannot Be Recycled": "Cannot be recycled", "Special Handling Items": "Special handling",
    "Waste Reduction Tips": "Reduce waste", "Benefits": "Why it matters",
}
SECTION_ORDER = ["Acceptable Items", "Non-Acceptable Items", "Items That Cannot Be Recycled", "Special Handling Items",
                 "Preparation Instructions", "Collection Method", "Waste Reduction Tips", "Benefits"]
MIN_GROUNDING, MIN_COVERAGE = 0.90, 0.75


# ----------------------------------------------------------------------------- knowledge base
def parse_policy_text(text: str) -> list[dict]:
    """Split a policy document into sections: [{'section', 'bullets', 'paragraph'}]."""
    sections, current = [], None
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith("-"):
            current = {"section": line[:-1].strip(), "bullets": [], "paragraph": []}
            sections.append(current)
        elif line.startswith("- "):
            if current is not None: current["bullets"].append(line[2:].strip())
        elif current is not None:
            current["paragraph"].append(line)
    return sections


def _norm(s): return re.sub(r"\s+", " ", s.lower()).strip()
def _body(s): return "; ".join(s["bullets"]) if s["bullets"] else " ".join(s["paragraph"])


@dataclass
class KnowledgeBase:
    policies: list
    chunks: list
    conflicts: pd.DataFrame            # bundled-document items that contradict the category-specific document
    related_policies: dict             # category -> [policy docs], authoritative first
    auth: dict                         # category -> {'policy': doc, 'sections': {name: parsed section}}


def build_knowledge_base(policy_path=None, descriptions_path=None) -> KnowledgeBase:
    policies = json.loads(open(policy_path or config.POLICY_JSON, encoding="utf-8").read())
    desc = pd.read_csv(descriptions_path or config.DESCRIPTIONS_CSV)

    single = [p for p in policies if len(p["categories_covered"]) == 1]
    bundled = [p for p in policies if len(p["categories_covered"]) > 1]
    auth = {p["categories_covered"][0]: {"policy": p,
            "sections": {s["section"]: s for s in parse_policy_text(p["document_text"])}} for p in single}

    # 1) conflicts: a bundled document ACCEPTS what the authoritative document REJECTS
    conflicts, bundled_parsed = [], []
    for p in bundled:
        for s in parse_policy_text(p["document_text"]):
            name = s["section"].upper().replace(" GUIDELINES", "")
            cat = "General" if name == "GENERAL REQUIREMENTS" else _CAT_BY_UPPER.get(name)
            if cat is None:
                continue
            kept = list(s["bullets"])
            if cat != "General" and "Non-Acceptable Items" in auth[cat]["sections"]:
                rejected = {_norm(b) for b in auth[cat]["sections"]["Non-Acceptable Items"]["bullets"]}
                for b in s["bullets"]:
                    if _norm(b) in rejected:
                        conflicts.append(dict(category=cat, item=b, listed_as_accepted_in=p["policy_type"],
                                              bundled_date=p["effective_date"],
                                              rejected_in=auth[cat]["policy"]["policy_type"],
                                              authoritative_date=auth[cat]["policy"]["effective_date"]))
                        kept.remove(b)
            bundled_parsed.append(dict(policy=p, category=cat, bullets=kept))

    # 2) chunks (one chunk = one section of one category) with provenance
    chunks = []
    def add(source, category, section, body, ids=(), ptype=None, date=None, authoritative=False):
        chunks.append(dict(chunk_id=len(chunks), source=source, category=category, section=section, body=body,
                           policy_ids=list(ids), policy_type=ptype, effective_date=date, authoritative=authoritative,
                           embed_text=f"{category} - {section}: {body}"))

    for cat, d in auth.items():
        p = d["policy"]
        for name, s in d["sections"].items():
            add("policy", cat, name, _body(s), [p["policy_id"]], p["policy_type"], p["effective_date"], True)

    general_seen = {}
    for b in bundled_parsed:
        p = b["policy"]
        if b["category"] == "General":
            key = _norm("; ".join(b["bullets"]))
            if key in general_seen:
                chunks[general_seen[key]]["policy_ids"].append(p["policy_id"])
            else:
                general_seen[key] = len(chunks)
                add("policy", "General", "General Requirements", "; ".join(b["bullets"]), [p["policy_id"]],
                    "Municipal programme documents", p["effective_date"], False)
        else:
            known = {_norm(x) for sec in auth[b["category"]]["sections"].values() for x in sec["bullets"]}
            new_items = [x for x in b["bullets"] if _norm(x) not in known]
            if new_items:   # information not in the authoritative document -> keep as its own chunk
                add("policy", b["category"], "Programme Guidelines", "; ".join(new_items), [p["policy_id"]],
                    p["policy_type"], p["effective_date"], False)

    for cat, g in desc.groupby("category"):
        for tip in sorted(g.disposal_instruction.unique()):   add("csv_tip", cat, "Disposal Tip", tip)
        for note in sorted(g.common_confusion.dropna().unique()): add("csv_confusion", cat, "Common Confusion", note)
        for comp in sorted(g.material_composition.unique()):  add("csv_material", cat, "Material Composition", comp)

    related = defaultdict(list)
    for p in policies:
        for c in p["categories_covered"]:
            related[c].append(p)
    for c in related:
        related[c].sort(key=lambda p: (len(p["categories_covered"]) > 1, p["policy_id"]))
    return KnowledgeBase(policies, chunks, pd.DataFrame(conflicts), dict(related), auth)


# ----------------------------------------------------------------------------- embeddings + retrieval
class Embedder:
    """MiniLM sentence embeddings when sentence-transformers + PyTorch load; otherwise a TF-IDF fallback
    (so the pipeline still runs, with weaker semantic matching, if PyTorch is broken)."""
    def __init__(self, corpus, prefer_dense: bool = True, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        if prefer_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.name = model_name
            except Exception as e:
                print(f"[Embedder] dense model unavailable ({type(e).__name__}); using TF-IDF fallback.")
        if self.model is None:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)
            self.name = "tfidf-fallback"

    def __call__(self, texts):
        texts = list(texts)
        if self.model is not None:
            return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        M = self.tfidf.transform(texts).toarray()
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


class PolicyRetriever:
    """Exact cosine retriever with category filtering and a 'policy-first' mode (authoritative sections always included)."""
    def __init__(self, chunks, embeddings, embed_fn, authority_bonus: float = 0.05):
        self.chunks, self.E, self.embed, self.bonus = chunks, embeddings, embed_fn, authority_bonus
        self.cat = np.array([c["category"] for c in chunks])
        self.auth = np.array([c["authoritative"] for c in chunks], dtype=float)

    def with_bonus(self, bonus):
        return PolicyRetriever(self.chunks, self.E, self.embed, bonus)

    def search(self, query, category=None, k=5, include_general=True):
        scores = self.E @ self.embed([query])[0] + self.bonus * self.auth
        mask = (self.cat != "General") if category is None else \
               ((self.cat == category) | ((self.cat == "General") & include_general))
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-scores[idx])][:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in idx]

    def retrieve_for_category(self, category, query=None, k=8):
        query = query or f"How to recycle {category}"
        core = [dict(c, score=1.0) for c in self.chunks if c["category"] == category and c["authoritative"]]
        core_ids = {c["chunk_id"] for c in core}
        extras = [c for c in self.search(query, category, k=k + len(core)) if c["chunk_id"] not in core_ids]
        return core + extras[:max(0, k - len(core))]


def build_retriever(kb: KnowledgeBase, embedder: Embedder | None = None, authority_bonus: float = 0.05):
    """Chunk embeddings are cached in Data/cache (keyed by chunk text + embedding model)."""
    config.ensure_dirs()
    texts = [c["embed_text"] for c in kb.chunks]
    embedder = embedder or Embedder(texts)
    path = config.CACHE_DIR / f"chunk_emb_{config.fingerprint(texts, embedder.name)}.npy"
    if path.exists():
        E = np.load(path)
    else:
        E = embedder(texts); np.save(path, E)
    return PolicyRetriever(kb.chunks, E, embedder, authority_bonus), embedder


# ----------------------------------------------------------------------------- composition, grounding, metrics
def build_prompt(query, ctx_chunks):
    excerpts = "\n".join(f"[{i + 1}] ({c['category']} - {c['section']}) {c['body']}" for i, c in enumerate(ctx_chunks))
    return ("Write step-by-step recycling instructions using only the policy excerpts.\n"
            f"Question: {query}\nPolicy excerpts:\n{excerpts}\nInstructions:")


def format_instructions(text: str) -> str:
    """T5's tokenizer turns newlines into spaces, so restore one numbered step / tip per line."""
    text = re.sub(r"\s+(?=\d+\.\s+[A-Z])", "\n", text)
    text = re.sub(r"\s+(?=(?:Tip|Note):)", "\n", text)
    return text.strip()


def compose_instructions(category, ctx_chunks, rng=None) -> str:
    """Deterministic, policy-faithful instructions built only from the retrieved chunks.
    Used (a) as the fine-tuning target and (b) as the extractive fallback."""
    by_sec = {c["section"]: c for c in ctx_chunks if c["source"] == "policy" and c["category"] == category}
    lines, n = [f"How to recycle {category}:"], 1
    for sec in SECTION_ORDER:
        if sec in by_sec:
            lines.append(f"{n}. {SECTION_LABELS[sec]}: {by_sec[sec]['body']}"); n += 1
    tips = [c["body"] for c in ctx_chunks if c["source"] == "csv_tip"]
    notes = [c["body"] for c in ctx_chunks if c["source"] == "csv_confusion"]
    if tips:  lines.append("Tip: " + (rng.choice(tips) if rng else tips[0]))
    if notes: lines.append("Note: " + notes[0])
    return "\n".join(lines)


_STOP = set("""a an the and or of to in on for with from by at as is are be it its this that these those if not no
               than then so can may must should will your you any all some more most other into out up use using""".split())
_LABEL_WORDS = {w for lab in SECTION_LABELS.values() for w in re.findall(r"[a-z]+", lab.lower())} | {"how", "recycle", "tip", "note"}
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


def grounding_score(generated, ctx_chunks) -> dict:
    """Share of content words in the generated text that appear in the retrieved policy text (1.0 = fully supported)."""
    ctx_words = set(_tok(" ".join(c["body"] + " " + c["category"] + " " + c["section"] for c in ctx_chunks)))
    words = [w for w in _tok(generated) if w not in _STOP and w not in _LABEL_WORDS and not w.isdigit() and len(w) > 2]
    if not words:
        return dict(score=0.0, unsupported=[])
    bad = [w for w in words if w not in ctx_words]
    return dict(score=1 - len(bad) / len(words), unsupported=sorted(set(bad)))


def label_coverage(generated, category, auth) -> float:
    """Share of the expected section labels (from the authoritative document) present in the output."""
    expected = [SECTION_LABELS[s] for s in SECTION_ORDER if s in auth[category]["sections"] and s != "Benefits"]
    return float(np.mean([lab.lower() in generated.lower() for lab in expected])) if expected else 1.0


def readability(text) -> dict:
    """Flesch Reading Ease + average sentence length (labels/bullets count as sentences)."""
    sents = [s for s in re.split(r"[.\n;]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    if not words or not sents:
        return dict(flesch=np.nan, words_per_sentence=np.nan)
    syl = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
    return dict(flesch=206.835 - 1.015 * len(words) / len(sents) - 84.6 * syl / len(words),
                words_per_sentence=len(words) / len(sents))


# ----------------------------------------------------------------------------- the advisor (retrieve -> generate -> verify)
class RecyclingAdvisor:
    """retrieve -> generate (if a generator is available) -> grounding check -> extractive fallback.
    Answers are cached in memory and in Models/instruction_cache.json, so after `warm_cache()` the assistant
    answers instantly and needs neither PyTorch nor the language model."""

    def __init__(self, kb: KnowledgeBase, retriever: PolicyRetriever, generator=None, gen_kwargs=None, use_disk_cache=True):
        self.kb, self.retriever, self.generator = kb, retriever, generator
        self.gen_kwargs = gen_kwargs or {}
        self.use_disk_cache = use_disk_cache
        self._cache = {}
        self._load_cache()

    # --- cache
    def _signature(self):
        """Generator weights + decoding settings: any change invalidates the cached answers."""
        sig = getattr(self.generator, "signature", "no-generator") if self.generator is not None else "no-generator"
        return f"{sig}|{json.dumps(self.gen_kwargs, sort_keys=True)}"

    def _load_cache(self):
        p = config.INSTRUCTION_CACHE
        if not (self.use_disk_cache and p.exists()):
            return
        blob = json.loads(p.read_text())
        generator_ok = self.generator is not None and getattr(self.generator, "available", False)
        if blob.get("signature") == self._signature() or not generator_ok:   # no generator -> any cache beats nothing
            self._cache = blob.get("entries", {})

    def save_cache(self):
        config.ensure_dirs()
        config.INSTRUCTION_CACHE.write_text(json.dumps(dict(signature=self._signature(), entries=self._cache), indent=1))

    def clear_cache(self, disk=True):
        self._cache = {}
        if disk and config.INSTRUCTION_CACHE.exists():
            config.INSTRUCTION_CACHE.unlink()

    # --- documents
    def policy_documents(self, category):
        docs = []
        for p in self.kb.related_policies[category]:
            primary = len(p["categories_covered"]) == 1
            removed = [c for c in self.kb.conflicts.to_dict("records")
                       if c["category"] == category and c["listed_as_accepted_in"] == p["policy_type"]] if len(self.kb.conflicts) else []
            docs.append(dict(policy_id=p["policy_id"], policy_type=p["policy_type"], effective_date=p["effective_date"],
                             jurisdiction=p["jurisdiction"], role="primary (authoritative)" if primary else "also applies",
                             text=p["document_text"] if primary else None,
                             note=(f"Lists {', '.join(r['item'] for r in removed)} as accepted, which contradicts the "
                                   f"{self.kb.auth[category]['policy']['policy_type']}; that item was ignored.") if removed else None))
        return docs

    # --- main entry
    def details(self, waste_category, query=None) -> dict:
        category = CANONICAL.get(str(waste_category).strip().lower())
        if category is None:
            raise ValueError(f"Unknown waste category {waste_category!r}. Expected one of: {CATEGORIES}")
        query = query or f"How to recycle {pretty(category)}"
        key = f"{category}||{query}"
        if key in self._cache:
            return self._cache[key]

        ctx = self.retriever.retrieve_for_category(category, query)
        text, mode = None, "extractive_only"
        if self.generator is not None and getattr(self.generator, "available", False):
            mode = "extractive_fallback"
            raw = self.generator.generate(build_prompt(query, ctx), **self.gen_kwargs)
            g, cov = grounding_score(raw, ctx), label_coverage(raw, category, self.kb.auth)
            if g["score"] >= MIN_GROUNDING and cov >= MIN_COVERAGE:      # safeguard against hallucination
                text, mode = raw, "generated"
        if text is None:
            text = compose_instructions(category, ctx)
        g, cov = grounding_score(text, ctx), label_coverage(text, category, self.kb.auth)
        result = dict(category=category, query=query, instructions=text, mode=mode, grounding=g["score"],
                      label_coverage=cov, unsupported_terms=g["unsupported"], policy_documents=self.policy_documents(category),
                      retrieved_chunks=[(c["section"], c["source"]) for c in ctx])
        self._cache[key] = result
        return result

    def warm_cache(self, save=True):
        """Pre-compute the default answer for every category (needs the generator once; free afterwards)."""
        for c in CATEGORIES:
            self.details(c)
        if save: self.save_cache()
