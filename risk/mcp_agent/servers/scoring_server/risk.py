"""
risk.py – Professional chemical risk scoring with normalised scores,
dampened organ overlaps, and calibrated thresholds.
Uses ChromaDB for organ knowledge base.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

import chromadb
from chromadb import EmbeddingFunction, Embeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Simple Groq client using config
try:
    from groq import Groq
    _groq_client = None
    def _get_groq_client():
        global _groq_client
        if _groq_client is None:
            _groq_client = Groq(api_key=config.GROQ_API_KEY)
        return _groq_client
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB persistence path
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"

# ─────────────────────────────────────────────────────────────────────────────
# Custom TF-IDF Embedding Function (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class _TFIDFEmbeddingFunction(EmbeddingFunction):
    """Lightweight TF-IDF cosine-similarity embedding function."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float]      = []
        self._fitted: bool          = False

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _fit(self, corpus: list[str]) -> None:
        tokenised = [self._tokenise(doc) for doc in corpus]
        all_tokens: set[str] = set()
        for toks in tokenised:
            all_tokens.update(toks)
        self._vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        V = len(self._vocab)
        N = len(corpus)
        df = [0] * V
        for toks in tokenised:
            for t in set(toks):
                idx = self._vocab.get(t)
                if idx is not None:
                    df[idx] += 1
        self._idf = [math.log((N + 1) / (df[i] + 1)) + 1.0 for i in range(V)]
        self._fitted = True

    def _embed_one(self, text: str) -> list[float]:
        tokens = self._tokenise(text)
        V = len(self._vocab)
        tf_vec = [0.0] * V
        for t in tokens:
            idx = self._vocab.get(t)
            if idx is not None:
                tf_vec[idx] += 1.0
        n = max(len(tokens), 1)
        tfidf = [(tf_vec[i] / n) * self._idf[i] for i in range(V)]
        norm = math.sqrt(sum(x * x for x in tfidf)) or 1.0
        return [x / norm for x in tfidf]

    def __call__(self, input: List[str]) -> Embeddings:
        if not self._fitted:
            self._fit(input)
        return [self._embed_one(t) for t in input]

# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB globals
# ─────────────────────────────────────────────────────────────────────────────
_chroma_client: Any    = None
_organ_collection: Any = None
_embedding_fn: Any     = None

def _get_chroma_client() -> Any:
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _chroma_client

def load_organ_kb(kb_path: str | Path) -> dict[str, Any]:
    """Load organ_priority_data.json into ChromaDB (new API)."""
    global _organ_collection, _embedding_fn
    with open(Path(kb_path), "r", encoding="utf-8") as f:
        kb: dict[str, Any] = json.load(f)

    client = _get_chroma_client()
    # Delete existing collection if exists to avoid version conflicts
    try:
        client.delete_collection("organs")
    except Exception:
        pass

    organs = kb.get("organs", [])
    documents = []
    metadatas = []
    ids = []

    for organ in organs:
        organ_id = organ["id"]
        aliases = organ.get("aliases", [])
        doc_text = f"{organ['name']}: {', '.join(aliases)}"
        documents.append(doc_text)
        ids.append(organ_id)
        metadatas.append({
            "organ_id": organ_id,
            "name": organ["name"],
            "priority_tier": organ.get("priority_tier", 4),
            "priority_weight": float(organ.get("priority_weight", 1.0)),   # lowered default
            "cumulation_multiplier": float(organ.get("cumulation_multiplier", 1.0)),
            "vitality": organ.get("vitality", "unknown"),
            "reversibility": organ.get("reversibility", "unknown"),
            "regeneration_capacity": organ.get("regeneration_capacity", "unknown"),
            "sofa_icu_mortality_pct": float(organ.get("sofa_icu_mortality_pct") or 0.0),
            "system": organ.get("system", ""),
        })

    _embedding_fn = _TFIDFEmbeddingFunction()
    _embedding_fn._fit(documents)

    _organ_collection = client.create_collection(
        name="organs",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    _organ_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return kb

def query_organ(organ_name: str) -> dict[str, Any]:
    """Retrieve organ metadata using vector search."""
    global _organ_collection
    if _organ_collection is None:
        raise RuntimeError("KB not loaded. Call load_organ_kb() first.")
    try:
        results = _organ_collection.query(
            query_texts=[organ_name],
            n_results=1,
            include=["metadatas"],
        )
        if results["metadatas"] and results["metadatas"][0]:
            return results["metadatas"][0][0]
    except Exception:
        pass
    # Neutral fallback defaults (was too high)
    return {
        "organ_id": organ_name,
        "name": organ_name,
        "priority_weight": 1.0,
        "cumulation_multiplier": 1.0,
        "vitality": "unknown",
        "reversibility": "unknown",
        "priority_tier": 4,
        "system": "unknown",
    }

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL RISK SCORING (fixed "always critical")
# ─────────────────────────────────────────────────────────────────────────────

DANGER_LEVEL_SCORES: dict[str, float] = {
    "CRITICAL": 5.0,
    "HIGH":     4.5,
    "MODERATE": 3.5,
    "LOW":      2.0,
    "SAFE":     1.0,
    "UNKNOWN":  0.5,
}

USER_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "asthma": {
        "lungs_respiratory": 1.5,
        "lungs": 1.5,
        "respiratory_system": 1.5,
    },
    "diabetes": {
        "pancreas": 1.4,
    },
    "newborn": {
        "brain_cns": 1.5,
        "brain": 1.5,
        "cns": 1.5,
    },
    "fetal": {
        "reproductive_system": 1.5,
        "reproductive": 1.5,
    },
}

ORGAN_NAME_MAP: dict[str, str] = {
    "skin":               "skin",
    "eyes":               "eyes",
    "eye":                "eyes",
    "liver":              "liver",
    "hepatic":            "liver",
    "kidney":             "kidneys",
    "kidneys":            "kidneys",
    "renal":              "kidneys",
    "respiratory":        "lungs_respiratory",
    "respiratory_system": "lungs_respiratory",
    "lungs":              "lungs_respiratory",
    "lung":               "lungs_respiratory",
    "cns":                "brain_cns",
    "brain":              "brain_cns",
    "nervous system":     "brain_cns",
    "blood":              "blood_hematopoietic",
    "heart":              "heart_cardiovascular",
    "cardiovascular":     "heart_cardiovascular",
    "immune":             "spleen_immune",
    "spleen":             "spleen_immune",
    "thyroid":            "thyroid",
    "reproductive":       "reproductive_system",
    "adrenal":            "adrenal_glands",
    "pancreas":           "pancreas",
    "gastrointestinal":   "gastrointestinal",
    "gi":                 "gastrointestinal",
    "musculoskeletal":    "musculoskeletal",
}

# Neutral defaults (down from 2.0, 1.1)
DEFAULT_PRIORITY_WEIGHT = 1.0
DEFAULT_CUMULATION_MULTIPLIER = 1.0

# -----------------------------------------------------------------------------
# 1. Normalised ingredient risk score (0–10)
# -----------------------------------------------------------------------------
def score_ingredient_risk(chemical: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a normalised ingredient risk score between 0 and 10.
    Uses non-linear confidence scaling to avoid over-penalising.
    """
    name = chemical.get("name", "UNKNOWN")
    verdict = chemical.get("verdict", {})
    danger_level = verdict.get("danger_level", "UNKNOWN").upper()
    base_score = DANGER_LEVEL_SCORES.get(danger_level, 0.5)   # 1..5

    confidence = chemical.get("resolution", {}).get("confidence", 0.5)
    confidence = max(0.0, min(confidence, 1.0))

    # Non-linear: low confidence reduces more than linear
    effective_score = base_score * (0.3 + 0.7 * confidence)

    # Scale to 0-10 (original max base 5 → 10)
    normalised = min(effective_score * 2.0, 10.0)

    return {
        "name": name,
        "danger_level": danger_level,
        "base_score": round(base_score, 2),
        "confidence": round(confidence, 2),
        "normalised_score": round(normalised, 2),
        "justification": verdict.get("justification", []),
    }

# -----------------------------------------------------------------------------
# 2. Dampened organ overlap score (0–10)
# -----------------------------------------------------------------------------
def score_organ_overlap(
    organ_overlap_data: dict[str, Any],
    product_chemicals_evaluated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    overlapping_organs = organ_overlap_data.get("overlapping_organs", [])
    if not overlapping_organs:
        return []

    # Pre‑compute normalised scores for each chemical
    chem_score_map: dict[str, float] = {}
    for chem in product_chemicals_evaluated:
        scored = score_ingredient_risk(chem)
        chem_score_map[scored["name"]] = scored["normalised_score"]

    results = []
    for entry in overlapping_organs:
        organ_name = entry.get("organ", "unknown")
        chemicals = entry.get("chemicals", [])
        count = len(chemicals)
        if count < 2:
            continue

        organ_meta = query_organ(organ_name)
        pw = organ_meta.get("priority_weight", DEFAULT_PRIORITY_WEIGHT)
        cm = organ_meta.get("cumulation_multiplier", DEFAULT_CUMULATION_MULTIPLIER)

        # Sum of normalised scores (each 0-10)
        chem_sum = sum(chem_score_map.get(c, 0.0) for c in chemicals)

        # Use square root to dampen the sum (avoid explosion)
        dampened_sum = math.sqrt(chem_sum) * 1.5   # multiplier keeps range reasonable
        dampened_sum = min(dampened_sum, 10.0)

        # Synergy: much lower than before (max 1.4, starting at 1.1 for 2 chems)
        synergy = 1.0 + (count - 1) * 0.1
        synergy = min(synergy, 1.4)

        raw_score = pw * dampened_sum * cm * synergy
        overlap_score = round(min(raw_score, 10.0), 2)

        risk_level = (
            "CRITICAL" if overlap_score > 8.5 else
            "HIGH"      if overlap_score > 6.0 else
            "MODERATE"  if overlap_score > 3.5 else
            "LOW"
        )

        results.append({
            "organ": organ_name,
            "organ_id": organ_meta.get("organ_id", organ_name),
            "organ_name_full": organ_meta.get("name", organ_name),
            "priority_weight": pw,
            "priority_tier": organ_meta.get("priority_tier", 4),
            "cumulation_multiplier": cm,
            "vitality": organ_meta.get("vitality", "unknown"),
            "reversibility": organ_meta.get("reversibility", "unknown"),
            "chemicals": chemicals,
            "chemical_count": count,
            "synergy_factor": round(synergy, 2),
            "chem_score_sum": round(chem_sum, 2),
            "dampened_sum": round(dampened_sum, 2),
            "overlap_score": overlap_score,
            "risk_level": risk_level,
        })

    results.sort(key=lambda x: x["overlap_score"], reverse=True)
    return results

# -----------------------------------------------------------------------------
# 3. Product total risk (weighted average, not sum)
# -----------------------------------------------------------------------------
def score_product_risk(product: dict[str, Any]) -> dict[str, Any]:
    product_id = product.get("product_id", "?")
    product_name = product.get("product_name", "Unknown")
    chems_eval = product.get("ingredients", {}).get("chemicals_evaluated", [])
    organ_overlap = product.get("combination_risks", {}).get("organ_overlap", {})

    # Ingredient risk: average of normalised scores
    ingredient_scores = [score_ingredient_risk(c) for c in chems_eval]
    if ingredient_scores:
        avg_ingredient_score = sum(s["normalised_score"] for s in ingredient_scores) / len(ingredient_scores)
    else:
        avg_ingredient_score = 0.0

    # Organ overlap risk: average of overlap scores
    organ_overlap_results = []
    overlap_avg_score = 0.0
    if organ_overlap.get("has_overlap", False):
        organ_overlap_results = score_organ_overlap(organ_overlap, chems_eval)
        if organ_overlap_results:
            overlap_avg_score = sum(r["overlap_score"] for r in organ_overlap_results) / len(organ_overlap_results)

    # Weighted average: 70% ingredient risk, 30% organ overlap risk
    total_product_score = round(0.7 * avg_ingredient_score + 0.3 * overlap_avg_score, 2)

    # Calibrated thresholds (0-10 scale)
    verdict = (
        "CRITICAL" if total_product_score >= 8.5 else
        "HIGH"      if total_product_score >= 6.0 else
        "MODERATE"  if total_product_score >= 3.5 else
        "LOW"
    )

    return {
        "product_id": product_id,
        "product_name": product_name,
        "ingredient_scores": ingredient_scores,
        "avg_ingredient_score": round(avg_ingredient_score, 2),
        "organ_overlap_results": organ_overlap_results,
        "overlap_avg_score": round(overlap_avg_score, 2),
        "total_product_score": total_product_score,
        "verdict": verdict,
        "summary": product.get("summary", {}),
        "drivers": product.get("drivers", []),
    }

# -----------------------------------------------------------------------------
# 4. Recurrence risk (unchanged logic, but uses new scores)
# -----------------------------------------------------------------------------
def score_recurrence_risk(global_summary: dict[str, Any]) -> list[dict[str, Any]]:
    organ_analysis = global_summary.get("organ_global_analysis", {})
    high_chems = set(global_summary.get("high_chemicals", []))
    critical_chems = set(global_summary.get("critical_chemicals", []))
    chem_freq: dict[str, int] = {}
    chem_products: dict[str, set] = {}

    for _organ, organ_data in organ_analysis.items():
        for chem, count in organ_data.get("chemical_frequency", {}).items():
            chem_freq[chem] = max(chem_freq.get(chem, 0), count)
        for chem, prods in organ_data.get("products_per_chemical", {}).items():
            chem_products.setdefault(chem, set()).update(prods)

    results = []
    for chem, freq in chem_freq.items():
        if freq < 2:
            continue
        if chem in critical_chems:
            dl = "CRITICAL"
            base_score = 5.0
        elif chem in high_chems:
            dl = "HIGH"
            base_score = 4.5
        else:
            dl = "MODERATE"
            base_score = 3.5

        recurrence_score = round(freq * base_score, 2)
        results.append({
            "chemical": chem,
            "frequency": freq,
            "products": sorted(chem_products.get(chem, set())),
            "danger_level": dl,
            "recurrence_score": recurrence_score,
        })

    results.sort(key=lambda x: x["recurrence_score"], reverse=True)
    return results

# -----------------------------------------------------------------------------
# 5. Ranking and reporting (unchanged)
# -----------------------------------------------------------------------------
def rank_products(product_risk_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(product_risk_results, key=lambda x: x["total_product_score"], reverse=True)

def generate_report_with_groq(analysis_result: dict[str, Any]) -> str:
    if not GROQ_AVAILABLE:
        return "[LLM Report Disabled: Groq not installed]"
    groq = _get_groq_client()
    ranked = analysis_result.get("ranked_products", [])
    recurrence = analysis_result.get("recurrence_risks", [])
    products = analysis_result.get("product_risk_results", [])

    prompt_summary = {
        "products": [
            {
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "verdict": p["verdict"],
                "total_product_score": p["total_product_score"],
                "avg_ingredient_score": p["avg_ingredient_score"],
                "overlap_avg_score": p["overlap_avg_score"],
                "drivers": p.get("drivers", []),
            }
            for p in products
        ],
        "recurrence_risks": recurrence,
        "ranking": [
            {
                "rank": r["rank"],
                "product_name": r["product_name"],
                "total_product_score": r["total_product_score"],
                "verdict": r["verdict"],
            }
            for r in ranked
        ],
        "highest_risk_product": analysis_result.get("highest_risk_product"),
    }

    system_prompt = """You are a senior toxicology risk analyst writing for product safety managers.
Write a clear, professional narrative report. Use exact numbers from the data.
Do not invent any data or chemicals not present in the input.

Generate a complete chemical risk analysis report with these 6 sections:
## 1. Executive Summary
## 2. Individual Ingredient Risk
## 3. Cross-Product Recurrence Risk
## 4. Cumulative Organ Overlap Risk
## 5. Product Risk Ranking
## 6. Recommendations"""
    user_prompt = f"ANALYSIS DATA:\n{json.dumps(prompt_summary, indent=2)}"
    try:
        response = groq.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=2500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Report Generation Failed]: {str(e)}"

def apply_user_adjustment(organ_scores: dict[str, float], user_type: str) -> dict[str, float]:
    if user_type not in USER_ADJUSTMENTS:
        return organ_scores
    adjustments = USER_ADJUSTMENTS[user_type]
    adjusted = organ_scores.copy()
    for organ, multiplier in adjustments.items():
        if organ in adjusted:
            adjusted[organ] = round(adjusted[organ] * multiplier, 3)
        for key in list(adjusted.keys()):
            if organ in key.lower() or key.lower() in organ:
                adjusted[key] = round(adjusted[key] * multiplier, 3)
    return adjusted

def run_pipeline(
    input_data: dict[str, Any],
    user_type: Optional[str] = None,
    generate_llm: bool = True,
) -> dict[str, Any]:
    products = input_data.get("products", [])
    global_summary = input_data.get("global_summary", {})

    product_risk_results = [score_product_risk(p) for p in products]
    recurrence_risks = score_recurrence_risk(global_summary)
    ranked_products = rank_products(product_risk_results)
    highest = ranked_products[0] if ranked_products else None

    if user_type and highest:
        for product_result in product_risk_results:
            for organ_result in product_result.get("organ_overlap_results", []):
                organ_name = organ_result.get("organ_id", organ_result.get("organ", ""))
                original_score = organ_result.get("overlap_score", 0)
                adjusted = apply_user_adjustment({organ_name: original_score}, user_type)
                organ_result["original_overlap_score"] = original_score
                organ_result["overlap_score"] = adjusted.get(organ_name, original_score)
                organ_result["user_adjusted"] = True
                organ_result["user_type"] = user_type

    analysis_result = {
        "product_risk_results": product_risk_results,
        "ranked_products": [
            {
                "rank": i + 1,
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "total_product_score": p["total_product_score"],
                "verdict": p["verdict"],
            }
            for i, p in enumerate(ranked_products)
        ],
        "highest_risk_product": {
            "product_id": highest["product_id"],
            "product_name": highest["product_name"],
            "score": highest["total_product_score"],
            "verdict": highest["verdict"],
        } if highest else None,
        "recurrence_risks": recurrence_risks,
        "global_organs_at_risk": global_summary.get("organs_under_pressure", []),
        "global_high_chemicals": global_summary.get("high_chemicals", []),
        "user_type_applied": user_type,
    }

    if generate_llm:
        analysis_result["llm_report"] = generate_report_with_groq(analysis_result)

    return analysis_result

# -----------------------------------------------------------------------------
# Optional: integration with risk_scoring.py for better hazard assessment
# Uncomment if you want to replace the simple danger_level scoring.
# -----------------------------------------------------------------------------
# try:
#     from risk_scoring import calculate_risk_for_chemical
#     def score_ingredient_risk_pro(chemical: dict) -> dict:
#         # Convert chemical to the format expected by risk_scoring
#         result = calculate_risk_for_chemical(
#             ghs_pictograms=chemical.get("ghs_pictograms", []),
#             h_codes=chemical.get("h_codes", []),
#             target_organs=chemical.get("target_organs", []),
#             chemical_classes=chemical.get("chemical_classes", []),
#             product_usage=chemical.get("product_usage", "")
#         )
#         # Map the 0-5 total_score to 0-10 normalised
#         normalised = result["total_score"] * 2
#         return {
#             "name": chemical.get("name", "UNKNOWN"),
#             "danger_level": result["verdict"],
#             "base_score": result["total_score"],
#             "confidence": 1.0,
#             "normalised_score": round(normalised, 2),
#             "justification": result["breakdown"],
#         }
#     # Now you could replace score_ingredient_risk with this version.
# except ImportError:
#     pass