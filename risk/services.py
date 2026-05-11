# risk/services.py
import sys
import os
import json
import asyncio
import concurrent.futures
import threading
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from django.conf import settings

# ----------------------------------------------------------------------
# 1. Make sure the MCP agent root is in sys.path
# ----------------------------------------------------------------------
MCP_AGENT_ROOT = Path(__file__).parent / "mcp_agent"
sys.path.insert(0, str(MCP_AGENT_ROOT.parent))
sys.path.insert(0, str(MCP_AGENT_ROOT))

# ----------------------------------------------------------------------
# 2. Disable scoring server if chromadb is missing
# ----------------------------------------------------------------------
import mcp_agent.agent.agent as agent_module
if "scoring" in agent_module.SERVER_PATHS:
    try:
        import chromadb
    except ImportError:
        del agent_module.SERVER_PATHS["scoring"]
        print("⚠️ Scoring server disabled (chromadb missing)\n")

# ----------------------------------------------------------------------
# 3. Import the real BiologicalAgent (no debug subclass)
# ----------------------------------------------------------------------
from mcp_agent.agent.agent import BiologicalAgent

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_product_data(full_report: dict) -> Dict[str, Any]:
    if not full_report or "products" not in full_report or not full_report["products"]:
        return {}
    first = full_report["products"][0]
    return dict(first) if isinstance(first, dict) else {}


def _summary_counts(product_data: Dict[str, Any]) -> Dict[str, Any]:
    summary = _as_dict(product_data.get("summary"))
    ingredients = _as_dict(product_data.get("ingredients"))
    chemicals = _as_list(ingredients.get("chemicals_evaluated"))
    safe_skipped = _as_list(ingredients.get("safe_skipped"))
    return {
        "total_ingredients": summary.get("total_ingredients", len(chemicals) + len(safe_skipped)),
        "chemicals_evaluated": summary.get("chemicals_evaluated", len(chemicals)),
        "critical_count": summary.get("critical", 0),
        "high_count": summary.get("high", 0),
        "moderate_count": summary.get("moderate", 0),
        "low_count": summary.get("low", 0),
        "safe_count": summary.get("safe", len(safe_skipped)),
        "organ_overlap_flags": summary.get("organ_overlap_flags", 0),
    }


def _report_status(product_data: Dict[str, Any]) -> str:
    if not product_data:
        return "minimal"
    summary = _as_dict(product_data.get("summary"))
    ingredients = _as_dict(product_data.get("ingredients"))
    chemicals = _as_list(ingredients.get("chemicals_evaluated"))
    if summary and chemicals:
        return "complete"
    if summary or chemicals:
        return "partial"
    return "minimal"


def _summary_text(product_data: Dict[str, Any]) -> str:
    name = product_data.get("product_name") or product_data.get("name") or "this product"
    status = _report_status(product_data)
    if status == "complete":
        return f"A complete verified analysis is available for {name}."
    if status == "partial":
        return f"A partial analysis is available for {name}. Verified safety signals are shown first."
    return f"We do not yet have enough verified data to complete the analysis for {name}."


def _key_findings(product_data: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    summary = _as_dict(product_data.get("summary"))
    ingredients = _as_dict(product_data.get("ingredients"))
    chemicals = _as_list(ingredients.get("chemicals_evaluated"))
    for field in ("drivers", "warnings", "key_findings", "recommendations"):
        for item in _as_list(product_data.get(field)):
            text = str(item).strip()
            if text and text not in findings:
                findings.append(text)
    for chem in chemicals:
        if not isinstance(chem, dict):
            continue
        verdict = _as_dict(chem.get("verdict"))
        danger = str(verdict.get("danger_level") or "").upper()
        if danger in {"CRITICAL", "HIGH"}:
            name = str(chem.get("name") or "").strip()
            if name and name not in findings:
                findings.append(name)
    combination = _as_dict(product_data.get("combination_risks"))
    organ_overlap = _as_dict(combination.get("organ_overlap"))
    for organ in _as_list(organ_overlap.get("overlapping_organs")):
        text = str(organ).strip()
        if text and text not in findings:
            findings.append(text)
    if not findings:
        total = summary.get("total_ingredients") or len(chemicals)
        findings.append(f"{total} ingredient signal(s) were checked.")
    return findings[:5]


def _organ_impact(product_data: Dict[str, Any]) -> List[str]:
    summary = _as_dict(product_data.get("summary"))
    organs = _as_list(summary.get("organs_under_pressure"))
    if organs:
        return [str(item) for item in organs if str(item).strip()]
    combination = _as_dict(product_data.get("combination_risks"))
    organ_overlap = _as_dict(combination.get("organ_overlap"))
    return [str(item) for item in _as_list(organ_overlap.get("overlapping_organs")) if str(item).strip()]


def _risk_breakdown(product_data: Dict[str, Any]) -> Dict[str, Any]:
    counts = _summary_counts(product_data)
    return {
        "critical": counts["critical_count"],
        "high": counts["high_count"],
        "moderate": counts["moderate_count"],
        "low": counts["low_count"],
        "safe": counts["safe_count"],
    }


def _normalize_investigation_report(full_report: dict) -> dict:
    product_data = _first_product_data(full_report)
    if not product_data:
        return {
            "report_status": "minimal",
            "summary": {"text": "We do not yet have enough verified data to complete this analysis."},
            "key_findings": [],
            "organ_impact": [],
            "risk_breakdown": {},
        }

    normalized = product_data.copy()
    summary = _as_dict(normalized.get("summary"))
    summary.setdefault("text", _summary_text(normalized))
    summary.update(_summary_counts(normalized))
    normalized["summary"] = summary
    normalized["report_status"] = _report_status(normalized)
    normalized["key_findings"] = _key_findings(normalized)
    normalized["organ_impact"] = _organ_impact(normalized)
    normalized["risk_breakdown"] = _risk_breakdown(normalized)
    return normalized


def _normalize_filtering_report(full_report: dict) -> dict:
    product_data = _first_product_data(full_report)
    if not product_data:
        return {
            "report_status": "minimal",
            "chemicals": [],
            "safe_skipped": [],
            "summary": "No verified ingredient filtering data is available yet.",
        }

    ingredients = _as_dict(product_data.get("ingredients"))
    chemicals = [chem.get("name") for chem in _as_list(ingredients.get("chemicals_evaluated")) if isinstance(chem, dict) and chem.get("name")]
    safe_skipped = _as_list(ingredients.get("safe_skipped"))
    status = _report_status(product_data)
    summary = _summary_text(product_data)
    return {
        "report_status": status,
        "chemicals": chemicals,
        "safe_skipped": safe_skipped,
        "summary": summary,
    }


# ----------------------------------------------------------------------
# 4. Global agent singleton (servers started once)
# ----------------------------------------------------------------------
_global_agent: Optional[BiologicalAgent] = None

def get_global_agent() -> BiologicalAgent:
    """Return a singleton BiologicalAgent instance (servers started once)."""
    global _global_agent
    if _global_agent is None:
        _global_agent = BiologicalAgent(start_servers=True)
        print("✅ Global MCP agent started (servers running).")
    return _global_agent


# ----------------------------------------------------------------------
# 5. Helper functions for extracting parts of the report
# ----------------------------------------------------------------------
def extract_filtering_report(full_report: dict) -> dict:
    return _normalize_filtering_report(full_report)


def extract_investigation_report(full_report: dict) -> dict:
    return _normalize_investigation_report(full_report)


def normalize_cumulative_report(report_data: dict) -> dict:
    if not report_data:
        return {
            "report_status": "minimal",
            "overall_assessment": "We do not yet have enough verified cumulative data to summarize the current profile.",
            "key_warnings": [],
            "global_summary": {},
            "scoring_analysis": {},
        }

    normalized = dict(report_data)
    normalized["report_status"] = "complete" if normalized.get("overall_assessment") else "partial"
    normalized.setdefault(
        "overall_assessment",
        "A partial cumulative assessment is available. Verified findings will deepen as more products are analyzed.",
    )
    normalized.setdefault("key_warnings", [])
    normalized.setdefault("global_summary", normalized.get("global_summary") or {})
    normalized.setdefault("scoring_analysis", normalized.get("scoring_analysis") or {})
    return normalized


def get_reports_folder() -> Path:
    reports_path = Path(settings.BASE_DIR) / "reports"
    reports_path.mkdir(exist_ok=True)
    return reports_path


# ----------------------------------------------------------------------
# 6. Main analysis function (single product)
# ----------------------------------------------------------------------
def analyze_ingredients_risks(
    ingredients_list: List[str],
    user_type: str = None,
    user_id: int = None,
    product_id: int = None,
    agent: Optional[BiologicalAgent] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, Any], List[str], str]:
    if not ingredients_list:
        logger.info("No ingredients provided, returning empty risks")
        return [], {}, [], ""

    # Use provided agent, otherwise get the global singleton
    if agent is None:
        agent = get_global_agent()
    else:
        logger.info("Using provided agent instance.")

    logger.info(f"Analyzing {len(ingredients_list)} ingredients with BiologicalAgent")
    saved_file_path = ""

    try:
        product = {
            "product_id": "django_scan",
            "product_name": "Product from Scan",
            "product_usage": "cosmetic",
            "exposure_type": "skin",
            "ingredient_list": [{"name": ing} for ing in ingredients_list]
        }

        result = agent.run_sync([product], user_type=user_type)
        report = result.get("report", {})

        risk_items = []
        for product_out in report.get("products", []):
            for chem in product_out.get("ingredients", {}).get("chemicals_evaluated", []):
                name = chem.get("name")
                danger = chem.get("verdict", {}).get("danger_level", "UNKNOWN")
                if danger in ("CRITICAL", "HIGH"):
                    level = "high"
                elif danger == "MODERATE":
                    level = "medium"
                else:
                    level = "low"
                risk_items.append({"ingredient": name, "level": level})

        try:
            reports_dir = get_reports_folder()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if user_id is not None and product_id is not None:
                filename = f"user_{user_id}_product_{product_id}_{timestamp}.json"
            else:
                first_ing = ingredients_list[0] if ingredients_list else "empty"
                safe_name = "".join(c for c in first_ing if c.isalnum())[:20]
                filename = f"agent_report_{timestamp}_{safe_name}.json"
            filepath = reports_dir / filename
            saved_file_path = str(filepath)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp,
                    "ingredients": ingredients_list,
                    "user_type": user_type,
                    "risk_items": risk_items,
                    "full_report": report
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Agent report saved to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save agent report to disk: {e}")

        # Return empty debug_log list (no extra debug logs collected)
        return risk_items, report, [], saved_file_path

    finally:
        # Do NOT close the agent if it's the global singleton
        # Only close if we created a temporary agent (but we never do here)
        pass


# ----------------------------------------------------------------------
# 7. Cumulative analysis function (multiple products)
# ----------------------------------------------------------------------
def analyze_cumulative_risks(
    products_with_reports: List[Dict[str, Any]],
    user_type: str = None,
    timeout_seconds: int = 600,
    agent: Optional[BiologicalAgent] = None,
) -> Dict[str, Any]:
    """
    Run cumulative analysis by calling the agent directly with multiple products.
    Prints detailed results to console and saves a JSON report.
    """
    import time
    import json
    from datetime import datetime
    from django.conf import settings

    start_time = time.time()
    print(f"🚀 Starting cumulative analysis with {len(products_with_reports)} products (timeout={timeout_seconds}s)")

    if not products_with_reports or len(products_with_reports) < 2:
        return {"error": "Cumulative analysis requires at least 2 products"}

    # ------------------------------------------------------------------
    # 1. Build the product list in the format the agent expects
    # ------------------------------------------------------------------
    agent_products = []
    for p in products_with_reports:
        agent_products.append({
            "product_id": p["product_id"],
            "product_name": p["product_name"],
            "product_usage": p.get("product_usage", "cosmetic"),
            "exposure_type": p.get("exposure_type", "skin"),
            "ingredient_list": p.get("ingredient_list", [])
        })

    print("📦 Agent products:")
    print(json.dumps(agent_products, indent=2))

    # ------------------------------------------------------------------
    # 2. Get the agent (reuse global singleton or use provided)
    # ------------------------------------------------------------------
    if agent is None:
        agent = get_global_agent()
        print("♻️ Using global agent for cumulative analysis (servers already running).")
    else:
        print("♻️ Using provided agent.")

    # ------------------------------------------------------------------
    # 3. Run the agent with the full product list
    # ------------------------------------------------------------------
    try:
        result = agent.run_sync(agent_products, user_type=user_type)
        report = result.get("report", {})

        # --- Print cumulative results to console (same as single but with cross‑product info) ---
        print("\n" + "="*70)
        print("📊 CUMULATIVE ANALYSIS RESULTS")
        print("="*70)

        # Global summary
        global_summary = report.get("global_summary", {})
        print(f"\n🌍 Global Summary:")
        print(f"   Products analysed: {len(agent_products)}")
        print(f"   Products to avoid: {global_summary.get('products_to_avoid', 0)}")
        print(f"   Products to reduce: {global_summary.get('products_to_reduce', 0)}")
        print(f"   High risk chemicals: {global_summary.get('high_chemicals', [])}")
        print(f"   Organs under pressure: {global_summary.get('organs_under_pressure', [])}")

        # Organ overlap (global analysis)
        organ_analysis = global_summary.get("organ_global_analysis", {})
        if organ_analysis:
            print("\n🧠 Organ Overlap (cross‑product):")
            for organ, data in organ_analysis.items():
                print(f"   {organ}: {data.get('total_unique_count', 0)} unique chemicals")
        else:
            print("\n🧠 No organ overlap detected.")

        # Scoring analysis (product rankings, recurrence)
        scoring = report.get("scoring_analysis", {})
        ranked = scoring.get("ranked_products", [])
        if ranked:
            print("\n🏆 Product Rankings (by risk score):")
            for r in ranked:
                print(f"   #{r['rank']}: {r['product_name']} - {r['verdict']} (score: {r['total_product_score']})")
        else:
            print("\n🏆 No product rankings available.")

        recurrence = scoring.get("recurrence_risks", [])
        if recurrence:
            print("\n⚠️ Recurrence Risks (chemicals in multiple products):")
            for r in recurrence:
                print(f"   {r['chemical']} appears in {r['frequency']} products (score: {r['recurrence_score']})")
        else:
            print("\n✅ No recurrence risks.")

        # Also print per‑product verdicts (like in single‑product analysis)
        product_verdicts = report.get("product_verdicts", [])
        if product_verdicts:
            print("\n📦 Product Verdicts:")
            for pv in product_verdicts:
                print(f"   {pv['product_name']}: {pv['risk_level']} - {pv['recommendation']}")
                if pv.get('risk_drivers'):
                    print(f"      Drivers: {', '.join(pv['risk_drivers'])}")

        print("\n" + "="*70)

        # ------------------------------------------------------------------
        # 4. Extract risk items for backward compatibility (optional)
        # ------------------------------------------------------------------
        risk_items = []
        for product_out in report.get("products", []):
            for chem in product_out.get("ingredients", {}).get("chemicals_evaluated", []):
                name = chem.get("name")
                danger = chem.get("verdict", {}).get("danger_level", "UNKNOWN")
                if danger in ("CRITICAL", "HIGH"):
                    level = "high"
                elif danger == "MODERATE":
                    level = "medium"
                else:
                    level = "low"
                risk_items.append({
                    "ingredient": name,
                    "level": level,
                    "product_id": product_out.get("product_id")
                })

        # ------------------------------------------------------------------
        # 5. Save the cumulative report (JSON file)
        # ------------------------------------------------------------------
        try:
            reports_dir = Path(settings.BASE_DIR) / "reports"
            reports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_name = agent_products[0].get("product_name", "cumulative")[:20]
            filename = f"cumulative_report_{timestamp}_{first_name}.json"
            filepath = reports_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp,
                    "user_type": user_type,
                    "product_count": len(agent_products),
                    "risk_items": risk_items,
                    "full_report": report
                }, f, indent=2, ensure_ascii=False)
            print(f"💾 Cumulative report saved to {filepath}")
        except Exception as e:
            print(f"⚠️ Could not save cumulative report: {e}")

        # Print timing summary
        elapsed = time.time() - start_time
        print(f"\n✅ Cumulative analysis completed in {elapsed:.1f}s")

        return report

    except Exception as e:
        import traceback
        print(f"❌ Cumulative analysis failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 8. API communication helpers
# ----------------------------------------------------------------------
def send_report_to_recommendation_api(
    report_dict: Dict[str, Any],
    base_url: str = "http://127.0.0.1:8000"
) -> Dict[str, Any]:
    url = f"{base_url}/api/recommend/research/report"
    try:
        response = requests.post(url, json=report_dict, timeout=60)
        if response.status_code == 200:
            logger.info("Recommendation API called successfully.")
            return response.json()
        else:
            logger.warning(f"Recommendation API returned {response.status_code}: {response.text[:200]}")
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        logger.warning("Recommendation API timed out after 60 seconds")
        return {"error": "Timeout"}
    except Exception as e:
        logger.error(f"Failed to call recommendation API: {e}")
        return {"error": str(e)}


def should_trigger_recommendation_api(report_dict: Dict[str, Any]) -> bool:
    product_verdicts = report_dict.get("product_verdicts", [])
    for pv in product_verdicts:
        rec = pv.get("recommendation", "").lower()
        risk_level = pv.get("risk_level", "").upper()
        if rec in ["reduce", "reduce_use", "eliminate"]:
            return True
        if risk_level in ["HIGH", "CRITICAL", "MODERATE"]:
            return True

    scoring = report_dict.get("scoring_analysis", {})
    for pr in scoring.get("product_risk_results", []):
        verdict = pr.get("verdict", "").upper()
        if verdict in ["HIGH", "CRITICAL", "MODERATE"]:
            return True

    return False


# ----------------------------------------------------------------------
# 9. Test (if run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_ingredients = ["Lysine", "Formaldehyde", "AQUA"]
    risk_items, report, debug_log, filepath = analyze_ingredients_risks(test_ingredients, user_type="fetal")
    print("\n=== FINAL RISK ITEMS ===")
    for item in risk_items:
        print(f"  {item['ingredient']}: {item['level']}")
    print(f"\nDebug log length: {len(debug_log)} lines")
    print(f"Report saved to: {filepath}")