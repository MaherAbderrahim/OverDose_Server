import json
from pathlib import Path
from risk.services import send_report_to_recommendation_api

report_file = Path("reports/user_1_product_4_20260508_185913.json")

if not report_file.exists():
    print(f"❌ File not found: {report_file}")
else:
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(f"✅ Loaded report from {report_file}")

    payload = report.get("full_report", report)
    result = send_report_to_recommendation_api(payload, base_url="http://127.0.0.1:8000")

    print("\n📤 Response from /api/recommend/research/report:")
    print(json.dumps(result, indent=2))