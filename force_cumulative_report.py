import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"

# Login (use your actual credentials)
response = requests.post(f"{BASE_URL}/users/auth/login/", json={"email": "maram@gmail.com", "password": "maram"})
token = response.json()["token"]
headers = {"Authorization": f"Token {token}"}

# Get all approved/saved products
response = requests.get(f"{BASE_URL}/products/my-products/?status=approved", headers=headers)
products = response.json().get("products", [])
print(f"Found {len(products)} approved/saved products")

if len(products) < 2:
    print("Need at least 2 approved products")
    exit()

# Build products_with_reports list (same as scan does)
from products.models import Product
from risk.services import analyze_cumulative_risks

# But you can't import Django models in a standalone script easily.
# Instead, call the endpoint that triggers cumulative analysis: scan a dummy image.
# Or simply trigger by creating a new product via API.

# Create a dummy product to trigger cumulative
dummy = {
    "name": "Trigger Product",
    "brand": "Test",
    "category": "cosmetic",
    "ingredients": ["Aqua", "Glycerin"],
    "barcode": "TRIGGER123",
    "extraction_method": "manual"
}
response = requests.post(f"{BASE_URL}/products/", headers=headers, json=dummy)
if response.status_code == 201:
    product_id = response.json()["id"]
    print(f"Created trigger product {product_id}")
    # Approve it (so it becomes part of approved/saved)
    requests.patch(f"{BASE_URL}/products/{product_id}/decision/", headers=headers, json={"decision": "approved"})

# Now scan a product (use a real image) to trigger cumulative analysis
with open("C:\Users\maram\Downloads\New folder\media\scans\Mixa_QR_OfmTwZL.png", "rb") as f:
    files = {"image": f}
    response = requests.post(f"{BASE_URL}/scan/", headers=headers, files=files)
    print(f"Scan status: {response.status_code}")
    if response.status_code == 201:
        result = response.json()
        print("Cumulative report from scan response:")
        print(json.dumps(result.get("cumulative_report"), indent=2))

# Get final ai_report
response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
print("\nFinal ai_report:")
print(json.dumps(response.json().get("ai_report"), indent=2))