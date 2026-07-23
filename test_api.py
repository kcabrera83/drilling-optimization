"""Tests for drilling-optimization API."""

import sys
import json

sys.path.insert(0, ".")

from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

TESTS_PASSED = 0
TESTS_FAILED = 0


def test(name, method, url, body=None, expect_status=200):
    global TESTS_PASSED, TESTS_FAILED
    try:
        if method == "GET":
            resp = client.get(url)
        elif method == "POST":
            resp = client.post(url, json=body)
        else:
            resp = client.get(url)
        data = resp.json()
        ok = resp.status_code == expect_status and data.get("status") == "ok"
        status = "PASS" if ok else "FAIL"
        if ok:
            TESTS_PASSED += 1
        else:
            TESTS_FAILED += 1
        print(f"  [{status}] {name} ({resp.status_code})")
        if not ok:
            print(f"         {data}")
    except Exception as e:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}: {e}")


def main():
    global TESTS_PASSED, TESTS_FAILED

    print("=" * 60)
    print("  Tests - Drilling Optimization")
    print("=" * 60)

    print("\n[1/4] Health test...")
    test("GET /api/health", "GET", "/api/health")

    print("\n[2/4] Models test...")
    test("GET /api/models", "GET", "/api/models")

    print("\n[3/4] Prediction test...")
    body = {
        "depth_m": 2000, "wob_klbf": 20, "rpm": 120,
        "flow_rate_gpm": 500, "mud_weight_ppg": 10.5,
        "formation": "arena", "bit_type": "polycrystalline",
        "bit_diameter_in": 12.25,
    }
    test("POST /api/predict", "POST", "/api/predict", body)

    print("\n[4/4] Optimization test...")
    test("POST /api/optimize", "POST", "/api/optimize",
         {"depth_m": 2000, "flow_rate_gpm": 500, "mud_weight_ppg": 10.5, "formation": "arena"})

    print("\n" + "=" * 60)
    total = TESTS_PASSED + TESTS_FAILED
    print(f"  Result: {TESTS_PASSED}/{total} tests passed")
    if TESTS_FAILED == 0:
        print("  All tests passed")
    else:
        print(f"  {TESTS_FAILED} test(s) failed")
    print("=" * 60)
    return 0 if TESTS_FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
