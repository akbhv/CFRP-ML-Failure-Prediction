
import time
import statistics
import json
import csv
from pathlib import Path

from fastapi.testclient import TestClient
from api.main import app

# Same example inputs for both endpoints
ml_payload = {
    "Nx": 100000.0,
    "Ny": 20000.0,
    "Nxy": 10000.0,
    "Mx": 0.0,
    "My": 0.0,
    "Mxy": 0.0,
}

analysis_payload = {
    "material": {
        "E1": 135e9,
        "E2": 10e9,
        "G12": 5e9,
        "nu12": 0.30,
    },
    "strengths": {
        "Xt": 1500e6,
        "Xc": 900e6,
        "Yt": 27e6,
        "Yc": 200e6,
        "S": 80e6,
    },
    "layup": [0, 45, -45, 90, 90, -45, 45, 0],
    "ply_thickness": 0.000125,
    "loads": ml_payload,
}

WARMUP_RUNS = 5
MEASURED_RUNS = 50
OUTPUT_FILE = Path("results/speed_benchmark.csv")


def benchmark_endpoint(client, endpoint, payload):
    # Warm-up: do not include these timings in the results.
    for _ in range(WARMUP_RUNS):
        response = client.post(endpoint, json=payload)
        response.raise_for_status()

    timings_ms = []

    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        response = client.post(endpoint, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.raise_for_status()
        timings_ms.append(elapsed_ms)

    timings_ms.sort()

    return {
        "endpoint": endpoint,
        "runs": MEASURED_RUNS,
        "median_ms": statistics.median(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
        "min_ms": min(timings_ms),
        "p95_ms": timings_ms[int(0.95 * (len(timings_ms) - 1))],
        "max_ms": max(timings_ms),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Models are loaded when api.main is imported, before timing starts.
    with TestClient(app) as client:
        # Confirm both endpoints work before benchmarking.
        ml_check = client.post("/ml-predict", json=ml_payload)
        classical_check = client.post("/analyze", json=analysis_payload)

        ml_check.raise_for_status()
        classical_check.raise_for_status()

        ml_data = ml_check.json()
        classical_data = classical_check.json()

        print("ML model version:", ml_data["model"]["version"])
        print("ML criteria returned:", len(ml_data["predictions"]))
        print(
            "Classical criteria returned:",
            len(classical_data["governing_results"]),
        )

        rows = [
            benchmark_endpoint(client, "/ml-predict", ml_payload),
            benchmark_endpoint(client, "/analyze", analysis_payload),
        ]

    ml_time = next(
        row["median_ms"] for row in rows
        if row["endpoint"] == "/ml-predict"
    )
    classical_time = next(
        row["median_ms"] for row in rows
        if row["endpoint"] == "/analyze"
    )

    speedup = classical_time / ml_time if ml_time > 0 else float("nan")

    print("\n--- SPEED BENCHMARK ---")
    for row in rows:
        print(f"\nEndpoint: {row['endpoint']}")
        for key in ("runs", "median_ms", "mean_ms", "min_ms",
                    "p95_ms", "max_ms"):
            print(f"  {key}: {row[key]:.4f}"
                  if key != "runs"
                  else f"  {key}: {row[key]}")

    print(f"\nMedian-time ratio (classical / ML): {speedup:.2f}x")
    print("Note: this is an in-process API benchmark, not a network test.")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
