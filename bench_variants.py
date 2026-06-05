"""
Compare hot inference speed between the original and enpt q4 models.
"""
import statistics as stats
import time

import numpy as np

from e5 import E5


def bench(model):
    qtext = "how much protein should a female eat per day"
    raw = ["the cat sat on the mat number %d and learned to embed text" % i for i in range(512)]

    for _ in range(8):
        model.query(qtext)
    lat = []
    for _ in range(80):
        t0 = time.time()
        model.query(qtext)
        lat.append((time.time() - t0) * 1000.0)

    for _ in range(4):
        model.passage(raw[:64])
    batch_runs = []
    for _ in range(6):
        t0 = time.time()
        model.passage(raw)
        batch_runs.append(time.time() - t0)

    best = min(batch_runs)
    med = stats.median(batch_runs)
    return {
        "query_ms_median": stats.median(lat),
        "query_ms_p95": float(np.percentile(lat, 95)),
        "batch_best_s": best,
        "batch_median_s": med,
        "throughput_best": 512.0 / best,
        "throughput_median": 512.0 / med,
    }


def report(label, metrics):
    print(label)
    for k, v in metrics.items():
        print(f"  {k}={v:.3f}")


def main():
    results = {
        "original": bench(E5(variant="original")),
        "enpt": bench(E5(variant="enpt")),
    }
    report("original", results["original"])
    report("enpt", results["enpt"])
    print("ratios")
    for key in ("query_ms_median", "query_ms_p95", "batch_best_s", "batch_median_s"):
        print(f"  enpt/original {key}={results['enpt'][key] / results['original'][key]:.3f}")
    for key in ("throughput_best", "throughput_median"):
        print(f"  enpt/original {key}={results['enpt'][key] / results['original'][key]:.3f}")


if __name__ == "__main__":
    main()
