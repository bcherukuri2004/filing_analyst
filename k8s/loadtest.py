"""Send a steady stream of searches to the cluster and record every answer.

Used to check that requests keep working while a pod is deleted or the app is
updated. It never calls /ask, so it spends no API credits.

    FILING_ANALYST_API_KEY=... python k8s/loadtest.py --seconds 60 > result.json

Every /search is recorded (time, HTTP status or error). A second thread asks
/health twice a second and records which version answered."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

QUESTIONS = ["What was NVIDIA total revenue in fiscal 2026?", "How much did Apple spend on research in 2025?",
             "What were Microsoft's total assets in 2026?", "Amazon operating income 2025",
             "Alphabet net income 2024"]


def call(url: str, body: dict | None, key: str, timeout: float) -> int | str:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:  # noqa: BLE001  (a refused or cut connection is a failure to record)
        return type(e).__name__


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=2, help="searches in flight at once")
    p.add_argument("--timeout", type=float, default=30)
    args = p.parse_args()
    key = os.environ["FILING_ANALYST_API_KEY"]
    start = time.time()
    stop = start + args.seconds
    results: list[tuple[float, int | str]] = []
    versions: list[tuple[float, str]] = []
    lock = threading.Lock()

    def searcher(n: int) -> None:
        i = n
        while time.time() < stop:
            q = QUESTIONS[i % len(QUESTIONS)]
            i += 1
            status = call(args.url + "/search", {"question": q, "k": 3, "rerank": False}, key, args.timeout)
            with lock:
                results.append((round(time.time() - start, 2), status))

    def watcher() -> None:
        while time.time() < stop:
            try:
                with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
                    v = json.loads(r.read()).get("version", "?")
            except Exception as e:  # noqa: BLE001
                v = "error:" + type(e).__name__
            versions.append((round(time.time() - start, 2), v))
            time.sleep(0.5)

    threads = [threading.Thread(target=searcher, args=(n,)) for n in range(args.workers)]
    threads.append(threading.Thread(target=watcher))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [(t, s) for t, s in results if s != 200]
    changes = [versions[0]] + [v for prev, v in zip(versions, versions[1:]) if v[1] != prev[1]] if versions else []
    print(json.dumps({"seconds": args.seconds, "workers": args.workers, "requests": len(results),
                      "ok": len(results) - len(failed), "failed": len(failed), "failures": failed[:50],
                      "version_changes": changes}, indent=1))


if __name__ == "__main__":
    main()
