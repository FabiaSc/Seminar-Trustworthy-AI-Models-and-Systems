# scripts/load_test_stub.py
"""
Simple load test stub for SDXL-Turbo FP16 vs MixDQ.

We support two arrival patterns:
- "burst" (all requests are ready immediately; existing behaviour),
- "poisson" (requests arrive over time with exponential inter-arrival times).

We measure end-to-end latency for each request and report p50 / p95.
"""

import time
import math
import random
import subprocess
import concurrent.futures as cf
from typing import List, Tuple

from scripts.controller import ResolutionController

# Base commands for FP16 and quantized runs
BASE_FP16 = [
    "python",
    "scripts/txt2img.py",
    "--config",
    "./configs/stable-diffusion/sdxl_turbo.yaml",
    "--base_path",
    "./logs/sdxl_mixdq_eval",
    "--batch_size",
    "1",
    "--num_imgs",
    "1",
    "--fp16",
    "--prompt",
    "a corgi in sunglasses",
]

BASE_QUANT = [
    "python",
    "scripts/quant_txt2img.py",
    "--base_path",
    "./logs/sdxl_mixdq_eval",
    "--batch_size",
    "1",
    "--num_imgs",
    "1",
    "--fp16",
    "--wbits",
    "8",
    "--prompt",
    "a corgi in sunglasses",
]

# ResolutionController is available for future extensions (not used directly here)
rc = ResolutionController(q_low=1, q_high=4, default=1024)


def infer_once(queue_len: int, quant: bool) -> float:
    """
    Launch one txt2img / quant_txt2img process with the given queue_len.

    Returns:
        Latency in milliseconds.
    """
    base_cmd = BASE_QUANT if quant else BASE_FP16
    # Appends queue_len so the downstream script can adapt resolution and behaviour
    cmd = list(base_cmd) + ["--queue_len", str(queue_len)]

    t0 = time.perf_counter()
    # Silences stdout and stderr to avoid flooding the console during load tests
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms


def compute_p50_p95(samples: List[float]) -> Tuple[float, float]:
    """Return empirical p50 / p95 from a list of samples."""
    if not samples:
        return float("nan"), float("nan")
    xs = sorted(samples)
    n = len(xs)

    def q(p: float) -> float:
        idx = int(p * (n - 1))
        return xs[idx]

    return q(0.50), q(0.95)


def run_batch(
    concurrency: int,
    total_requests: int,
    adaptive: bool,
    quant: bool,
) -> Tuple[float, float]:
    """
    Existing behaviour: submit `total_requests` as fast as possible, up to
    `concurrency` in-flight workers. queue_len is either 0 (static) or
    approximated by in-flight (adaptive).
    """
    lat: List[float] = []
    futures = set()
    in_flight = 0

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        # Starts an initial wave of jobs up to the concurrency limit
        while in_flight < concurrency and total_requests > 0:
            q_for_call = max(in_flight, 0) if adaptive else 0
            futures.add(ex.submit(infer_once, q_for_call, quant))
            in_flight += 1
            total_requests -= 1

        # As futures complete, backfills new jobs until total_requests is exhausted
        while futures:
            done, not_done = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            futures = not_done
            for f in done:
                try:
                    lat.append(f.result())
                except Exception:
                    lat.append(float("nan"))
                in_flight -= 1

                if total_requests > 0:
                    q_for_call = max(in_flight, 0) if adaptive else 0
                    futures.add(ex.submit(infer_once, q_for_call, quant))
                    in_flight += 1
                    total_requests -= 1

    p50, p95 = compute_p50_p95(lat)
    return p50, p95


def run_poisson(
    lam: float,
    total_requests: int,
    concurrency: int,
    adaptive: bool,
    quant: bool,
) -> Tuple[float, float]:
    """
    More realistic load test: requests arrive as a Poisson process with rate `lam`
    (requests per second). Inter-arrival times are exponential with mean 1/lam.

    Behaviour:
    - generates arrival times T_0, T_1, ..., T_{N-1} with Exp(lam),
    - submits jobs at wall-clock times that respect those arrivals,
    - limits parallelism with `concurrency`,
    - approximates queue_len by current in_flight if `adaptive=True`,
      otherwise keeps queue_len=0 (static policy).
    """
    lat: List[float] = []
    futures = set()
    in_flight = 0

    # Pre-generates inter-arrival times so randomness is fixed before the loop
    inter_arrivals = []
    for _ in range(total_requests):
        u = random.random()
        ia = -math.log(1.0 - u) / lam  # Exp(lam) in seconds
        inter_arrivals.append(ia)

    arrival_times = []
    t = 0.0
    for ia in inter_arrivals:
        t += ia
        arrival_times.append(t)

    t_start = time.perf_counter()
    next_req_idx = 0  # index of next request to submit

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        while next_req_idx < total_requests or futures:
            now = time.perf_counter()
            elapsed = now - t_start

            # Submits all arrivals that are due and fit within the concurrency limit
            while (
                next_req_idx < total_requests
                and arrival_times[next_req_idx] <= elapsed
                and in_flight < concurrency
            ):
                if adaptive:
                    queue_len = max(in_flight, 0)
                else:
                    queue_len = 0

                futures.add(ex.submit(infer_once, queue_len, quant))
                in_flight += 1
                next_req_idx += 1

            # Collects completed futures without blocking on the remaining ones
            done = {f for f in futures if f.done()}
            for f in done:
                futures.remove(f)
                try:
                    lat.append(f.result())
                except Exception:
                    lat.append(float("nan"))
                in_flight -= 1

            # Short sleep to avoid busy-waiting when no new work is ready
            time.sleep(0.005)

    p50, p95 = compute_p50_p95(lat)
    return p50, p95


if __name__ == "__main__":
    # 1) BURST – FP16 vs W8, static vs adaptive queue_len
    for conc in (1, 2):  # can be reduced if VRAM or runtime is tight
        # FP16
        p50_s_fp, p95_s_fp = run_batch(
            concurrency=conc, total_requests=16,
            adaptive=False, quant=False
        )
        p50_a_fp, p95_a_fp = run_batch(
            concurrency=conc, total_requests=16,
            adaptive=True, quant=False
        )
        print(
            f"[BURST FP16  conc={conc}] "
            f"STATIC p50={p50_s_fp:.1f}ms p95={p95_s_fp:.1f}ms  |  "
            f"ADAPTIVE p50={p50_a_fp:.1f}ms p95={p95_a_fp:.1f}ms"
        )

        # W8 (MixDQ)
        p50_s_w8, p95_s_w8 = run_batch(
            concurrency=conc, total_requests=16,
            adaptive=False, quant=True
        )
        p50_a_w8, p95_a_w8 = run_batch(
            concurrency=conc, total_requests=16,
            adaptive=True, quant=True
        )
        print(
            f"[BURST W8    conc={conc}] "
            f"STATIC p50={p50_s_w8:.1f}ms p95={p95_s_w8:.1f}ms  |  "
            f"ADAPTIVE p50={p50_a_w8:.1f}ms p95={p95_a_w8:.1f}ms"
        )

    # 2) POISSON – FP16 vs W8, static vs adaptive queue_len
    for lam in (0.5, 1.0, 2.0, 4.0):  # req/s
        # FP16
        p50_s_fp, p95_s_fp = run_poisson(
            lam=lam, total_requests=32,
            concurrency=2, adaptive=False, quant=False
        )
        p50_a_fp, p95_a_fp = run_poisson(
            lam=lam, total_requests=32,
            concurrency=2, adaptive=True, quant=False
        )
        print(
            f"[POISSON FP16 λ={lam:.1f} req/s] "
            f"STATIC p50={p50_s_fp:.1f}ms p95={p95_s_fp:.1f}ms  |  "
            f"ADAPTIVE p50={p50_a_fp:.1f}ms p95={p95_a_fp:.1f}ms"
        )

        # W8 (MixDQ)
        p50_s_w8, p95_s_w8 = run_poisson(
            lam=lam, total_requests=32,
            concurrency=2, adaptive=False, quant=True
        )
        p50_a_w8, p95_a_w8 = run_poisson(
            lam=lam, total_requests=32,
            concurrency=2, adaptive=True, quant=True
        )
        print(
            f"[POISSON W8   λ={lam:.1f} req/s] "
            f"STATIC p50={p50_s_w8:.1f}ms p95={p95_s_w8:.1f}ms  |  "
            f"ADAPTIVE p50={p50_a_w8:.1f}ms p95={p95_a_w8:.1f}ms"
        )
