# scripts/controller_policy.py
from dataclasses import dataclass

# Controls image resolution based on queue length and offline latency stats
@dataclass
class ResolutionController:
    q_low: int = 1            # lower queue threshold for hysteresis
    q_high: int = 4           # upper queue threshold for hysteresis
    default: int = 1024       # default resolution
    # p95 offline latency in ms, can be updated with fresh measurements
    p95_by_res_ms: dict = None
    target_p95_ms: float = 1000.0  # SLO: p95 <= 1.0 s

    def __post_init__(self):
        # Fills default p95 values if none are provided and initializes state
        if self.p95_by_res_ms is None:
            self.p95_by_res_ms = {512: 133.0, 768: 223.3, 1024: 371.2}
        self._r = self.default

    def choose(self, queue_len: int) -> int:
        # Picks a resolution using queue-length-based hysteresis
        if queue_len > self.q_high:
            self._r = 512
        elif queue_len < self.q_low:
            self._r = 1024
        else:
            self._r = 768
        return self._r

    def best_res_for_slo(self) -> int:
        # Returns the highest resolution whose p95 latency satisfies the SLO
        ok = [r for r, p in self.p95_by_res_ms.items() if p <= self.target_p95_ms]
        return max(ok) if ok else min(self.p95_by_res_ms)
