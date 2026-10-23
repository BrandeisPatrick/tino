"""Keep training from taking the whole Mac down.

MLX's own memory limit only raises once RAM and swap are exhausted, which is too late on a 16 GB laptop. This guard
checks two things before every optimizer step and stops cleanly instead:

- MLX memory: active + cached GPU memory over `mlx_cap_gb` clears the cache; active memory alone over the cap aborts.
- The whole system: macOS's free-memory level (kern.memorystatus_level, the "free percentage" memory_pressure prints)
  under `pause_free_pct` pauses training until other apps release memory; under `abort_free_pct`, or swap growing by
  more than `max_swap_growth_gb` since start, or a pause lasting `max_pause_min`, aborts.
An abort raises ResourceAbort; the trainer logs it, pushes the run record, and exits with the last good checkpoint kept.
"""
import re
import subprocess
import time

import mlx.core as mx

GB = 2**30


class ResourceAbort(RuntimeError):
    pass


def free_pct():
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)([MG])", out)
    return 0.0 if not m else float(m.group(1)) / (1024 if m.group(2) == "M" else 1)


class ResourceGuard:
    def __init__(self, mlx_cap_gb=9.0, cache_gb=1.0, pause_free_pct=25, abort_free_pct=12, max_swap_growth_gb=3.0,
                 max_pause_min=30, log=print):
        self.cap = mlx_cap_gb * GB
        self.pause_free, self.abort_free = pause_free_pct, abort_free_pct
        self.max_swap_growth, self.max_pause = max_swap_growth_gb, max_pause_min * 60
        self.log = log
        mx.set_memory_limit(int(self.cap))
        mx.set_cache_limit(int(cache_gb * GB))
        self.swap0 = swap_used_gb()
        self.paused_sec = 0.0
        self.min_free_seen = free_pct()

    def snapshot(self):
        return {"free_pct": free_pct(), "swap_gb": round(swap_used_gb(), 2), "mlx_active_gb": round(mx.get_active_memory() / GB, 2),
                "mlx_cache_gb": round(mx.get_cache_memory() / GB, 2), "mlx_peak_gb": round(mx.get_peak_memory() / GB, 2)}

    def check(self):
        if mx.get_active_memory() + mx.get_cache_memory() > self.cap:
            mx.clear_cache()
        if mx.get_active_memory() > self.cap:
            raise ResourceAbort(f"MLX active memory {mx.get_active_memory()/GB:.1f} GB is over the {self.cap/GB:.1f} GB cap")
        growth = swap_used_gb() - self.swap0
        if growth > self.max_swap_growth:
            raise ResourceAbort(f"swap grew {growth:.1f} GB since start (limit {self.max_swap_growth} GB)")
        f = free_pct()
        self.min_free_seen = min(self.min_free_seen, f)
        if f < self.abort_free:
            raise ResourceAbort(f"system free memory at {f}% (abort below {self.abort_free}%)")
        if f < self.pause_free:
            mx.clear_cache()
            self.log(f"[guard] system free memory {f}% < {self.pause_free}%: pausing until it recovers")
            t0 = time.time()
            while free_pct() < self.pause_free + 5:
                if free_pct() < self.abort_free:
                    raise ResourceAbort(f"system free memory fell to {free_pct()}% while paused")
                if self.paused_sec + time.time() - t0 > self.max_pause:
                    raise ResourceAbort(f"paused for more than {self.max_pause/60:.0f} min waiting for memory")
                time.sleep(10)
            self.paused_sec += time.time() - t0
            self.log(f"[guard] resumed after {time.time()-t0:.0f}s, free memory {free_pct()}%")
