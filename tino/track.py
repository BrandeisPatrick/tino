"""Run tracking. Every run gets a directory under runs/ that is committed and pushed to GitHub as it progresses;
weights go to the Hugging Face Hub; Weights & Biases is used too when the machine is logged in."""
import json
import os
import platform
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PATHS = ["tino", "scripts", "configs", "pyproject.toml", "uv.lock"]


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def code_state():
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_files": dirty.splitlines()}


def environment():
    import mlx.core as mx
    import mlx_lm
    info = {"python": platform.python_version(), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "machine": platform.machine(), "os": platform.platform()}
    try:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["memory_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30)
    except Exception:
        pass
    return info


class Run:
    def __init__(self, name, config, allow_dirty=False, sync=True):
        state = code_state()
        if state["dirty"] and not allow_dirty:
            raise SystemExit("Code has uncommitted changes. Commit first so the run is reproducible, or pass --allow-dirty.\n"
                             + "\n".join(state["dirty_files"]))
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + name
        self.dir = os.path.join(REPO_ROOT, "runs", self.id)
        self.rel = os.path.relpath(self.dir, REPO_ROOT)
        self.sync_enabled = sync
        os.makedirs(self.dir, exist_ok=True)
        self.write("config.json", config)
        self.write("provenance.json", {"run_id": self.id, "code": state, "environment": environment(),
                                       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        self.wandb = None
        if os.environ.get("TINO_WANDB", "1") != "0":
            try:
                import wandb
                if wandb.api.api_key:
                    self.wandb = wandb.init(project="tino", name=self.id, config=config, dir=os.path.join(REPO_ROOT, ".wandb"))
            except Exception:
                self.wandb = None

    def write(self, name, obj):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=1)

    def log(self, name, record, step=None):
        with open(os.path.join(self.dir, name), "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb is not None:
            flat = {f"{name.split('.')[0]}/{k}": v for k, v in record.items() if isinstance(v, (int, float))}
            self.wandb.log(flat, step=step)

    def sync(self, message):
        """Commit only this run's directory and push. Failures warn instead of killing training."""
        if not self.sync_enabled:
            return
        try:
            git("add", self.rel)
            if subprocess.run(["git", "diff", "--cached", "--quiet", "--", self.rel], cwd=REPO_ROOT).returncode == 0:
                return
            git("commit", "-m", f"run {self.id}: {message}", "--", self.rel)
            git("push", "-q", "origin", "HEAD")
        except Exception as e:
            print(f"[track] sync failed (will retry at next sync): {e}", flush=True)
