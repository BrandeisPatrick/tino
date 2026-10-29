#!/usr/bin/env python
"""Render the article figures from the tracked results.

Inputs, all in the repository: docs/results.json (written by scripts/collect_results.py),
the per-question prediction files under runs/<id>/predictions/, and the zero-shot
evaluation in runs/20261023-185358-think-eval/final.json. Outputs: paper/figures/*.svg
(inlined by scripts/build_article.py), *.png for a quick look, and stats.json with the
paired-difference numbers the captions quote.

    uv run python scripts/figures.py
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

# Ink and chrome. build_article.py swaps these exact values for theme tokens, so keep
# every colour in the figures one of the constants below.
INK = "#1b1b1b"
INK2 = "#5c5b57"
MUTED = "#8a8882"
GRID = "#e3e2dc"
AXIS = "#c4c3bb"
SURFACE = "#ffffff"
TINT = "#f3f2ee"
TEACHER = "#6f6d67"
# One categorical slot per model, fixed order (validated palette; see docs/TRACKING.md).
C = {
    "adapter": "#2a78d6",
    "finetuned": "#eb6834",
    "distil": "#1baf7a",
    "distil_a1": "#eda100",
    "adapter_post": "#e87ba4",
    "thinking": "#6250d6",
}
# Ordinal ramp for the three test splits, easiest to hardest.
ORD = ["#86b6ef", "#256abf", "#104281"]

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "xtick.major.size": 3, "ytick.major.size": 0, "xtick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "svg.fonttype": "none", "axes.facecolor": "none", "figure.facecolor": "none",
})

RESULTS = json.load(open(ROOT / "docs" / "results.json"))
RUNS: dict[str, dict] = {}
for _r in sorted(RESULTS["runs"], key=lambda r: r["id"]):
    if _r.get("smoke"):
        continue
    if _r["name"] not in RUNS or _r["status"] == "finished":
        RUNS[_r["name"]] = _r
TEACHER_RUN = RUNS["teacher-qwen3-4b"]
THINK = json.load(open(ROOT / "runs" / "20261023-185358-think-eval" / "final.json"))

SPLITS = [
    ("decision-v7/test", "Trained sources", "decision-v7 test, n = 1,440"),
    ("transfer-v4/test", "New sources", "transfer-v4 test, n = 764"),
    ("transfer-v9/test", "Hardest", "transfer-v9 test, n = 1,264"),
]
MODELS = [
    ("tino-finetuned", "full fine-tune, gold labels", ["tino-finetuned", "tino-finetuned-s1"], C["finetuned"]),
    ("tino-adapter", "LoRA r16, gold labels", ["tino-adapter", "tino-adapter-s1"], C["adapter"]),
    ("tino-adapter-post", "LoRA r16, post-trained start", ["tino-adapter-post"], C["adapter_post"]),
    ("tino-distil", "LoRA r16, half gold, half teacher", ["tino-distil", "tino-distil-s1"], C["distil"]),
    ("tino-distil-a1", "LoRA r16, teacher only", ["tino-distil-a1", "tino-distil-a1-s1"], C["distil_a1"]),
]
KEV_BAR = {"decision-v7/test": 0.808, "transfer-v4/test": 0.642}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", facecolor="none", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", facecolor="white", dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def metric(run, split, key="acc"):
    m = run["metrics"].get(split)
    return None if m is None else m.get(key)


def preds(run, split):
    path = ROOT / run["dir"] / "predictions" / (split.replace("/", "--") + ".jsonl.gz")
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def scored(rows):
    """(rec_idx, qid) -> (correct, confidence of the chosen option)."""
    out = {}
    for r in rows:
        z = np.asarray(r["logits"], dtype=float)
        p = np.exp(z - z.max())
        p /= p.sum()
        out[(r["rec_idx"], r["qid"])] = (int(int(p.argmax()) == int(r["label"])), float(p.max()))
    return out


def paired_stats(diff, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, n)].mean()
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    m = wins + losses
    if m == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"n": n, "mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "wins": wins, "losses": losses, "p": p}


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


def dot_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- figure 2
def fig_accuracy():
    labels = [f"{m[0]}\n{m[1]}" for m in MODELS] + ["Qwen3-4B-Instruct\nteacher, zero-shot"]
    n = len(labels)
    ys = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    fig.subplots_adjust(wspace=0.10)
    for ax, (split, title, sub) in zip(axes, SPLITS):
        ax.set_title(title, loc="left", pad=15)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, ha="left", va="bottom")
        xs_all = []
        for y, (name, _, runs, color) in zip(ys, MODELS):
            vals = [v for v in (metric(RUNS[r], split) for r in runs) if v is not None]
            xs_all += vals
            if len(vals) > 1:
                ax.plot(sorted(vals), [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)
            ax.scatter(vals, [y] * len(vals), s=40, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.annotate(f"{np.mean(vals):.3f}", (max(vals), y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        tv = metric(TEACHER_RUN, split)
        xs_all.append(tv)
        ax.scatter([tv], [ys[-1]], s=40, marker="D", color=TEACHER, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.annotate(f"{tv:.3f}", (tv, ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
        if split in KEV_BAR:
            ax.axvline(KEV_BAR[split], color=AXIS, lw=0.9, zorder=1)
            ax.text(KEV_BAR[split], -1.0, f"Kev-0.6B {KEV_BAR[split]:.3f}", ha="center", va="center",
                    fontsize=7.5, color=MUTED)
        lo, hi = min(xs_all), max(xs_all)
        ax.set_xlim(lo - 0.025, hi + 0.075)
        ax.xaxis.set_major_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        dot_axis(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-1.5, n - 0.5)
    axes[1].set_xlabel("Accuracy on the test split. Dots are seeds; the number is their mean.")
    save(fig, "fig2-accuracy")


# ----------------------------------------------------------------------------- figure 3
def fig_paired():
    rows = []
    stats = {}
    comps = [
        ("Adapter vs full fine-tune  (RQ2)", C["adapter"],
         [("tino-adapter", "tino-finetuned"), ("tino-adapter-s1", "tino-finetuned-s1")]),
        ("Half-teacher distillation vs gold-label adapter  (RQ4)", C["distil"],
         [("tino-distil", "tino-adapter"), ("tino-distil-s1", "tino-adapter-s1")]),
        ("Teacher-only distillation vs gold-label adapter  (RQ4)", C["distil_a1"],
         [("tino-distil-a1", "tino-adapter"), ("tino-distil-a1-s1", "tino-adapter-s1")]),
        ("Post-trained start vs base start, one seed  (RQ3b)", C["adapter_post"],
         [("tino-adapter-post", "tino-adapter")]),
    ]
    for title, color, pairs in comps:
        rows.append((title, color, None))
        for split, sname, sub in SPLITS:
            diffs = []
            for a, b in pairs:
                pa, pb = preds(RUNS[a], split), preds(RUNS[b], split)
                if pa is None or pb is None:
                    continue
                sa, sb = scored(pa), scored(pb)
                keys = sorted(set(sa) & set(sb))
                diffs.append(np.array([sa[k][0] - sb[k][0] for k in keys], dtype=float))
            if not diffs:
                continue
            st = paired_stats(np.concatenate(diffs))
            st["seeds"] = len(diffs)
            rows.append((f"{sname}, {sub.split(',')[0]}", color, st))
            stats[f"{title.strip()} | {split}"] = st
    rows.append(("Thinking vs direct answering, zero-shot Qwen3-4B  (RQ3a)", C["thinking"], None))

    def think_scores(model):
        out = {}
        for p in THINK["models"][model]["predictions"]:
            z = np.asarray(p["logits"], dtype=float)
            out[(p["suite"], p["rec_idx"], p["qid"])] = int(int(z.argmax()) == int(p["label"]))
        return out

    si, sth = think_scores("qwen3-4b-instruct"), think_scores("qwen3-4b-thinking")
    for suite, sname in [("decision-v7/development", "Trained sources, development"),
                         ("transfer-v4/development", "New sources, development")]:
        keys = sorted(k for k in si if k[0] == suite and k in sth)
        st = paired_stats(np.array([sth[k] - si[k] for k in keys], dtype=float))
        st["seeds"] = 1
        rows.append((f"{sname}, {len(keys)} questions", C["thinking"], st))
        stats[f"Thinking vs direct | {suite}"] = st

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.4, 0.33 * n + 0.8))
    y = n - 1
    los, his = [], []
    for label, color, st in rows:
        if st is None:
            ax.text(-0.015, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8.8, fontweight="bold", color=INK)
        else:
            m, lo, hi = 100 * st["mean"], 100 * st["lo"], 100 * st["hi"]
            los.append(lo)
            his.append(hi)
            ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)
            ax.scatter([m], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.text(-0.015, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8.5, color=INK2)
            ax.text(1.015, y, f"{m:+.1f}  [{lo:+.1f}, {hi:+.1f}]   {fmt_p(st['p'])}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=INK2)
        y -= 1
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(math.floor(min(los)) - 1, math.ceil(max(his)) + 1)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Difference in accuracy, percentage points (first minus second), with a 95% bootstrap interval "
                  "and a two-sided sign-test p; both seeds pooled where two exist")
    fig.subplots_adjust(left=0.30, right=0.74)
    save(fig, "fig3-paired")
    return stats


# ----------------------------------------------------------------------------- figure 4
def fig_calibration():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.9), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.34})
    a.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=1)
    for name, _, runs, color in MODELS:
        if name == "tino-adapter-post":
            continue
        conf, cor = [], []
        for r in runs:
            p = preds(RUNS[r], "transfer-v4/test")
            if p is None:
                continue
            for c_, cf in scored(p).values():
                cor.append(c_)
                conf.append(cf)
        conf, cor = np.array(conf), np.array(cor)
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 25:
                xs.append(conf[m].mean())
                ys.append(cor[m].mean())
        a.plot(xs, ys, color=color, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label=name,
               solid_joinstyle="round")
    a.set_xlim(0.2, 1.0)
    a.set_ylim(0.2, 1.0)
    a.set_aspect("equal")
    a.set_xlabel("Confidence of the chosen option, raw (no temperature)")
    a.set_ylabel("Accuracy of questions in the bin")
    a.set_title("a. Reliability on new sources, raw probabilities", loc="left")
    a.legend(loc="upper left")
    a.grid()
    a.set_axisbelow(True)
    a.text(0.98, 0.23, "below the diagonal: over-confident", ha="right", va="bottom", fontsize=7.5, color=MUTED)

    rows = [(m[0], m[2], m[3]) for m in MODELS] + [("4B teacher, zero-shot", None, TEACHER)]
    ys = np.arange(len(rows))[::-1]
    for y, (name, runs, color) in zip(ys, rows):
        if runs is None:
            vals = [TEACHER_RUN["teacher_temperature"]]
            b.scatter(vals, [y], s=40, marker="D", color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        else:
            vals = [RUNS[r]["temperature"] for r in runs if RUNS[r].get("temperature")]
            b.scatter(vals, [y] * len(vals), s=40, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        b.annotate(", ".join(f"{v:.2f}" for v in sorted(vals)), (max(vals), y), xytext=(7, 0),
                   textcoords="offset points", va="center", fontsize=8, color=INK2)
    b.set_xscale("log")
    b.set_xlim(0.6, 22)
    b.set_xticks([0.7, 1, 1.5, 2, 3, 5, 10])
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.xaxis.set_minor_locator(NullLocator())
    b.axvline(1, color=INK2, lw=0.9, zorder=1)
    b.text(1, len(rows) - 0.35, "T = 1: calibrated as trained", ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.set_yticks(ys)
    b.set_yticklabels([r[0] for r in rows])
    b.set_ylim(-0.6, len(rows) + 0.3)
    b.set_xlabel("Temperature fitted on the calibration split (log scale); above 1 means over-confident")
    b.set_title("b. Over-confidence of every full-data run", loc="left")
    dot_axis(b)
    save(fig, "fig4-calibration")


# ----------------------------------------------------------------------------- figure 5
ABLATION = [
    ("Baseline", [("LoRA r16, all projections, lr 1e-4, cross-entropy", ["abl-adapter-4k", "abl-adapter-4k-s1"])]),
    ("Method", [("Full fine-tune, lr 1e-5", ["abl-finetuned-4k"])]),
    ("Adapter", [("Rank 4", ["abl-rank4-4k"]),
                 ("Rank 64, alpha scaled with rank", ["abl-rank64-4k"]),
                 ("Rank 64, alpha fixed", ["abl-rank64-a32-4k"]),
                 ("Rank 16, attention projections only", ["abl-attn-only-4k", "abl-attn-only-4k-s1"])]),
    ("Learning rate", [("5e-5", ["abl-lr5e-5-4k"]), ("2e-4", ["abl-lr2e-4-4k", "abl-lr2e-4-4k-s1"])]),
    ("Loss", [("Cross-entropy + Brier", ["abl-brier-4k"])]),
    ("Teacher weight", [("0.25", ["abl-distil-a025-4k"]),
                        ("0.5", ["abl-distil-a05-4k", "abl-distil-a05-4k-s1"]),
                        ("1.0, no gold labels", ["abl-distil-a1-4k"])]),
    ("1,000 questions", [("LoRA r16", ["abl-adapter-1k"]), ("Full fine-tune", ["abl-finetuned-1k"])]),
]


def fig_ablation():
    rows = []
    for group, items in ABLATION:
        rows.append((group, None))
        rows.extend(items)
    n = len(rows)
    ys = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 0.3 * n + 1.0), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.30)
    panels = [("decision-v7/development", "Trained sources (development, n = 1,468)"),
              ("transfer-v4/development", "New sources (development, n = 764)")]
    base = RUNS["abl-adapter-4k"], RUNS["abl-adapter-4k-s1"]
    for ax, (split, title) in zip(axes, panels):
        ax.set_title(title, loc="left")
        collapsed_max = max(metric(RUNS[r], split) for r in RUNS
                            if RUNS[r].get("collapsed") and metric(RUNS[r], split))
        uniform = collapsed_max + 0.012
        ax.axvspan(0, uniform, color=TINT, zorder=0)
        ax.text(uniform, ys[0] + 0.55, " uniform-guess level", ha="left", va="center", fontsize=7.5, color=MUTED)
        bm = np.mean([metric(b, split) for b in base])
        ax.axvline(bm, color=AXIS, lw=0.9, zorder=1)
        for y, (label, runs) in zip(ys, rows):
            if ax is axes[0]:
                ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=8.8 if runs is None else 8.5, fontweight="bold" if runs is None else "normal",
                        color=INK if runs is None else INK2)
            if runs is None:
                continue
            vals, collapsed = [], []
            for i, r in enumerate(runs):
                run = RUNS[r]
                v = metric(run, split)
                if run.get("collapsed"):
                    collapsed.append((v, run["seed"]))
                    ax.scatter([v], [y], s=46, marker="x", color=INK2, linewidth=1.4, zorder=3)
                    continue
                vals.append(v)
                ax.scatter([v], [y], s=40, color=C["adapter"] if i == 0 else SURFACE, edgecolor=C["adapter"],
                           linewidth=1.3, zorder=3)
                if v < uniform + 0.03:
                    ax.annotate("near uniform", (v, y), xytext=(7, 0), textcoords="offset points",
                                va="center", fontsize=7.5, color=MUTED)
            if collapsed:
                if len(runs) == 1:
                    note = "collapsed"
                elif len(collapsed) == len(runs):
                    note = "collapsed, both seeds"
                else:
                    note = f"collapsed, seed {collapsed[0][1]}"
                ax.annotate(note, (max(v for v, _ in collapsed), y), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=MUTED)
            if len(vals) == 2 and min(vals) >= uniform + 0.03:
                ax.plot(sorted(vals), [y, y], color=C["adapter"], lw=1.4, zorder=2)
        lo = min(metric(RUNS[r], split) for r in RUNS if r.startswith("abl-"))
        hi = max(metric(RUNS[r], split) for r in RUNS if r.startswith("abl-"))
        ax.set_xlim(lo - 0.03, hi + 0.06)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        dot_axis(ax)
    axes[0].set_yticks([])
    axes[0].set_ylim(-0.7, n - 0.2)
    handles = [
        Line2D([], [], marker="o", color=C["adapter"], mec=C["adapter"], ls="", ms=6.5, label="seed 0"),
        Line2D([], [], marker="o", color=SURFACE, mec=C["adapter"], mew=1.3, ls="", ms=6.5, label="seed 1"),
        Line2D([], [], marker="x", color=INK2, ls="", ms=6.5, mew=1.4, label="collapsed to uniform"),
        Line2D([], [], color=AXIS, lw=0.9, label="baseline mean"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8)
    axes[0].set_xlabel("Development accuracy after 2 epochs on a 4,000-question subset (1,000 for the last group)")
    axes[0].xaxis.set_label_coords(1.04, -0.05)
    save(fig, "fig5-ablation")


# ----------------------------------------------------------------------------- figure 6
def fig_cost():
    ft = [RUNS["tino-finetuned"], RUNS["tino-finetuned-s1"]]
    ad = [RUNS["tino-adapter"], RUNS["tino-adapter-s1"]]
    panels = [
        ("Trained weights", lambda r: r["trainable_params"] / 1e6, "{:.1f}M"),
        ("Published file", lambda r: r.get("weights_mb"), "{:,.0f} MB"),
        ("Peak training memory", lambda r: r["peak_gb"], "{:.1f} GB"),
        ("Training time, 2 epochs", lambda r: r["train_minutes"], "{:.0f} min"),
        ("Fitted temperature", lambda r: r["temperature"], "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.0))
    fig.subplots_adjust(wspace=0.25)
    for i, (ax, (title, f, fmt)) in enumerate(zip(axes, panels)):
        top = 0
        for y, (runs, color, label) in enumerate([(ad, C["adapter"], "tino-adapter"), (ft, C["finetuned"], "tino-finetuned")]):
            vals = [v for v in (f(r) for r in runs) if v is not None]
            mean = float(np.mean(vals))
            top = max(top, max(vals))
            ax.barh(y, mean, height=0.52, color=color, zorder=2)
            if len(vals) > 1 and max(vals) - min(vals) > 1e-9:
                ax.scatter(vals, [y] * len(vals), s=18, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.text(max(vals), y, "  " + fmt.format(mean), va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, top * 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["tino-adapter", "tino-finetuned"] if i == 0 else [])
    save(fig, "fig6-cost")


# ----------------------------------------------------------------------------- figure 7
SOURCE_NAMES = {
    "contrastive_deadline": "Deadline rules (held-out)", "contrastive_authorization": "Authorization rules (held-out)",
    "composition_held_and_or": "Held-out and/or compositions", "contrastive_return_window": "Return-window rules",
    "contrastive_quantity_limit": "Quantity-limit rules", "contrastive_age_eligibility": "Age-eligibility rules",
    "contrastive_spend_threshold": "Spend-threshold rules", "agnews": "AG News topic", "agnews_yn": "AG News yes/no",
    "amazon": "Amazon stars", "boolq": "BoolQ", "dbpedia14": "DBpedia class", "imdb": "IMDB sentiment", "mnli": "MNLI",
    "sst5": "SST-5 sentiment", "trec": "TREC question type", "yelp": "Yelp stars", "yelp_yn": "Yelp yes/no",
    "emotion": "Emotion", "mmlu": "MMLU", "paws": "PAWS paraphrase", "qnli": "QNLI", "sciq": "SciQ",
    "tweet_offensive": "Tweet offensive",
}


def fig_thinking():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    inst = THINK["models"]["qwen3-4b-instruct"]
    think = THINK["models"]["qwen3-4b-thinking"]
    rows = []
    for suite in ("decision-v7/development", "transfer-v4/development"):
        for src, d in inst["suites"][suite]["by_source"].items():
            if d["n"] < 6:
                continue
            t = think["suites"][suite]["by_source"][src]["accuracy"]
            rows.append((SOURCE_NAMES.get(src, src), d["n"], d["accuracy"], t, suite))
    rows.sort(key=lambda r: (r[3] - r[2], r[3]))
    ys = np.arange(len(rows))
    for y, (name, n, di, th, suite) in zip(ys, rows):
        a.plot([di, th], [y, y], color=AXIS, lw=1.6, zorder=1)
        a.scatter([di], [y], s=36, marker="D", color=TEACHER, edgecolor=SURFACE, linewidth=1.1, zorder=3)
        if abs(th - di) < 1e-9:
            a.scatter([th], [y], s=64, facecolors="none", edgecolors=C["thinking"], linewidth=1.6, zorder=4)
        else:
            a.scatter([th], [y], s=44, color=C["thinking"], edgecolor=SURFACE, linewidth=1.1, zorder=4)
    a.set_yticks(ys)
    a.set_yticklabels([f"{r[0]}  (n = {r[1]})" for r in rows])
    a.set_xlim(0.1, 1.05)
    a.xaxis.set_major_locator(MultipleLocator(0.2))
    a.set_xlabel("Accuracy on the 200-question development subset")
    a.set_title("a. Where reasoning first helps, by source", loc="left")
    dot_axis(a)
    handles = [Line2D([], [], marker="D", color=TEACHER, ls="", ms=6, label="answers directly  (0.33 s per question)"),
               Line2D([], [], marker="o", color=C["thinking"], ls="", ms=7, label="reasons first  (31.6 s per question, 95×)"),
               Line2D([], [], marker="o", color=SURFACE, mec=C["thinking"], mew=1.6, ls="", ms=7.5, label="same accuracy either way")]
    a.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=8)

    for r, color, label in [("tino-adapter", C["adapter"], "from Qwen3-0.6B-Base  (tino-adapter)"),
                            ("tino-adapter-post", C["adapter_post"], "from post-trained Qwen3-0.6B  (tino-adapter-post)")]:
        run = RUNS[r]
        seen, xs, ys_ = set(), [], []
        for s, v in run["dev_curve"]:
            if s in seen:
                continue
            seen.add(s)
            xs.append(s)
            ys_.append(v)
        b.plot(xs, ys_, color=color, lw=2, marker="o", ms=4, mec=SURFACE, mew=0.9, label=label)
        best = dict(zip(xs, ys_)).get(run["best_step"])
        if best is not None:
            b.scatter([run["best_step"]], [best], s=90, facecolors="none", edgecolors=color, linewidth=1.5, zorder=4)
    b.set_xlabel("Optimizer step (two epochs)")
    b.set_ylabel("Development accuracy, trained sources")
    b.set_title("b. Base or post-trained start, same adapter recipe", loc="left")
    b.legend(loc="lower right", fontsize=8)
    b.grid(axis="y")
    b.set_axisbelow(True)
    b.text(0.02, 0.97, "ring: checkpoint selected for the test sets", transform=b.transAxes, ha="left", va="top",
           fontsize=7.5, color=MUTED)
    save(fig, "fig7-thinking")


# ----------------------------------------------------------------------------- figure 8
def fig_alpha():
    groups = [(0.0, ["tino-adapter", "tino-adapter-s1"]), (0.5, ["tino-distil", "tino-distil-s1"]),
              (1.0, ["tino-distil-a1", "tino-distil-a1-s1"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.6))
    fig.subplots_adjust(wspace=0.25)
    for ax, key, ylabel, title in [(a, "acc", "Test accuracy (higher is better)", "a. Accuracy against the teacher's share of the loss"),
                                   (b, "brier", "Brier score on the test split (lower is better)", "b. Probability quality against the teacher's share")]:
        for (split, sname, sub), color in zip(SPLITS, ORD):
            xs, means = [], []
            for alpha, runs in groups:
                vals = [metric(RUNS[r], split, key) for r in runs]
                ax.scatter([alpha] * len(vals), vals, s=34, color=color, edgecolor=SURFACE, linewidth=1.1, zorder=3)
                xs.append(alpha)
                means.append(float(np.mean(vals)))
            ax.plot(xs, means, color=color, lw=2, zorder=2, label=f"{sname.lower()} ({sub.split(',')[0]})")
            if key == "acc":
                tv = metric(TEACHER_RUN, split, "acc")
                ax.scatter([1.34], [tv], s=36, marker="D", color=color, edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ticks = [0, 0.5, 1.0] + ([1.34] if key == "acc" else [])
        labels = ["0\ngold labels only\ntino-adapter", "0.5\ntino-distil", "1.0\nteacher only\ntino-distil-a1"] + \
                 (["4B teacher\nzero-shot"] if key == "acc" else [])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.12, 1.5 if key == "acc" else 1.12)
        ax.set_xlabel("Teacher weight α in   α · KL(teacher ‖ student) + (1 − α) · CE(gold)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    a.legend(loc="center left", fontsize=8)
    save(fig, "fig8-alpha")


# ----------------------------------------------------------------------------- figure 9
def fig_curves():
    fig, ax = plt.subplots(figsize=(10.6, 3.7))
    for name, _, runs, color in MODELS:
        for i, r in enumerate(runs):
            run = RUNS[r]
            seen, xs, ys = set(), [], []
            for s, v in run["dev_curve"]:
                if s in seen:
                    continue
                seen.add(s)
                xs.append(s)
                ys.append(v)
            ax.plot(xs, ys, color=color, lw=2 if i == 0 else 1.3, alpha=1 if i == 0 else 0.5,
                    marker="o", ms=3.6 if i == 0 else 3, mec=SURFACE, mew=0.8, label=name if i == 0 else None)
            best = dict(zip(xs, ys)).get(run["best_step"])
            if best is not None:
                ax.scatter([run["best_step"]], [best], s=80, facecolors="none", edgecolors=color, linewidth=1.4,
                           alpha=1 if i == 0 else 0.6, zorder=4)
    ax.set_xlabel("Optimizer step; two epochs over 15,576 questions at 2,048 padded tokens per batch")
    ax.set_ylabel("Development accuracy, trained sources (n = 1,468)")
    ax.set_ylim(0.66, 0.83)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    ax.text(0.01, 0.97, "bold: seed 0; faint: seed 1; ring: checkpoint selected for the test sets",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    save(fig, "fig9-training")


# ----------------------------------------------------------------------------- figure 1
def fig_schematic():
    W, H = 980, 352
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    p.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK2}"/></marker></defs>')

    def label(x, y, text, size=9.5, color=INK2, weight="normal", anchor="start", spacing=1):
        p.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" letter-spacing="{spacing}">{text}</text>')

    def chip(x, y, w, text, fill=TINT, stroke=AXIS, color=INK, size=11.5, h=26):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}"/>')
        p.append(f'<text x="{x + 9}" y="{y + h / 2 + 4}" font-size="{size}" fill="{color}">{text}</text>')

    def box(x, y, w, h, title, lines, accent=None):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{AXIS}"/>')
        if accent:
            p.append(f'<rect x="{x + 10}" y="{y + 10}" width="9" height="9" rx="2" fill="{accent}"/>')
        p.append(f'<text x="{x + (26 if accent else 12)}" y="{y + 19}" font-size="12" font-weight="bold" fill="{INK}">{title}</text>')
        for i, line in enumerate(lines):
            p.append(f'<text x="{x + 12}" y="{y + 36 + 14 * i}" font-size="10.5" fill="{INK2}">{line}</text>')

    def arrow(x1, y1, x2, y2):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1" marker-end="url(#arrow)"/>')

    # Input row
    label(20, 16, "INPUT · ONE CAUSAL ROW PER QUESTION")
    chip(20, 24, 600, "⟨state⟩  order placed 9 days ago · delivered 2 days later · returns accepted within 14 days of delivery")
    chip(20, 58, 300, "⟨question⟩  Can this order still be returned?")
    chip(20, 92, 112, "⟨opt⟩ yes ⟨/opt⟩")
    chip(142, 92, 106, "⟨opt⟩ no ⟨/opt⟩")
    chip(258, 92, 184, "⟨opt⟩ needs review ⟨/opt⟩")
    chip(452, 92, 86, "⟨decide⟩", fill=SURFACE, stroke=INK2)
    for x in (76, 195, 350, 495):
        arrow(x, 120, x, 138)
    # Backbone
    p.append(f'<rect x="20" y="140" width="600" height="40" rx="6" fill="{SURFACE}" stroke="{INK2}"/>')
    p.append(f'<text x="32" y="157" font-size="12" font-weight="bold" fill="{INK}">Qwen3-0.6B-Base</text>')
    p.append(f'<text x="32" y="172" font-size="10.5" fill="{INK2}">28 layers · hidden size 1,024 · a single forward pass, no generated text</text>')
    # Hidden states
    for x, t in ((76, "h(⟨/opt⟩₁)"), (195, "h(⟨/opt⟩₂)"), (350, "h(⟨/opt⟩₃)"), (495, "h(⟨decide⟩)")):
        p.append(f'<line x1="{x}" y1="180" x2="{x}" y2="198" stroke="{INK2}" stroke-width="1"/>')
        p.append(f'<circle cx="{x}" cy="204" r="6" fill="{SURFACE}" stroke="{INK2}" stroke-width="1.2"/>')
        p.append(f'<text x="{x}" y="223" font-size="9.5" fill="{INK2}" text-anchor="middle">{t}</text>')
        p.append(f'<line x1="{x}" y1="226" x2="{x}" y2="238" stroke="{INK2}" stroke-width="1"/>')
    # Pointer head
    p.append(f'<rect x="20" y="240" width="600" height="40" rx="6" fill="{SURFACE}" stroke="{INK2}"/>')
    p.append(f'<text x="32" y="257" font-size="12" font-weight="bold" fill="{INK}">Pointer head, trained from scratch</text>')
    p.append(f'<text x="32" y="272" font-size="10.5" fill="{INK2}">scoreᵢ = ⟨ W<tspan font-size="8">k</tspan> h(⟨/opt⟩ᵢ) , W<tspan font-size="8">q</tspan> h(⟨decide⟩) ⟩ / √256 for each option, then softmax over the options</text>')
    arrow(320, 280, 320, 298)
    # Output
    label(20, 312, "OUTPUT · ONE PROBABILITY PER OPTION, TEMPERATURE-SCALED")
    for i, (t, v) in enumerate((("yes", 0.81), ("no", 0.12), ("needs review", 0.07))):
        x0 = 20 + i * 205
        p.append(f'<text x="{x0}" y="334" font-size="11" fill="{INK}">{t}</text>')
        p.append(f'<rect x="{x0 + 84}" y="325" width="{int(100 * v)}" height="11" rx="2" fill="{C["adapter"]}"/>')
        p.append(f'<text x="{x0 + 90 + int(100 * v)}" y="334" font-size="10.5" fill="{INK2}">{v:.2f}</text>')
    # Three routes
    label(650, 16, "THREE WAYS TO TRAIN IT")
    box(650, 24, 310, 92, "Full fine-tune → tino-finetuned",
        ["every backbone weight and the head: 441M parameters",
         "Adafactor for the backbone (lr 1e-5), AdamW for the head",
         "2 epochs · 2,048 padded tokens per batch · 6.5–7.7 GB peak",
         "gold labels, cross-entropy"], accent=C["finetuned"])
    box(650, 128, 310, 92, "Adapter → tino-adapter",
        ["backbone frozen; LoRA rank 16 on q, k, v, o, gate, up, down",
         "10.6M trained parameters · a 43 MB file",
         "AdamW (lr 1e-4) · same schedule · 4.0 GB peak",
         "gold labels, cross-entropy"], accent=C["adapter"])
    box(650, 232, 310, 106, "Distillation → tino-distil, tino-distil-a1",
        ["adapter recipe; the teacher is Qwen3-4B-Instruct-2507 (4-bit)",
         "it reads the same question as lettered multiple choice;",
         "its letter probabilities, softened at T = 9, are soft labels",
         "loss = α · KL(teacher ‖ student) + (1 − α) · CE(gold)",
         "α = 0.5 (tino-distil) or 1, no gold labels at all (tino-distil-a1)"], accent=C["distil"])
    p.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig1-method.svg").write_text("\n".join(p), encoding="utf-8")
    print("wrote fig1-method")


def main():
    fig_schematic()
    fig_accuracy()
    stats = fig_paired()
    fig_calibration()
    fig_ablation()
    fig_cost()
    fig_thinking()
    fig_alpha()
    fig_curves()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("wrote stats.json")


if __name__ == "__main__":
    main()
