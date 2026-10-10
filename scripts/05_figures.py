"""Redraw Figures 1 and 2 from the JSON that scripts/02_results.py wrote.

Every point is read from results/paper/, so a figure cannot drift away from the
numbers in the tables. Run the `scaling`, `tuning` and `phrases` stages first.

    python scripts/05_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # imported after the backend is set

OUT = Path("results/paper")
FIGS = Path("paper/figures")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(name: str):
    p = OUT / f"{name}.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p} -- run: python scripts/02_results.py --stage {name}"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def curve(rows: list[dict]) -> tuple[list[float], list[float]]:
    """A filter's (false positive, detection) trace, as plotted.

    The sweeps run until both rates reach zero; the tail is a stack of identical
    points on the origin, so repeats are collapsed to one.
    """
    pts = [(r["fp"], r["detect"]) for r in rows]
    kept = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    return [x for x, _ in kept], [y for _, y in kept]


def at(rows: list[dict], key: str, value) -> dict:
    """The sweep row where `key` equals `value` -- a project's shipped default."""
    for r in rows:
        if r[key] == value:
            return r
    raise KeyError(f"no row with {key}={value}")


def figure_scaling(scaling) -> None:
    """Fig. 1: both rates rise with corpus size; their ratio does not."""
    sizes = [r["corpus"] for r in scaling]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.4))

    for key, style, color, label in [
        ("s1_det", "-o", BLUE, "s1, flagged"),
        ("s1_fp", "--o", BLUE, "s1, false positive"),
        ("ot3_det", "-s", ORANGE, "OpenThoughts3, flagged"),
        ("ot3_fp", "--s", ORANGE, "OpenThoughts3, false positive"),
    ]:
        dashed = style.startswith("--")
        ax1.plot(
            sizes,
            [r[key] for r in scaling],
            style,
            color=color,
            lw=1.4 if dashed else 1.6,
            ms=3.5,
            alpha=0.55 if dashed else 1.0,
            label=label,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Training documents")
    ax1.set_ylabel("Benchmark items (%)")
    ax1.set_title("(a) Both rates rise with corpus size", loc="left")
    ax1.set_ylim(0, 45)
    ax1.legend(frameon=False, loc="upper left", handlelength=1.8, borderpad=0.1)

    s1_frac = [r["s1_frac"] for r in scaling]
    ot3_frac = [r["ot3_frac"] for r in scaling]
    ax2.plot(sizes, s1_frac, "-o", color=BLUE, lw=1.8, ms=4, label="s1")
    ax2.plot(sizes, ot3_frac, "-s", color=ORANGE, lw=1.8, ms=4, label="OpenThoughts3")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 0.8)
    ax2.set_xlabel("Training documents")
    ax2.set_ylabel("False positives / flagged")
    ax2.set_title("(b) The error fraction does not", loc="left")

    # Label the lines in place rather than with a second legend. The offsets are
    # hand-placed; the anchors come from the data.
    i_s1, i_ot3 = len(sizes) - 2, sizes.index(100_000)
    ax2.annotate(
        "s1",
        (sizes[i_s1], s1_frac[i_s1]),
        xytext=(0, 8),
        textcoords="offset points",
        color=BLUE,
        fontsize=7.5,
        ha="center",
    )
    ax2.annotate(
        "OpenThoughts3",
        (sizes[i_ot3], ot3_frac[i_ot3]),
        xytext=(0, -13),
        textcoords="offset points",
        color=ORANGE,
        fontsize=7.5,
        ha="center",
    )

    ticks = [s for s in sizes if s != 150_000]
    for axis in (ax1, ax2):
        axis.grid(True, alpha=0.7, lw=0.5)
        axis.set_axisbelow(True)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{size / 1000:g}k" for size in ticks])
        axis.minorticks_off()

    fig.savefig(FIGS / "scaling.pdf")
    fig.savefig(FIGS / "scaling.png", dpi=220)
    print(f"wrote {FIGS}/scaling.{{pdf,png}}")


def figure_frontier(tuning, phrases) -> None:
    """Fig. 2: the precision/recall frontier, derived without labels."""
    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    for rows, style, color, lw, ms, label in [
        (tuning["word"], "-o", BLUE, 1.6, 3.5, r"$n$-gram, whitespace"),
        (tuning["qwen"], "-s", ORANGE, 1.6, 3.5, r"$n$-gram, Qwen2 tokens"),
        (tuning["fuzz"], "-^", AQUA, 1.9, 4.5, "fuzz.ratio (whole string)"),
        (phrases, "-D", YELLOW, 1.6, 3.5, r"phrase agreement ($\geq$ frac of 20)"),
    ]:
        x, y = curve(rows)
        ax.plot(x, y, style, color=color, lw=lw, ms=ms, label=label)

    # Ring each project's shipped default, read from the sweep it sits on.
    s1 = at(tuning["word"], "n", 8)
    ot3 = at(tuning["qwen"], "n", 13)
    ot114k = at(tuning["fuzz"], "threshold", 95)
    loose = at(tuning["fuzz"], "threshold", 65)
    for row, marker, color in [
        (s1, "o", BLUE),
        (ot3, "s", ORANGE),
        (ot114k, "^", AQUA),
    ]:
        ax.plot(row["fp"], row["detect"], marker, mfc="none", mec=color, ms=10, mew=1.4)

    for row, text, xytext, color in [
        (s1, r"s1 ships here ($n$=8)", (9.2, 17.0), BLUE),
        (ot3, "OpenThoughts3 ships\nhere ($n$=13)", (12.2, 31.5), ORANGE),
        (
            ot114k,
            "OpenThoughts-114K ships here\n(cutoff 95): finds nothing",
            (7.6, 1.2),
            AQUA,
        ),
        (loose, "same method,\ncutoff 65", (4.4, 17.0), AQUA),
    ]:
        ax.annotate(
            text,
            (row["fp"], row["detect"]),
            xytext=xytext,
            fontsize=6.5,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
        )

    ax.set_xlabel("False positive rate (%)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_xlim(-0.8, 20)  # n=5 sits far off to the right; see the caption
    ax.set_ylim(-1.5, 40)
    ax.grid(True, alpha=0.7, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderpad=0.1)

    fig.savefig(FIGS / "frontier.pdf")
    fig.savefig(FIGS / "frontier.png", dpi=220)
    print(f"wrote {FIGS}/frontier.{{pdf,png}}")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    figure_scaling(load("scaling"))
    figure_frontier(load("tuning"), load("phrases"))


if __name__ == "__main__":
    main()
