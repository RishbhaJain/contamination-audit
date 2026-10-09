"""Check every headline number in the paper against the JSON that produced it.

A transcription error between a run and the manuscript is invisible to the
compiler and to every reviewer who does not re-run the pipeline. This reads
results/paper/*.json, derives the figures the paper states, and reports any that
are missing from or inconsistent with paper/arr_decontamination.tex.

It is not a substitute for regenerating from a clean checkout, which also catches
a stale JSON. It catches the step after that one.

    python scripts/22_verify_paper.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER = Path("paper/arr_decontamination.tex")
OUT = Path("results/paper")


def load(name: str):
    p = OUT / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def norm(tex: str) -> str:
    """Strip LaTeX number formatting so '100{,}000' matches '100000'."""
    t = tex.replace("{,}", "").replace("\\%", "%").replace("~", " ")
    return re.sub(r"\s+", " ", t)


def main() -> int:
    if not PAPER.exists():
        print(f"missing {PAPER}")
        return 1
    tex = norm(PAPER.read_text(encoding="utf-8"))

    checks: list[tuple[str, str, bool]] = []

    def expect(label: str, needle: str):
        checks.append((label, needle, norm(needle) in tex))

    # ---- Table 2: per-filter counts on both benchmark groups
    main_j = load("main")
    if main_j:
        nm, nc = main_j["n_measure"], main_j["n_control"]
        expect("Table 2 measurement size", str(nm))
        expect("Table 2 control size", str(nc))
        for name, v in main_j["filters"].items():
            m, c = len(v["measure"]), len(v["control"])
            expect(f"{name} measurement count", f"{100*m/nm:.1f}%")
            expect(f"{name} control count", f"{100*c/nc:.1f}%")

    # ---- Consensus and the bootstrap intervals on the headline rates
    con = load("consensus")
    if con:
        expect("items flagged by >=1 filter", str(con["flagged_any"]))
        one = con["consensus"].get("1")
        if one:
            expect("flagged by exactly one", f"{one['pct']:.1f}%")
            expect("exactly-one CI", f"[{one['ci'][0]:.1f}, {one['ci'][1]:.1f}]")
        for name, r in con.get("rates", {}).items():
            cc = r["control"]
            if cc["pct"] == 0:
                continue   # the paper does not print a [0.0, 0.0] interval
            expect(f"{name} FPR CI", f"[{cc['ci'][0]:.1f}, {cc['ci'][1]:.1f}]")

    # ---- Scaling: the endpoints and the error-fraction series
    sc = load("scaling")
    if sc:
        a, b = sc[0], sc[-1]
        expect("scaling first corpus size", f"{a['corpus']}")
        expect("scaling last corpus size", f"{b['corpus']}")
        for k in ("s1_frac", "ot3_frac"):
            if a.get(k) and b.get(k):
                expect(f"{k} first", f"{a[k]:.2f}")
                expect(f"{k} last", f"{b[k]:.2f}")

    # ---- Mechanism
    mech = load("mechanism")
    if mech:
        expect("distinct FP n-grams", str(mech["distinct_ngrams"]))
        expect("false positives traced", str(mech["false_positives"]))

    # ---- Matched same-competition pairs
    mt = load("matched")
    if mt:
        for nm_, r in mt.items():
            expect(f"matched {nm_} s1", f"{r['s1']:.1f}%")

    # ---- Third corpus
    nemo = load("nemotron")
    if nemo:
        expect("nemotron measurement size", str(nemo["n_measure"]))
        expect("nemotron control size", str(nemo["n_control"]))
        for name, r in nemo["filters"].items():
            if r["fp"] or r["detect"]:
                expect(f"nemotron {name} FPR", f"{r['fp']:.1f}%")

    # ---- decon
    dec = load("decon")
    if dec:
        sweep = dec.get("threshold_sweep", {})
        if "0.2" in sweep:
            expect("decon at threshold 0.2", str(sweep["0.2"]))

    missing = [(l, n) for l, n, ok in checks if not ok]
    print(f"{len(checks) - len(missing)}/{len(checks)} figures found verbatim in the paper\n")
    if missing:
        print("NOT FOUND (stale, reworded, or genuinely wrong):")
        for label, needle in missing:
            print(f"  {label:38s} expected: {needle}")
        print("\nA miss is not automatically an error: the paper may phrase a number")
        print("differently, or omit it deliberately. Each one needs a human look.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
