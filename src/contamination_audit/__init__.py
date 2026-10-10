"""contamination_audit -- label-free measurement of decontamination filter error.

Reimplements published decontamination filters and runs them against benchmarks
that postdate the corpus under test. A flag against such a benchmark is provably
a false positive, which makes filter error measurable without annotation.

Entry points are the numbered scripts in ``scripts/``; see the README.
"""

__version__ = "1.0.0"
