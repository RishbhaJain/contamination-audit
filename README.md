# Decontaminated by Whose Standard?

Code and data for the paper *Decontaminated by Whose Standard? Label-Free
Measurement of Decontamination Filter Error*.

The paper runs several published benchmark-decontamination filters over the same
training corpora and benchmarks, and measures how far they disagree. It also
introduces a label-free way to measure a filter's false-positive rate: a
benchmark published *after* a corpus was assembled cannot be in it, so any flag
against such an item is an error that needs no annotation to identify.

Everything here runs on CPU. There is no model inference, no GPU requirement and
no API key anywhere in the pipeline.

## Layout

| Path | What it holds |
|---|---|
| `src/contamination_audit/filters.py` | the filter implementations, one per published project |
| `scripts/01_prepare_data.py` | fetches corpora and benchmark groups from HuggingFace |
| `scripts/02_results.py` | regenerates every number the paper reports |
| `scripts/03_run_decon.py` | wrapper for Ai2's `decon` (a separate Rust binary) |
| `scripts/04_verify_numbers.py` | checks the paper's text against the regenerated JSON |
| `scripts/05_figures.py` | redraws Figures 1 and 2 from the regenerated JSON |
| `data/raw/bench_*.jsonl` | the benchmark groups (committed; small) |
| `results/paper/*.json` | regenerated results, one file per paper section |
| `tests/` | unit tests for the filters |

## The two benchmark groups

This split is the paper's method, and it is defined in `scripts/01_prepare_data.py`.

- **Measurement**: benchmarks published *before* the corpus was assembled. A flag
  here may be correct or incorrect.
- **Control**: benchmarks published *after*. These cannot be in the corpus, so
  every flag against them is a false positive by construction.

The split is a function of the corpus, not a fixed list. Each corpus in the paper
gets its own, derived from its release date: a benchmark that is a control for one
corpus can be a measurement item for a later one.

## Reproducing

```bash
pip install -e .
make data        # streams the corpora; slow, and resumable if a connection drops
make results     # regenerates results/paper/*.json and prints each figure
make figures
make verify      # confirms the paper's numbers match the regenerated JSON
make test
```

`make results` prints each number in the form the paper states it, so the two can
be diffed directly. Individual sections:

```bash
python scripts/02_results.py --stage main        # the main comparison table
python scripts/02_results.py --stage validate    # tests the control assumption
python scripts/02_results.py --stage scaling     # Figure 1 (slowest, ~25 min)
python scripts/02_results.py --stage tuning      # Figure 2
python scripts/02_results.py --stage phrases     # Table 5, phrase agreement
python scripts/02_results.py --stage matched     # matched competition pairs
python scripts/02_results.py --stage nemotron    # third corpus
```

`--help` lists all eleven stages. Without `make`, the targets above are just the
five numbered scripts, run in order.

### decon

Ai2's `decon` is deliberately **not** reimplemented here. Build and run the real one:

```bash
git clone https://github.com/allenai/decon && cd decon && cargo build --release
cd - && make decon DECON_BIN=decon/target/release/decon
```

This matters. An earlier attempt reproduced only its candidate-generation stage —
any shared 5-gram over `cl100k` tokens, without the cluster expansion and weighted
scoring — and that version flagged 199 of 200 items where the real binary flags 0.
A two-stage filter cannot be approximated by its first stage. The registry marks
partial implementations `faithful=False` and excludes them from comparisons.

## Notes on reproduction

- **Streaming is flaky.** `scripts/01_prepare_data.py` resumes by skipping rows
  already on disk; rerun it if a pull drops.
- **Corpus order matters.** Every scaling point takes the first *N* rows of the
  corpus file, so a partial pull shifts every number. Complete the pull before
  comparing against the paper.
- **Runtimes are single-threaded Python.** Model tokenizers dominate; the largest
  scaling point takes roughly 25 minutes.
- **Determinism.** Bootstrapping is seeded, the filters are deterministic, and no
  LLM is called anywhere. Where a stage reports an example n-gram it ties-breaks
  lexicographically, because set iteration order varies with `PYTHONHASHSEED`.
- **Figures read the JSON.** Both figures are drawn from `results/paper/*.json`
  rather than transcribed constants, so they cannot drift from the tables.

## Licence

MIT, see `LICENSE`.
