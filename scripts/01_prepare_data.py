"""Fetch the corpus and benchmarks the paper's results are computed over.

Everything written here is derived from public HuggingFace datasets, so the files
are gitignored and regenerated rather than committed. The split between the two
benchmark groups is the paper's central design and is defined here:

  measurement -- competitions held on or before September 2025, so they could
                 plausibly be inside a corpus assembled by October 2025.
  control     -- competitions held January 2026 onward, so they provably cannot
                 be. Any flag against these is a false positive by construction.

Olmo 3 was announced 2025-11-20; we treat Dolci as closed by roughly October 2025
and leave an ambiguous band (Oct--Dec 2025) out of both groups.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging

_log = logging.getLogger("prepare_data")

RAW = Path("data/raw")

MEASUREMENT = [
    "apex_2025",
    "apex-shortlist",
    "smt_2025",
    "cmimc_2025",
    "brumo_2025",
    "aime_2025",
    "hmmt_feb_2025",
    "aime_2024_I",
    "aime_2024_II",
    "hmmt_feb_2024",
]
CONTROL = [
    "aime_2026",
    "hmmt_feb_2026",
    "usamo_2026",
    "arxivmath-0126",
    "arxivmath-0326",
    "arxivmath-0626",
    "brokenarxiv-0326",
    "brokenarxiv-0626",
]

# A control set is only valid against a corpus it postdates, so it is defined per
# corpus. Nemotron-SFT-Math-v4 was released 2026-05-01, which puts AIME 2026 and
# the Jan-Mar arXiv sets inside its window; only June 2026 onward can serve.
CONTROL_POST_MAY_2026 = [
    "arxivmath-0626",
    "arxivmath-0826",
    "brokenarxiv-0626",
    "brokenarxiv-0826",
    "arxivlean-0626",
]


def pull_olymmath() -> None:
    from datasets import load_dataset

    for cfg in ["en-hard", "en-easy"]:
        tag = cfg.replace("-", "_")
        out = RAW / f"olymmath_{tag}.jsonl"
        ds = load_dataset("RUC-AIBOX/OlymMATH", cfg, split="test")
        with out.open("w", encoding="utf-8") as f:
            for i, r in enumerate(ds):
                f.write(
                    json.dumps(
                        {
                            "id": f"olym_{tag}_{i:04d}",
                            "problem": r.get("problem") or r.get("question", ""),
                            "answer": str(r.get("answer", "")),
                        }
                    )
                    + "\n"
                )
        _log.info("OlymMATH %s: %d items -> %s", cfg, len(ds), out)


def pull_matharena(names: list[str], group: str, out: Path) -> None:
    from datasets import load_dataset

    rows, failed = [], []
    for name in names:
        try:
            ds = load_dataset(f"MathArena/{name}", split="train")
        except Exception as e:  # noqa: BLE001  -- any fetch failure is reported below
            failed.append((name, str(e)[:60]))
            continue
        if "problem" not in ds.column_names:
            failed.append((name, f"no problem field: {ds.column_names}"))
            continue
        for i, r in enumerate(ds):
            rows.append(
                {
                    "id": f"{name}_{i:04d}",
                    "benchmark": name,
                    "group": group,
                    "problem": r["problem"],
                    "answer": str(r.get("answer", "")),
                }
            )
        _log.info("  %-22s %4d", name, len(ds))
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _log.info("%s: %d items -> %s", group, len(rows), out)

    # Every number in the paper is a fraction of one of these groups, so a group
    # that is quietly short produces results that look fine and are not the
    # paper's. Stop instead.
    if failed:
        for name, err in failed:
            _log.error("  FAILED %s: %s", name, err)
        raise SystemExit(
            f"{len(failed)} of {len(names)} {group} sets did not "
            f"download; the group would be incomplete"
        )


def _first_user(messages) -> str:
    """The first user turn, across the field names these datasets use.

    Content is sometimes a plain string and sometimes a list of typed parts;
    both appear in the corpora here.
    """
    for message in messages or []:
        if (message.get("role") or message.get("from")) in ("user", "human"):
            content = message.get("content") or message.get("value") or ""
            if isinstance(content, str):
                return content
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    return ""


def pull_corpus(
    dataset: str, out: Path, target: int, id_field: str, question, flush_every: int
) -> None:
    """Stream a corpus, keeping one question per document -- all a filter reads.

    Long streaming pulls drop their connection, so this appends and resumes from
    what is already on disk instead of restarting. `skip` gets us back to roughly
    the right place; the id set makes the overlap harmless, which matters because
    documents with no question are consumed without being written.
    """
    from datasets import load_dataset

    seen: set[str] = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                seen.add(json.loads(line)["train_id"])
            except json.JSONDecodeError:
                _log.warning("skipping malformed line in %s", out)
        _log.info("resuming with %d rows already on disk", len(seen))

    n = len(seen)
    with out.open("a", encoding="utf-8") as f:
        for attempt in range(30):
            if n >= target:
                break
            try:
                ds = load_dataset(dataset, split="train", streaming=True)
                if n:
                    ds = ds.skip(n)
                for row in ds:
                    rid = row.get(id_field)
                    if rid in seen:
                        continue
                    q = question(row)
                    if not q.strip():
                        continue
                    seen.add(rid)
                    f.write(
                        json.dumps(
                            {"train_id": rid, "problem": q, "source": row.get("source")}
                        )
                        + "\n"
                    )
                    n += 1
                    if n % flush_every == 0:
                        f.flush()
                        _log.info("  %d rows", n)
                    if n >= target:
                        break
            except Exception as e:  # noqa: BLE001  -- any transport error retries
                _log.warning(
                    "  stream dropped (%s), retry %d", type(e).__name__, attempt + 1
                )
                time.sleep(5)

    if n < target:
        raise SystemExit(f"{dataset}: got {n} of {target} rows after 30 attempts")
    _log.info("%s: %d rows -> %s", dataset, n, out)


def pull_dolci(target: int) -> None:
    pull_corpus(
        "allenai/Dolci-Think-SFT-32B",
        RAW / "dolci_questions.jsonl",
        target,
        id_field="id",
        question=lambda row: _first_user(row.get("messages")),
        flush_every=50_000,
    )


def pull_nemotron(target: int) -> None:
    """Nemotron-SFT-Math-v4, released 2026-05-01, decontamination described in
    prose only -- a third corpus and a second prose-derived case. Its `problem`
    field is already the question."""
    pull_corpus(
        "nvidia/Nemotron-SFT-Math-v4",
        RAW / "nemotron_questions.jsonl",
        target,
        id_field="uuid",
        question=lambda row: row.get("problem") or "",
        flush_every=25_000,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-rows",
        type=int,
        default=390_000,
        help="the paper's largest scaling point",
    )
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument(
        "--nemotron",
        action="store_true",
        help="also fetch the Nemotron corpus and its later control set",
    )
    parser.add_argument("--skip-nemotron-corpus", action="store_true")
    args = parser.parse_args()
    configure_logging()
    RAW.mkdir(parents=True, exist_ok=True)

    pull_olymmath()
    _log.info("measurement set (competitions <= Sept 2025):")
    pull_matharena(MEASUREMENT, "measure", RAW / "bench_measure.jsonl")
    _log.info("control set (competitions Jan 2026 onward):")
    pull_matharena(CONTROL, "control", RAW / "bench_control.jsonl")

    if args.nemotron:
        _log.info("control set valid against a May 2026 corpus (June 2026 onward):")
        pull_matharena(
            CONTROL_POST_MAY_2026,
            "control_post_may26",
            RAW / "bench_control_post_may26.jsonl",
        )

    if not args.skip_corpus:
        pull_dolci(args.corpus_rows)
    if args.nemotron and not args.skip_nemotron_corpus:
        pull_nemotron(100_000)


if __name__ == "__main__":
    main()
