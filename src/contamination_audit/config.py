"""Logging and seeding for the pipeline entry points.

The filters themselves are deterministic and take no configuration; the only
randomness anywhere is the bootstrap in the results stage, which seeds its own
generator explicitly. `seed_everything` is kept for callers that add sampling.
"""

from __future__ import annotations

import logging
import random


def seed_everything(seed: int) -> None:
    """Seed Python's RNG, and NumPy's if NumPy is present."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def configure_logging(level: int = logging.INFO) -> None:
    """Standard logger config used by every script entry point."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
