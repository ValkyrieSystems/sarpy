"""Utilities for measuring performance"""

__classification__ = "UNCLASSIFIED"

import contextlib
import logging
import time

_PATH = []


@contextlib.contextmanager
def howlong(label):
    """Print how long a section of code took

    Args
    ----
    label: str
        String printed alongside duration

    """
    global _PATH
    _PATH.append(label)
    start = time.perf_counter()
    try:
        yield start
    finally:
        indent = "  " * (len(_PATH) - 1)
        logging.info(f"howlong: {indent}{label}: {time.perf_counter() - start}")
        _PATH.pop()
