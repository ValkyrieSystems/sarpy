"""Utilities for measuring performance"""

__classification__ = "UNCLASSIFIED"

import contextlib
import logging
import time
import tracemalloc

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
    mem_start, peak_start = tracemalloc.get_traced_memory()
    start = time.perf_counter()
    try:
        yield start
    finally:
        stop = time.perf_counter() - start
        indent = "  " * (len(_PATH) - 1)
        log_str = f'howlong: {indent}{label}: {stop:.3f}s'

        if tracemalloc.is_tracing():
            mem_stop, peak_stop = tracemalloc.get_traced_memory()
            mem_diff = mem_stop - mem_start
            log_str += f' | end: {mem_stop/1024/1024:.1f}MiB ({mem_diff/1024/1024:+.1f}MiB)  peak: {peak_stop/1024/1024:.1f}MiB'
            if peak_stop > peak_start:
                log_str += '*'

        logging.info(log_str)
        _PATH.pop()
