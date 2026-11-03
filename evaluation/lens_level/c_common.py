# -*- coding: utf-8 -*-
r"""c_common.py -- branch C guard (overnight_261015).  Import this FIRST in every branch-C script, before numpy / any evaluator.

* BLAS / OMP threads = 1 by default (env C_BLAS_THREADS overrides), set before numpy is imported.  Reason (measured 261015, first C1 run): the exec'd evaluator heads (purity646 line 111
  REF9, lensband_eval controls) assert BIT-EXACT equality with values stored from 1-thread runs; with 3 OpenBLAS threads the overlap came out 0.7526630793457789 instead of ...788 and the
  assertion fired.  1 thread on the 3 pinned cores never oversubscribes them.
* Watchdog thread (opt-in, only when env SIM_MAX_RSS_GB is set): if the working set exceeds that many GB the process is ended (exit code 9) and a line is appended to progress.md.
* progress(msg): append a time-stamped line to progress.md (first line is written by hand as 'STATUS: IN PROGRESS').
"""
import os
import sys
import threading
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = os.environ.get("C_BLAS_THREADS", "1")
import psutil  # noqa: E402

BASE = r"<WORK_ROOT>\overnight_261015\fix_261015\C_lens_analyses"      # fix round: progress.md of the fix folder
PROGRESS = os.path.join(BASE, "progress.md")
LIMIT_GB = float(os.environ.get("SIM_MAX_RSS_GB", "0"))


THREADS = os.environ["OPENBLAS_NUM_THREADS"]                                       # value numpy starts with (later os.environ edits by imported modules do not change the live BLAS)
import numpy  # noqa: E402,F401  (numpy loaded AFTER the env is set)


def progress(msg):
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def _watch():
    p = psutil.Process()
    while True:
        try:
            rss = p.memory_info().rss / 1e9
            if LIMIT_GB > 0 and rss > LIMIT_GB:
                progress("WATCHDOG: working set %.2f GB > %.1f GB -- process ended (exit 9)" % (rss, LIMIT_GB))
                os._exit(9)
        except Exception:                                                            # noqa: BLE001
            pass
        time.sleep(3.0)


if LIMIT_GB > 0:
    threading.Thread(target=_watch, daemon=True).start()


def affinity():
    return sorted(psutil.Process().cpu_affinity())
