# RSoft process management utilities and config loader
# Shared by Main_adam.py and Main_nlopt.py

# Explicitly export underscore-prefixed names so `from core.rsoft_utils import *` works
__all__ = [
    '_rsoft_stagger',
    '_mpiexec_lock_acquire',
    '_mpiexec_lock_release',
    '_mpiexec_wait_workers_and_release',
    '_check_rsoft_success_fullwave',
    '_parse_config',
    '_get',
    '_HAS_PSUTIL',
    '_psutil',
    'save_schedule_state',
    'load_schedule_state',
]

import ast
import os
import tempfile
import time

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except Exception:
    _psutil = None
    _HAS_PSUTIL = False

# -- RSoft start-interval guarantee (cross-process, race-free) -----------------------
_RSOFT_LOCK_FILE     = os.path.join(tempfile.gettempdir(), "rsoft_launch.lock")
_RSOFT_MUTEX_FILE    = os.path.join(tempfile.gettempdir(), "rsoft_launch.mutex")
_RSOFT_START_GAP     = 3.0  # s -- wait for mpiexec initialisation to finish (serialisation lock)
_RSOFT_MUTEX_STALE   = 30  # s -- mutex file stale criterion (crash recovery)

def _rsoft_stagger():
    deadline = time.time() + 60
    while True:
        try:
            fd = os.open(_RSOFT_MUTEX_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(_RSOFT_MUTEX_FILE) > _RSOFT_MUTEX_STALE:
                    os.remove(_RSOFT_MUTEX_FILE)
                    continue
            except Exception:
                pass
            if time.time() > deadline:
                break
            time.sleep(0.05)
        except Exception:
            break
    try:
        now = time.time()
        last = 0.0
        if os.path.exists(_RSOFT_LOCK_FILE):
            try:
                with open(_RSOFT_LOCK_FILE, 'r', encoding="utf-8") as _f:
                    last = float(_f.read().strip() or 0)
            except (ValueError, OSError):
                # If the lock file is corrupted (partial write due to contention, etc.), skip the stagger and
                # proceed -- it is overwritten with a proper value immediately below, which self-heals.
                last = 0.0
        elapsed = now - last
        if elapsed < _RSOFT_START_GAP:
            time.sleep(_RSOFT_START_GAP - elapsed)
        # Write to a temporary file and swap with os.replace -- prevents torn writes between concurrent writers
        # (opening directly with 'w' sometimes corrupted the file with NULs when several processes overwrote it
        #  on an SMB shared drive; confirmed 2026-07-07)
        #
        # Advisory timestamp update -- the ENTIRE block below MUST be non-fatal.
        # The tmp-file open() itself can raise FileNotFoundError (WinError 2) when
        # the SMB share parent dir is momentarily unreachable under heavy concurrent
        # launches (2026-07-17: this unhandled raise killed the MGC 3D extrudeseed
        # job at iter 100, and orphan cleanup then deleted its checkpoint). os.replace
        # can likewise raise PermissionError (WinError 5) when another process holds a
        # transient handle on the destination (2026-07-09: killed 4 NA=0.9 jobs).
        # A missed 3s stagger only weakens anti-collision -- it must NEVER crash the run.
        try:
            _tmp = f"{_RSOFT_LOCK_FILE}.{os.getpid()}.tmp"
            with open(_tmp, 'w', encoding="utf-8") as _f:
                _f.write(str(time.time()))
            _replaced = False
            for _att in range(5):
                try:
                    os.replace(_tmp, _RSOFT_LOCK_FILE)
                    _replaced = True
                    break
                except (PermissionError, OSError):
                    time.sleep(0.2)
            if not _replaced:
                try:
                    os.remove(_tmp)
                except OSError:
                    pass
        except OSError:
            # SMB share momentarily unreachable (parent dir lost, etc.) -- skip the stagger silently.
            pass
    finally:
        try:
            os.remove(_RSOFT_MUTEX_FILE)
        except Exception:
            pass

# -- mpiexec initialisation serialisation lock -----------------------------------------
_MPIEXEC_INIT_LOCK  = os.path.join(tempfile.gettempdir(), "mpiexec_init.lock")
_MPIEXEC_INIT_STALE = 60   # s -- stale timeout for crash recovery

def _mpiexec_lock_acquire():
    """Exclusive mpiexec initialization lock acquisition. Hold until _mpiexec_lock_release()."""
    deadline = time.time() + 86400
    while True:
        try:
            fd = os.open(_MPIEXEC_INIT_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(_MPIEXEC_INIT_LOCK) > _MPIEXEC_INIT_STALE:
                    os.remove(_MPIEXEC_INIT_LOCK)
                    continue
            except Exception:
                pass
            if time.time() > deadline:
                return  # fail open
            time.sleep(0.2)
        except Exception:
            return  # fail open

def _mpiexec_lock_release():
    """Release the mpiexec initialization lock."""
    try:
        os.remove(_MPIEXEC_INIT_LOCK)
    except Exception:
        pass

def _mpiexec_wait_workers_and_release(proc, mpi_np, timeout=30):
    """After detecting mpi_np fullwave MPI processes in the subtree of the fwmpirun proc, release the lock.
    If psutil is unavailable, wait a fixed 3 s (fallback).
    """
    try:
        if _HAS_PSUTIL:
            try:
                ps_proc = _psutil.Process(proc.pid)
            except Exception:
                return
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    workers = sum(
                        1 for c in ps_proc.children(recursive=True)
                        if "fullwave" in c.name().lower()
                    )
                    if workers >= mpi_np:
                        break
                except _psutil.NoSuchProcess:
                    break  # proc exited (mpiexec crash)
                time.sleep(0.1)
        else:
            time.sleep(3.0)  # fixed wait when psutil is absent
    finally:
        _mpiexec_lock_release()


def _check_rsoft_success_fullwave(cwd, prev_size=0, n_expected=1):
    """fullwave/fwmpirun zombie check: log.txt must have n_expected new completions."""
    logfile = os.path.join(cwd, "log.txt")
    if not os.path.exists(logfile):
        return False
    size = os.path.getsize(logfile)
    if size <= prev_size:
        return False
    with open(logfile, "rb") as _f:
        _f.seek(prev_size)
        new_content = _f.read().decode("utf-8", errors="ignore")
    return new_content.count("Actual Computation Time") >= n_expected


# ===========================================================================
# External config file loader
# ===========================================================================
def _parse_config(path):
    cfg = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                cfg[key.strip()] = val.split("#")[0].strip()
    except FileNotFoundError:
        pass
    return cfg


def _get(cfg, key, default):
    if key not in cfg:
        return default
    raw = cfg[key]
    if isinstance(default, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return ast.literal_eval(raw)
    return raw  # str


# ===========================================================================
# NLopt beta sub-epoch schedule persistence (Main_nlopt.py restart support)
# ===========================================================================
# NLopt's internal LD_CCSAQ solver state (dual variables, approximation history,
# etc.) cannot be serialized -- only the density (x0) is checkpointed today via
# Metagrating_density_file / Metalens_density_file / Metasurface_density_file.
# On a restart (start_epoch re-entry, or an in-run retry) the reloaded
# density was previously always resumed at the FIRST (warmup) schedule entry,
# silently discarding every sub-epoch already completed.
#
# save_schedule_state() is called once per completed sub-epoch (i.e. once per
# `for beta_epoch, max_eval, activate_lw in epoch_schedule:` loop iteration,
# right after solver.optimize() returns) and OVERWRITES a single small text
# file -- it is the single source of truth for "how far did we get". On the
# next resume, load_schedule_state() is consulted (same RESTART_FROM_CHECKPOINT
# condition that triggers the density warm-reload): if the file exists and its
# recorded epoch matches the current cluster_n, the schedule loop skips every
# sub-epoch whose index is < the recorded one and resumes AT the recorded
# sub-epoch (using the reloaded density as x0 and the recorded beta). This is
# sub-epoch granularity, not iteration granularity -- the CCSAQ solver for the
# resumed sub-epoch restarts fresh from the checkpointed density; iterations
# already spent inside that specific sub-epoch before the interruption are not
# recovered, only the completed sub-epochs before it.
def save_schedule_state(path, *, epoch, schedule_index, beta, max_eval, iteration):
    """Overwrite the schedule-state file with the position just completed.

    Called after each sub-epoch (each `epoch_schedule` entry) finishes, so the
    file always reflects "the last sub-epoch that fully completed". Plain
    ASCII key=value, one entry per line, encoding="utf-8" (cp949-safe).
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\n")
        f.write(f"schedule_index={schedule_index}\n")
        f.write(f"beta={beta}\n")
        f.write(f"max_eval={max_eval}\n")
        f.write(f"iteration={iteration}\n")
    os.replace(tmp_path, path)  # atomic overwrite -- avoids truncated reads if interrupted


def load_schedule_state(path, *, epoch):
    """Read back a schedule-state file written by save_schedule_state().

    Returns None if the file does not exist, is malformed, or its recorded
    `epoch` does not match the current cluster_n (stale state from an earlier
    epoch must never be applied to a different one). Otherwise returns a dict
    with keys: epoch (int), schedule_index (int), beta (float), max_eval (int),
    iteration (int).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = {}
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                raw[key.strip()] = val.strip()
        state = {
            "epoch":          int(raw["epoch"]),
            "schedule_index": int(raw["schedule_index"]),
            "beta":           float(raw["beta"]),
            "max_eval":       int(raw["max_eval"]),
            "iteration":      int(raw["iteration"]),
        }
    except (KeyError, ValueError):
        return None
    if state["epoch"] != epoch:
        return None
    return state
