r"""Coherent-field reconstruction of the TM output-beam maps from an angular-spectrum sweep -- Fig. 6(b), branch C 261015.

COPY of the data part of orig_scripts\beam_maps_v6.py (md5 061770e1...): `channel_field` is VERBATIM, the grids
X_UM / Z_SUB_UM / Z_AIR_UM, the CHANNELS lists (TM passes) and the per-order incoherent summation of `make_panel` are
unchanged.  Plotting / arrows / labels of the original are NOT reproduced here (the deck draws them as native shapes);
the picture is made by render_fig6b_li.py.  The only change is the data SOURCE: a sweep json made on the Li basis.

usage:  python -I -B beam_maps_v6_li.py control    rebuild the LAURENT maps from beam_sweep_raw_v6.json and compare them
                                                    with the stored TM_pass{1,2}_channels_v6.npz (validates this copy)
        python -I -B beam_maps_v6_li.py li         build the Li maps from data_li\beam_sweep_raw_li_TM_pass{1,2}.json
Read-only on the carrier_opmap folder; writes into this folder.  Pinned to cores 14-15 after imports.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_DATA = r"<WORK_ROOT>\carrier_opmap_260929\data"

X_UM = np.linspace(-260.0, 260.0, 901)
Z_SUB_UM = np.linspace(-260.0, 0.0, 451)
Z_AIR_UM = np.linspace(0.0, 260.0, 451)

CHANNELS = {   # (side, m) of the four output orders per TM pass -- verbatim from beam_maps_v6.py
    "TM_pass1": [("sub", -2), ("sub", -1), ("air", -1), ("sub", 0)],
    "TM_pass2": [("sub", 0), ("sub", 1), ("air", 1), ("sub", 2)],
}


def channel_field(SWEEP, pass_name, side, m):
    """Verbatim from beam_maps_v6.py (flux-correct amplitude reconstruction); SWEEP passed in instead of a global."""
    META = SWEEP["meta"]
    W0_UM = META["w0_um"]
    K0 = 2.0 * np.pi / META["lam_um"]
    N_SUB = META["n_sub"]
    G = META["G_per_um"]
    p = SWEEP["passes"][pass_name]
    samples = p["samples"]
    dkx = np.array([s["dkx"] for s in samples])
    ddkx = dkx[1] - dkx[0]
    raw_w = np.exp(-dkx**2 * W0_UM**2 / 4.0)
    e0 = np.sum(raw_w) * ddkx
    weights = raw_w * ddkx / e0
    kx0 = p["kx0"]
    kx_i_arr = kx0 + dkx
    kz0_sub = np.sqrt(np.maximum((N_SUB * K0)**2 - kx_i_arr**2, 0.0))

    if side == "incident":
        Xrow = weights[:, None] * np.exp(1j * np.outer(kx_i_arr, X_UM))
        Zrow = np.exp(1j * np.outer(kz0_sub, Z_SUB_UM))
        return (Zrow.T @ Xrow), Z_SUB_UM
    kx_m = kx_i_arr + m * G
    if side == "sub":
        kz2 = (N_SUB * K0)**2 - kx_m**2
        Rm = np.array([s["R"][str(m)] for s in samples])
        arg = np.array([(s["argR"][str(m)] or 0.0) for s in samples])
        zdom = Z_SUB_UM
        zsign = -1.0
        power = Rm
        n_med = N_SUB
    else:
        kz2 = K0**2 - kx_m**2
        Tm = np.array([s["T"][str(m)] for s in samples])
        arg = np.array([(s["argT"][str(m)] or 0.0) for s in samples])
        zdom = Z_AIR_UM
        zsign = +1.0
        power = Tm
        n_med = 1.0
    prop = kz2 > 0
    if not np.any(prop) or np.max(power) <= 1e-9:
        return np.zeros((zdom.size, X_UM.size), dtype=complex), zdom
    kz = np.sqrt(np.maximum(kz2, 0.0))
    cos_ratio = np.where(prop, kz0_sub * n_med / np.maximum(kz, 1e-30) / N_SUB, 0.0)
    amp = np.where(prop, np.sqrt(np.maximum(power, 0.0) * cos_ratio), 0.0) * weights
    Xrow = (amp * np.exp(1j * np.radians(arg)))[:, None] \
        * np.exp(1j * np.outer(kx_m, X_UM))
    Zrow = np.exp(1j * zsign * np.outer(kz, zdom))
    return (Zrow.T @ Xrow), zdom


def build_channels(SWEEP, pass_name):
    """Same arrays as make_panel's chan_I (float32): incident, sub_m*, air_m*, total_sub, total_air."""
    E_inc, _ = channel_field(SWEEP, pass_name, "incident", None)
    I_inc = np.abs(E_inc)**2
    combined_sub = I_inc.copy()
    combined_air = np.zeros((Z_AIR_UM.size, X_UM.size))
    chan_I = {"incident": I_inc.astype(np.float32)}
    for side, m in CHANNELS[pass_name]:
        E, _ = channel_field(SWEEP, pass_name, side, m)
        I = np.abs(E)**2
        chan_I["%s_m%+d" % (side, m)] = I.astype(np.float32)
        if side == "sub":
            combined_sub += I
        else:
            combined_air += I
    chan_I["total_sub"] = combined_sub.astype(np.float32)
    chan_I["total_air"] = combined_air.astype(np.float32)
    return chan_I


def save_npz(path, chan_I, note):
    np.savez_compressed(path, X_um=X_UM.astype(np.float32), Z_sub_um=Z_SUB_UM.astype(np.float32),
                        Z_air_um=Z_AIR_UM.astype(np.float32),
                        units="power density (Poynting magnitude), incident beam peak = 1",
                        normalisation=("per-order INCOHERENT sum of flux-correct amplitudes (as beam_maps_v6.py); "
                                       + note), **chan_I)


def main(mode):
    if mode == "control":
        with open(os.path.join(ORIG_DATA, "beam_sweep_raw_v6.json"), encoding="utf-8") as f:
            SW = json.load(f)
        rows = []
        for pn in ("TM_pass1", "TM_pass2"):
            mine = build_channels(SW, pn)
            ref = np.load(os.path.join(ORIG_DATA, "%s_channels_v6.npz" % pn), allow_pickle=True)
            worst = 0.0
            for k, v in mine.items():
                d = float(np.abs(v.astype(float) - ref[k].astype(float)).max())
                worst = max(worst, d)
                print("control %s %-10s max|mine - stored| = %.3e  (stored max %.4f)" % (pn, k, d, float(ref[k].max())), flush=True)
            rows.append((pn, worst))
            save_npz(os.path.join(HERE, "data_li", "%s_channels_v6_rebuilt_control.npz" % pn), mine, "CONTROL rebuild of the Laurent v6 maps")
        print("CONTROL worst over all arrays:", rows, flush=True)
    else:
        for pn in ("TM_pass1", "TM_pass2"):
            with open(os.path.join(HERE, "data_li", "beam_sweep_raw_li_%s.json" % pn), encoding="utf-8") as f:
                SW = json.load(f)
            assert SW["meta"]["rcwa_rule"] == "liy" and SW["meta"]["rcwa_ff"] == 1 and SW["meta"]["ng_sweep"] == 257
            chan = build_channels(SW, pn)
            out = os.path.join(HERE, "data_li", "%s_channels_li.npz" % pn)
            save_npz(out, chan, "Li basis: rcwa_rule liy, rcwa_ff 1, nG 257, termination 0.16:172 (n1708)")
            print("wrote", out, {k: float(v.max()) for k, v in chan.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "li")
