# -*- coding: utf-8 -*-
"""Figure 4(f) retro-reflection beam map of the ADOPTED retroreflective grating (Li, nG 257, 121 angles) for a Gaussian beam of
1/e^2 intensity radius w0 = 100 um, rebuilt with the construction of upper_grating_beams.py fig_retro() (angular-spectrum sum of
the incident beam and the reflected orders m = 0, -1, -2 with their complex coefficients), from the stored angular file
<DATA_ROOT>\Result\261012_scdc_liseam\beam\new_liy_n121\angular_new_liy_n121_w100.dat (no new RCWA).
Frame = the picture of slide 4 'Image 1': x -200 .. 1000 um, z -592.39 .. 0 um; aspect 127.6 x 78.2 pt -> 1632 x 1000 px.
Plotted: (I_incident + I_reflected) normalised to the peak, white -> #D55E00 (as the existing panels; incoherent sum, so the
counter-propagating beams show no standing-wave fringes, the same choice as in2d_beams.py).
Check: the reflected intensity on the bottom face is compared with profile_new_liy_n121_w100.dat (same run)."""
import io
import os

import numpy as np
from PIL import Image

SRC = r"<DATA_ROOT>\Result\261012_scdc_liseam\beam\new_liy_n121"
OUT = r"<PAPER_ROOT>\Figure\261012_Figure_4_beams_w100"
LAM, N, T, PERIOD = 1.55, 1.4440, 592.39, 1.31539
G = LAM / PERIOD                                  # grating vector in units of k0
X0, X1, Z0, Z1, NX, NZ = -200.0, 1000.0, -T, 0.0, 1632, 1000

a = np.loadtxt(os.path.join(SRC, "angular_new_liy_n121_w100.dat"), comments="#")
kx, A = a[:, 0], a[:, 2].astype(complex)
r = {-2: a[:, 3] * np.exp(1j * np.radians(a[:, 4])), -1: a[:, 5] * np.exp(1j * np.radians(a[:, 6])),
     0: a[:, 7] * np.exp(1j * np.radians(a[:, 8]))}
k0 = 2 * np.pi / LAM
kxc = kx[len(kx) // 2]
s = T * np.tan(np.arcsin(kxc / N))                # walk to the grating: beam centre at (s, 0)
A = A * np.exp(-1j * (kx - kxc) * k0 * s)


def kz(v):
    return np.sqrt(np.maximum(N * N - v * v, 0.0))


xs = np.linspace(X0, X1, NX)
zs = np.linspace(Z0, Z1, NZ)
ein = np.zeros((NZ, NX), complex)
eout = np.zeros((NZ, NX), complex)
for i in range(len(kx)):
    ex = np.exp(1j * k0 * kx[i] * xs)[None, :]
    ein += A[i] * ex * np.exp(1j * k0 * kz(kx[i]) * zs)[:, None]
    for m in (0, -1, -2):
        ko = kx[i] + m * G
        eout += A[i] * r[m][i] * np.exp(1j * k0 * ko * xs)[None, :] * np.exp(-1j * k0 * kz(ko) * zs)[:, None]
Ii, Io = np.abs(ein) ** 2, np.abs(eout) ** 2
ref = Ii.max()

# check against the stored bottom-face profile of the same run (columns: x, I_incident, I_outgoing, normalised to the incident max)
prof = np.loadtxt(os.path.join(SRC, "profile_new_liy_n121_w100.dat"), comments="#")
pi = np.interp(prof[:, 0], xs, Ii[0] / ref)
po = np.interp(prof[:, 0], xs, Io[0] / ref)
sel = (prof[:, 0] >= X0) & (prof[:, 0] <= X1)
print("bottom-face check vs stored profile: max |dI_inc| %.2e, max |dI_out| %.2e (in window, %d samples)"
      % (np.abs(pi - prof[:, 1])[sel].max(), np.abs(po - prof[:, 2])[sel].max(), sel.sum()))
m = np.abs(xs) < 300
cx_i = (xs[m] * Ii[0][m]).sum() / Ii[0][m].sum()
cx_o = (xs[m] * Io[0][m]).sum() / Io[0][m].sum()
print("main-lobe centroid on the bottom face: incident %.4f um, reflected %.4f um, shift %.4f um" % (cx_i, cx_o, cx_o - cx_i))
print("beam centre at the grating x = s = %.2f um; reflected / incident power on the bottom face %.6f" % (s, Io[0].sum() / Ii[0].sum()))

I = (Ii + Io) / (Ii + Io).max()
c = np.array([0xD5, 0x5E, 0x00]) / 255.0
rgb = 1.0 - I[..., None] * (1.0 - c[None, None, :])
img = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)[::-1]          # z up
os.makedirs(OUT, exist_ok=True)
Image.fromarray(img).save(os.path.join(OUT, "beam_retro_TM_w100.png"))
np.savez_compressed(os.path.join(OUT, "beam_retro_TM_w100_data.npz"), x_um=xs, z_um=zs, I_incident=(Ii / ref).astype(np.float32),
                    I_reflected=(Io / ref).astype(np.float32), I_plot=I.astype(np.float32))
io.open(os.path.join(OUT, "README_beam_retro_w100.md"), "w", encoding="utf-8").write(__doc__ + "\nFiles: beam_retro_TM_w100.png (1632 x 1000, "
    "extent x -200..1000 um, z -592.39..0 um, z up), beam_retro_TM_w100_data.npz (the arrays). Label: R = 99.995 %, Delta x_c = -15 nm "
    "(main-lobe shift at w0 = 100 um, facts C9b).\n")
print("saved", os.path.join(OUT, "beam_retro_TM_w100.png"))
