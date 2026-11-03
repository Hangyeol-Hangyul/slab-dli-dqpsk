# -*- coding: utf-8 -*-
"""Temperature drift of the DLI bias phase for a ONE-SYMBOL (10 ps) delay.

(a) our fused-silica slab (recomputed from first principles, compared with the SI value)
(b) waveguide delay lines (Si wire, SiN, silica PLC, polymer) with cited constants
Labels: MEASURED (value printed in a paper), CALC (formula), ASSUMED (typical value, not verified).
Run:  python -X utf8 calc_thermal.py     (writes thermal_drift.csv and slab_recompute.csv beside this file)
"""
import csv
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))
C = 299792458.0
LAM = 1550e-9
F0 = C / LAM
K0 = 2 * math.pi / LAM
TAU = 10e-12
FSR_GHZ = 100.0
PM_PER_GHZ = (LAM ** 2 / C) * 1e12 * 1e9  # pm of wavelength per GHz at 1550 nm

# ---------------------------------------------------------------- slab (a)
T0 = 592.39e-6          # slab thickness [m]            SI Table S3
THETA = math.radians(54.69)  # internal angle           SI / task
N0 = 1.444024           # phase index, SI Eq. (S3) text
NG0 = 1.462596          # group index, SI Eq. (S3) text
LAM_PERIOD = 1315.39e-9  # SI notation, Lambda


def slab_phase_deg(dT, dndT, a_t, a_L, n0=N0):
    """Phi_s = 2 t k_z, k_z = sqrt((n k0)^2 - G^2); k0 fixed (fixed laser).
    The period of the top/bottom metagrating is defined by the angle: G = n0 k0 sin(theta)."""
    G0 = n0 * K0 * math.sin(THETA)
    t = T0 * (1 + a_t * dT)
    n = n0 + dndT * dT
    G = G0 / (1 + a_L * dT)
    kz = math.sqrt((n * K0) ** 2 - G ** 2)
    return math.degrees(2 * t * kz)


def slab_drift(dndT, alpha, h=0.5):
    """central difference of Phi_s [deg/K] split by cause"""
    def d(**kw):
        a = dict(dndT=0.0, a_t=0.0, a_L=0.0)
        a.update(kw)
        return (slab_phase_deg(+h, **a) - slab_phase_deg(-h, **a)) / (2 * h)
    thick = d(a_t=alpha)
    index = d(dndT=dndT)
    period = d(a_L=alpha)
    total = d(dndT=dndT, a_t=alpha, a_L=alpha)
    return thick, index, period, total


L_GEOM = 2 * T0 / math.cos(THETA)              # extra path of arm B [m]
TAU_SLAB = NG0 * L_GEOM / C
PHI0_DEG = slab_phase_deg(0.0, 0.0, 0.0, 0.0)
THETA_GRATING = math.degrees(math.asin(LAM / (N0 * LAM_PERIOD)))


def closed_form(dndT, alpha):
    """(360/lambda) (dn/dT + n alpha) L_geom  -- identical to the exact derivative (checked below)"""
    return 360.0 / LAM * (dndT + N0 * alpha) * L_GEOM


slab_cases = [
    # name, dn/dT, alpha, source
    ("SI value (dn/dT 1.1e-5 Malitson via Katz; alpha 5.5e-7 implied by SI 0.126 deg/K)", 1.1e-5, 5.5e-7,
     "SI Note S1; Katz arXiv:cond-mat/9712327"),
    ("Katz (dn/dT 1.1e-5; alpha 0.4e-6, AIP Handbook 1957)", 1.1e-5, 0.4e-6, "Katz arXiv:cond-mat/9712327"),
    ("Corning 7980 at 1.55 um, 293 K (dn/dT 8.16e-6; alpha 5.2e-7 = upper end of NIST 4.0-5.2e-7)", 8.16e-6, 5.2e-7,
     "Rego, Sensors 23, 6023 (2023), Table 1 (Leviton data); NIST/Egan range quoted there"),
    ("Corning 7980 with alpha 4.0e-7 (lower end)", 8.16e-6, 4.0e-7, "Rego 2023"),
    ("Homosil (highest dn/dT 8.53e-6; alpha 5.2e-7)", 8.53e-6, 5.2e-7, "Rego 2023"),
]
slab_rows = []
for name, dn, al, src in slab_cases:
    th, ix, pe, tot = slab_drift(dn, al)
    cf = closed_form(dn, al)
    slab_rows.append(dict(case=name, dn_dT=dn, alpha=al, thickness_degK=th, index_degK=ix, period_degK=pe,
                          total_degK=tot, closed_form_degK=cf, GHz_per_K=tot / 360.0 * FSR_GHZ,
                          deg_per_K_per_6deg_window_K=6.0 / tot, source=src))
SI_TOTAL = slab_rows[0]["total_degK"]
BEST = slab_rows[2]["total_degK"]

# ---------------------------------------------------------------- waveguides (b)
# platform: n_eff, n_g, dn_eff/dT, alpha_sub, label tags and sources
wg = [
    dict(platform="Si wire (SOI, oxide clad) CALC", n_eff=2.4, n_g=4.2, dn=1.8e-4, al=2.5e-6,
         tags="n_g: manuscript input (Table 1); n_eff: ASSUMED typical 450x220 nm TE; dn_eff/dT: ASSUMED = bulk Si TOC 1.8e-4 (Voigt thesis Sec. 5.4; 1.86e-4 quoted by Prasad 2017); alpha: Si 2.5e-6 (Voigt thesis Sec. 5.4)",
         src="Voigt TU Berlin thesis (Dokument_45) Sec. 5.4; Prasad, Selvaraja, Varma arXiv:1710.03605"),
    dict(platform="SiN (LPCVD, SiO2 clad, Si substrate) CALC", n_eff=1.7, n_g=2.0, dn=2.45e-5, al=2.5e-6,
         tags="n_g: manuscript input; n_eff: ASSUMED; dn_eff/dT: MEASURED material value of stoichiometric Si3N4 2.45e-5 used as upper bound for n_eff (oxide 0.95e-5 and confinement lower it); alpha: Si substrate 2.5e-6",
         src="Arbabi and Goddard, Opt. Lett. 38, 3878 (2013), doi 10.1364/OL.38.003878"),
    dict(platform="Silica PLC (Si substrate) CALC with substrate expansion", n_eff=1.45, n_g=1.47, dn=1.0e-5, al=2.5e-6,
         tags="n_g: manuscript input; n_eff: ASSUMED; dn_eff/dT: MEASURED/quoted 1e-5 per C for silica glass (Nasu 2010); alpha: Si substrate 2.5e-6",
         src="Nasu et al., IEICE Trans. Electron. E93-C, 1191 (2010), doi 10.1587/transele.E93.C.1191"),
    dict(platform="Silica PLC CALC thermo-optic term only", n_eff=1.45, n_g=1.47, dn=1.0e-5, al=0.0,
         tags="as above but expansion term set to 0 (reproduces the measured 1.4 GHz/K of Nasu 2010)",
         src="Nasu 2010"),
    dict(platform="Polymer (SU-8 class, on Si) CALC indicative", n_eff=1.5, n_g=1.5, dn=-1.21e-4, al=2.5e-6,
         tags="dn/dT: MEASURED SU-8 -1.21e-4 (Guan 2016, quoted by Xie 2023 review; wavelength not stated); n_eff and n_g: ASSUMED 1.5; alpha: ASSUMED substrate-constrained Si 2.5e-6 (free polymer expansion 1e-5 to 2.2e-4 would change magnitude and sign); sign negative",
         src="Xie, Chen, Li, Yi, Polymers 15, 3721 (2023), doi 10.3390/polym15183721"),
]
rows = []


def add_row(platform, kind, n_eff, n_g, dn, al, L, degK, ghzK, tags, src):
    rows.append(dict(platform=platform, kind=kind, n_eff=n_eff, n_g=n_g, dn_eff_dT_perK=dn, alpha_perK=al,
                     L_10ps_mm=(L * 1e3 if L is not None else ""),
                     deg_per_K=degK, GHz_per_K=ghzK, pm_per_K=abs(ghzK) * PM_PER_GHZ,
                     ratio_to_slab_SI=degK / SI_TOTAL, K_per_6deg=6.0 / abs(degK),
                     basis=tags, sources=src))


# our slab rows
for key, idx in (("SI constants", 0), ("recomputed, Corning 7980 constants", 2)):
    r = slab_rows[idx]
    add_row("Fused-silica slab (this work), " + key, "CALC", N0, NG0, r["dn_dT"], r["alpha"], L_GEOM,
            r["total_degK"], r["GHz_per_K"],
            "n, n_g, t, theta: SI Table S3; full derivative of Phi_s = 2 t kz incl. thickness, index and period",
            r["source"])

for w in wg:
    L = C * TAU / w["n_g"]
    deg = 360.0 / LAM * (w["dn"] + w["n_eff"] * w["al"]) * L
    ghz = deg / 360.0 * FSR_GHZ
    add_row(w["platform"], "CALC", w["n_eff"], w["n_g"], w["dn"], w["al"], L, deg, ghz, w["tags"], w["src"])

# literature (delay independent GHz/K -> deg/K at FSR 100 GHz = 3.6 deg per GHz)
lit = [
    ("Silica PLC DLI (FSR 21.9 GHz), temperature dependence of the transmission peak", 1.4,
     "MEASURED: 'temperature dependence of 1.4 GHz/C, due to the temperature dependent refractive index of silica'",
     "Nasu et al. 2010 (doi 10.1587/transele.E93.C.1191), Sec. 2.1"),
    ("Silica DLI, typical drift", 1.25, "MEASURED/typical: 'Silica-based DLIs typically drift by 1.25 GHz/C'",
     "Lize et al., IEEE Photon. Technol. Lett. 19(23), 1886 (2007) (doi not retrieved), Sec. V"),
    ("SOI 4-um rib DLI (Voigt 2008)", 10.0, "MEASURED/stated: 'intrinsically high tuning efficiency (~10 GHz/K)'",
     "Voigt et al., IEEE Photon. Technol. Lett. 20(8), 614 (2008) (doi not retrieved)"),
    ("SOI 4-um MZ-DI (thesis)", 80.0 / PM_PER_GHZ, "MEASURED: 'wavelength shift is about 10 GHz/K (80 pm/K)'",
     "Voigt, TU Berlin thesis (Dokument_45), Sec. 7"),
    ("Si nanowire air-clad microring (387 x 220 nm)", 62.72 / PM_PER_GHZ,
     "MEASURED: dlambda/dT = 62.72 pm/C (mean of 3); n_g 4.64 simulated",
     "Prasad, Selvaraja, Varma, arXiv:1710.03605"),
    ("Silica PLC DLI after resin-groove athermalization", 8.8 / 80.0,
     "MEASURED residual 8.8 GHz over -10..70 C = 80 K; mean slope CALC (second order, not linear)",
     "Nasu et al. 2010, Sec. 2.1"),
    ("Optoplex passive athermal DPSK DLI", 0.02,
     "MEASURED (vendor): 'measured frequency drift over temperature is only ~0.02 GHz/C'; seen in a web-search snippet, page text not confirmed",
     "Optoplex 10 Gbps DPSK DLI data sheet"),
]
for name, ghz, tags, src in lit:
    deg = ghz / FSR_GHZ * 360.0
    add_row(name, "LITERATURE", "", "", "", "", None, deg, ghz, tags, src)

# InP: no verified constant -> explicit empty row
rows.append(dict(platform="InP waveguide", kind="NOT COMPUTED", n_eff="", n_g="", dn_eff_dT_perK="", alpha_perK="",
                 L_10ps_mm="", deg_per_K="", GHz_per_K="", pm_per_K="", ratio_to_slab_SI="", K_per_6deg="",
                 basis="no citable measured dn_eff/dT of an InP waveguide was retrieved; Melati 2016 (doi 10.1063/1.4970937) only states that the group index drifts about twice as fast as the phase index",
                 sources="Melati, Waqas, Alippi, Melloni, J. Appl. Phys. 120, 213102 (2016)"))

fields = ["platform", "kind", "n_eff", "n_g", "dn_eff_dT_perK", "alpha_perK", "L_10ps_mm", "deg_per_K",
          "GHz_per_K", "pm_per_K", "ratio_to_slab_SI", "K_per_6deg", "basis", "sources"]
with open(os.path.join(OUT, "thermal_drift.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    for r in rows:
        rr = dict(r)
        for k in ("n_eff", "n_g", "dn_eff_dT_perK", "alpha_perK", "L_10ps_mm", "deg_per_K", "GHz_per_K",
                  "pm_per_K", "ratio_to_slab_SI", "K_per_6deg"):
            if isinstance(rr[k], float):
                rr[k] = "%.6g" % rr[k]
        wr.writerow(rr)

sf = ["case", "dn_dT", "alpha", "thickness_degK", "index_degK", "period_degK", "total_degK", "closed_form_degK",
      "GHz_per_K", "deg_per_K_per_6deg_window_K", "source"]
with open(os.path.join(OUT, "slab_recompute.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    wr = csv.DictWriter(fh, fieldnames=sf)
    wr.writeheader()
    for r in slab_rows:
        wr.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})

# ---------------------------------------------------------------- console summary
print("SLAB geometry: L_geom = 2t/cos(theta) = %.2f um ; tau = n_g L/c = %.4f ps ; Phi_s = %.1f deg" %
      (L_GEOM * 1e6, TAU_SLAB * 1e12, PHI0_DEG))
print("theta from grating eq. with Lambda=1315.39 nm: %.4f deg (task value 54.69)" % THETA_GRATING)
for r in slab_rows:
    print("%-95s thick %.4f idx %.4f per %.4f total %.4f (closed %.4f) GHz/K %.3f" %
          (r["case"][:95], r["thickness_degK"], r["index_degK"], r["period_degK"], r["total_degK"],
           r["closed_form_degK"], r["GHz_per_K"]))
print("SI total (0.126+5.237+0.252) = 5.615 ; recomputed with SI constants = %.4f ; residual %.4f deg/K" %
      (SI_TOTAL, SI_TOTAL - 5.615))
print("Corning 7980 best estimate %.3f deg/K ; SI overestimate = %.2f deg/K = %.1f %%" %
      (BEST, SI_TOTAL - BEST, 100 * (SI_TOTAL - BEST) / BEST))
print("range with alpha 4.0e-7: %.3f ; Homosil: %.3f" % (slab_rows[3]["total_degK"], slab_rows[4]["total_degK"]))
print("2pi window: SI %.1f K ; best %.1f K" % (360 / SI_TOTAL, 360 / BEST))
print("pm per GHz at 1550 nm: %.4f" % PM_PER_GHZ)
print("%-62s %9s %9s %9s %8s %8s %8s" % ("platform", "L[mm]", "deg/K", "GHz/K", "pm/K", "x SI", "x best"))
for r in rows:
    if r["deg_per_K"] == "":
        print(r["platform"], r["kind"])
        continue
    Ls = ("%.3f" % r["L_10ps_mm"]) if r["L_10ps_mm"] != "" else "-"
    print("%-62s %9s %9.3f %9.3f %8.2f %8.2f %8.2f" % (r["platform"][:62], Ls, r["deg_per_K"], r["GHz_per_K"],
                                                  r["pm_per_K"], r["deg_per_K"] / SI_TOTAL, r["deg_per_K"] / BEST))
