# -*- coding: utf-8 -*-
"""Assemble C2_results.md from the generated tables and the text blocks below (no new physics here; numbers are read from the JSON outputs)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "c2_tables.json"), encoding="utf-8"))
S = json.load(open(os.path.join(HERE, "c2_s23.json"), encoding="utf-8"))
RAW = open(os.path.join(HERE, "C2_tables_section_raw.md"), encoding="utf-8").read()
CH = ["1546.003", "1546.801", "1547.6", "1548.399", "1549.199", "1550.0", "1550.802", "1551.604", "1552.408", "1553.212", "1554.017"]

# ---------------- phase table
ph = ["| ch k | wavelength (nm) | slab geometric | retro arg R(-2) | carrier alone (n1768), for reference | atoms, palette mean (carrier included) | TOTAL common = slab + retro + atoms | same as laser offset (GHz, at 3.6 deg per GHz) | palette spread of dpsi: peak to peak / rms (deg) | TOTAL minus linear fit (deg) |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in S["phase_rows"]:
    ph.append("| %+d | %s | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.2f | %.3f / %.3f | %+.3f |" % (r["k"], r["ch"], r["geo"], r["retro"], r["carrier_only"], r["atoms_mean"], r["total_common"], r["offset_GHz_total"], r["ptp_abs_dpsi"], r["rms_abs_dpsi"], r["resid_after_linear"]))
PH = "\n".join(ph)

# ---------------- F table
Fk = S["F"]
fr = ["| ch k | wavelength (nm) | lane | F, decayed tail | F, held tail | F relative to 1550 (dB, decayed) | F relative to the ceiling-limited 1/2 S_max = 0.14815 (dB, decayed) |", "|---|---|---|---|---|---|---|"]
for c, k in (("1546.003", 5), ("1550.0", 0), ("1554.017", -5)):
    for lane in "xy":
        fd = Fk["%s|%s|decay" % (c, lane)]
        fh = Fk["%s|%s|held" % (c, lane)]
        f0 = Fk["1550.0|%s|decay" % lane]
        fr.append("| %+d | %s | %s | %.5f | %.5f | %+.3f | %.3f |" % (k, c, lane, fd, fh, 10 * math.log10(fd / f0), 10 * math.log10(fd / (0.5 * (4 / 9) ** 1.5))))
FT = "\n".join(fr)

# lane imbalance per wavelength
inp = T["input"]
imb = []
for lab in ("1546.00", "1547.60", "1550.00", "1552.41", "1554.02"):
    imb.append("%s nm: %+.5f dB" % (lab, 10 * math.log10(inp[lab]["eta_x"] / inp[lab]["eta_y"])))
IMB = "; ".join(imb)

TEXT = r"""@@STATUSLINE@@
# C2 -- wavelength-channel data, device-level interference term, isolator insertion loss (analysis of existing data)

Written 2026-10-14 (task C2). Output folder <WORK_ROOT>\paper_scdc_261011_v3\review\tbd_analysis_261014\ . Read-only access to <DATA_ROOT> (nothing written there, no RCWA solver, no deletions). Scripts: c2_extract.py (tables), c2_s23.py (totals and Eq. (S23)), c2_summ.py (palette summary), c2_report.py and c2_assemble.py (this file); outputs c2_tables.md/.json, c2_s23.md/.json, c2_summ.md, carrier_band_table_production_out.txt; copies of the original scripts with their output folder repointed: carrier_band_table_copy.py, lensband_eval_copy\lensband_eval.py and lensband_eval_v2.py (run: v2 with --terms-all-channels, v1 with --models all --no-control; v2 reproduces the stored lensband_eval_v2_result.json for all three channels, exit 0, all controls PASS).

Marking: SOURCE = value read from a file that was read (path given); ARITH = arithmetic on sourced numbers (script named); PROPOSAL = a suggestion, not a source. The PROV markers of Note S12 carry no data paths (they give the group-delay-dispersion formula and the domain-size remark); the data paths below come from <WORK_ROOT>\paper_scdc_261011_v3\facts_v3.md rows B14, C6, C7, J15 and from the notes in <WORK_ROOT>\wlband_261012\ .

## 0. Basis and caveats that apply to every number below

1. Channels: the 11 frequencies f0 + k x 100 GHz, k = +5 ... -5, i.e. 1546.003 ... 1554.017 nm (exact grid: 1546.0034, 1546.8011, 1547.5996, 1548.3989, 1549.1990, 1550.0000, 1550.8018, 1551.6044, 1552.4079, 1553.2122, 1554.0173 nm; the sim783 jobs use 1546.003, 1546.801, 1547.600, 1548.399, 1549.199, 1550.000, 1550.802, 1551.604, 1552.408, 1553.212, 1554.017).
2. Fixed incidence angle (CHECK.md D1 of <WORK_ROOT>\wlband_261012\CHECK.md): carrier, atoms and the retroreflective grating are solved at every channel at the 1550 nm angle (kx/G = 1550/lambda, +0.26 % at 1546.003 nm and -0.26 % at 1554.017 nm), the convention of sim778 and of the input-metagrating band data. In the device the input metagrating fixes kx = G, so the device would see the matched angle. All channel values below are therefore values of the fixed-angle model, not of the device; the size of the difference is not computed.
3. Delay-arm purity per channel comes from the carrier-at-token jobs (sim783 wl_tok: token of each atom, nG_eval 161, one termination, the termination held at its 1550 value; out_l3_nm 1150 for 212.5, 257.5 and 302.5 against the atom records at H3 = 875 nm; CHECK_TOK.md finding 1). The atoms are eval_only re-scores (Li, nG 1793) that reproduce the stored records at 1550 nm (night_261013.md, CHECK.md section 2).
4. The lens totals use the geometry model fixed (lens designed at 1550 nm, propagated at the channel, its own chromatic shift not charged). Re-running the evaluator with all five geometry models on the complete cells (cells plus 1550 atom terms): at 1546.003 nm fixed -0.5982 / -0.6129 dB, fixed_redref (chromatic defocus charged) -0.5988 / -0.6135, redesign -0.6015 / -0.6160; at 1554.017 nm fixed -0.4953 / -0.5051, fixed_redref -0.4968 / -0.5067, redesign -0.4931 / -0.5032 dB; the model choice moves the total by at most 0.0033 dB, so it does not matter for the verdict (lensband_eval_copy\run_v1_allmodels_stdout.txt).
5. The criteria quoted (eta at least 85 %, echo at most -20.5 dB, purity at least 0.996722 on both arms, lens total at least -0.65 dB, aperture echo at most -20.5 dB) are the 1550 nm criteria of the design. Whether they apply at every channel is an open decision (fig7_definition.md section 8.2 item 5: <WORK_ROOT>\loop_261008\fig7_plan_261010\fig7_definition.md); the tables report numbers against them and do not decide it.
6. The rho_* keys of the atom JSON (arg rho, abs rho) use the matched angle per channel (CHECK.md N4) and are not used here.

# (a) Branch efficiency and phase error versus channel

## A0. What exists per element and channel (summary)

| Element | Channels with data | Basis | Quantity |
|---|---|---|---|
| Input metagrating | 5 of 11 (+5, +3, 0, -3, -5) | Li nG 901 | branch efficiency x / y, leakage, reflection; no phase |
| Retroreflective grating | 11 of 11 | Li nG 257, fixed angle | R(-2) and arg R(-2) |
| Slab | 11 of 11 | analytic | geometric DLI phase |
| Bottom metagrating (carrier) | 11 of 11 | Li nG 257, five terminations | eta, echo, purities, bias |
| Atoms (8 levels) | 11 of 11 | Li nG 1793, eval_only | A, B, dpsi, eta, echo, purity (reference arm), delay-arm purity via token |
| Atom-level lens terms | 11 of 11 | chain a1 | deficit, purity f200 |
| Lens totals and aperture echo | 3 of 11 (+5, 0, -5), x ramps only | RCWA supercells, scalar propagation | L_tot, C_PD, aperture echo |

@@TABLES@@

### A5. Phase versus channel: common part and differential parts

Convention (derivation from Notes S2, S5 and Eq. (S13); consistent with the RCWA convention field exp(+i k z); the carrier-alone bias in the table is beta_c = arg R(0) + arg T(+1) - arg T(-1), checked against the driver key bias_extract_arms): the DLI phase of an output quadrant is the phase of arm B minus that of arm A. Arm B is R(0) at +theta, the slab twice, R(-2) of the retroreflective grating, and T(+1) at -theta; arm A is T(-1) at +theta. Hence DLI phase = slab geometric phase + arg R(-2) + dpsi, with dpsi = phi_B - phi_A of Eq. (S13) (for the carrier alone: arg R(0) + arg T(+1) - arg T(-1)). The atom job already contains the carrier (CHECK.md D2), so the carrier alone is shown for reference and is not added. All values are changes relative to 1550.000 nm, in degrees; dpsi is the usage-weighted mean over the eight levels (usage fractions 15.23, 9.27, 13.88, 11.15, 9.85, 12.14, 15.82, 12.65 % for 122.5 blank, 32.5, 77.5, 167.5, 212.5, 257.5, 302.5, 347.5; facts_v3 row H6). Script c2_s23.py, output c2_s23.md.

@@PHASETABLE@@

Reading of the table (ARITH from the sourced columns):
- Common to all outputs (all four quadrants, both branches) and removed by the single bias loop: the whole TOTAL column, which runs from +6.61 deg at 1546.003 nm to -15.61 deg at 1554.017 nm (peak to peak 22.2 deg). At 3.6 deg per GHz this is a laser offset from the grid frequency of +1.84 GHz at the blue edge and -4.34 GHz at the red edge (or about 2.8 K of slab temperature at the red edge for 5.6 deg per K, SI Table S24). The slab term is at most 4.75 deg of it: the bottom metagrating with atoms (+7.00 to -6.54 deg) contributes more than the slab and the retroreflective grating (+4.29 to -4.32 deg) as much, which the sentence of Note S12 on the slab alone (0.05 to 0.19 deg inside the signal band, 4.750 deg over the channels) does not show.
- The TOTAL is not flat but nearly linear in frequency: fit slope +2.220 deg per 100 GHz channel (retro +0.861, atoms +1.354; carrier alone +1.327). A phase that grows by 2.22 deg per 100 GHz is an extra group delay of 0.062 ps (ARITH, c2_s23.py), 0.6 % of the 10 ps delay: the effective DLI delay is 10.062 ps, the free spectral range 99.4 GHz instead of 100.0 GHz, unless the slab is thinned by about 3.65 um (10.0003 ps per 592.39 um). The sentence of Note S12 that the thickness for 10 ps is 592.39 um and that the band-minimum thickness lies 12 nm from it refers to the slab alone. After removing the linear fit the TOTAL still deviates by up to 2.71 deg at the band edges (the slab group-delay dispersion).
- Differential between the eight palette levels (positions in the aperture): dpsi spread over the palette peak to peak 5.26 deg at 1546.003, 5.13 at 1550.000, 4.86 deg at 1554.017 (rms about the weighted mean 1.55, 1.52, 1.47 deg); the change of dpsi from 1550 differs between levels by only 0.70 deg peak to peak at 1546.003 nm (+6.65 to +7.35) and 0.53 deg at 1554.017 nm (-6.86 to -6.33), rms 0.22 and 0.17 deg. So almost the whole channel dependence of dpsi is common to the levels. The lens phase A of the atoms moves by +1.71 to +3.30 deg at 1546.003 nm and -3.64 to -1.97 deg at 1554.017 nm; its usage-weighted mean (+2.56 and -2.85 deg) is a global phase of the lens (no focusing loss), the rms about it is 0.56 and 0.61 deg, a cosine penalty of 0.0004 and 0.0005 dB (ARITH, c2_extract.py; the lens totals above contain the effect through the deficit term).
- Differential between quadrants: the quadrant biases -45, -135, +135 and +45 deg are set by a rigid lateral shift d of the bottom pattern with its atoms, beta_q = -2 G d (Note S5, facts G1), and G = 2 pi / Lambda contains no wavelength; Note S5 states that Eq. (S9) is exact for any factorization rule and any truncation (a shift multiplies order m by exp(-i m G d)), so the 90 deg spacing is the same at every channel and the channel-dependent phase of the table rotates all four quadrants alike. Differential error between quadrants: 0 by the translation identity (analytic). The RCWA confirmation in Note S5 (81 harmonics: -45.00, -90.00, +180.00 deg) is at 1550 nm; no solver check exists at other channels, the +2 element of the identity is not yet verified by a solver (PROV in Note S3), and the seams between the quadrants are not modelled.
- Differential between the two branches: the second branch is an exact rotated copy (Note S2; derived, not simulated), so every channel-dependent phase in the table is common to both branches. Remaining differences: (i) the registration term 2G (d_x - d_y) = 0.5474 deg per nm of registration difference, wavelength independent (20 deg budget = 36.5 nm, 10 deg = 18.3 nm), (ii) the lane amplitude imbalance of the input metagrating, @@IMB@@ (largest 0.0019 dB; by the exact sigma_d symmetry this is solver error, Note S6), (iii) the phase of the input metagrating is not stored per wavelength and cancels inside each branch (it acts before the split into the arms; memory note of 260820 and fig7_definition.md 8.1; S4:80 of the old draft lists it, an inconsistency that stays open).
- What the loop does not remove: at design level no quadrant or branch error follows from the table; what remains is the palette spread (a within-aperture phase spread that lowers the interference overlap by about 0.0015 dB at 1550 nm for 1.5 deg rms, ARITH: sigma squared over 2) and the 0.062 ps delay error of the folded path, which acts inside the signal band (about 1.1 deg at +-50 GHz for 0.0222 deg per GHz, ARITH).

### A6. Efficiency versus channel (what can be assembled)

The device-level interference term per unit power in one branch, F = (1/2) T sqrt(R_rt) S_bar C_PD of Eq. (S23) with the definitions checked in part (b), needs T (5 channels), R_rt (11), S_bar (11) and C_PD (3). Only +5, 0, -5 have all four factors; C_PD at the edges is on the x-ramp basis, so the entries below are x-ramp values (at 1550 nm the final verdict basis x + y + 45 deg is 0.054 dB lower, part (b)). T at the edges is the nG 901 value and at 1550 nm the nG 901 value as well, so that the ratio is on one basis.

@@FTABLE@@

F rises toward the red edge (atoms: eta 86.0 to 88.3 % over the band, weighted mean) while the input metagrating falls (0.0254 dB at 1546.003 nm and 0.0270 dB at 1554.017 nm, x lane; y lane 0.0233 and 0.0285 dB) and the lens total is -0.063 dB (1546.003) and +0.026 dB (1554.017) against 1550 nm; the net variation of the interference term is 0.15 dB or less over the band on the x-ramp basis. The interference term hides what limits the band: the echo and purity criteria (tables A2, A3, A4) fail on the red side before the efficiency changes.

### A7. Gap list (what is missing, with the reason)

1. Input metagrating: no data at the six channels +4, +2, +1, -1, -2, -4; no phase per wavelength (argued to cancel); G4a echo and G4b relaunch are not stored per wavelength (only eta, leakage, reflection); band values are nG 901 while the 1550 nm judge is nG 2601 (difference 1.05e-4 in efficiency).
2. Retroreflective grating: finite-beam simulation exists at 1550 nm only; plane-wave values at 11 channels at the fixed angle.
3. Fixed-angle convention for retro, carrier and atoms (section 0 item 2): the matched-angle (kx held) evaluation is not computed; delay-arm purity by token approximation (section 0 item 3); the carrier is evaluated with the fixed termination set of five terminations, not with the actual atom reflection per channel.
4. Lens: totals and aperture echo only at 1546.003, 1550 and 1554.017 nm and only for x ramps. y ramps: the sim789 folders hold 31 of the 33 result folders present at 00:50 on 2026-10-14 (y18 at both edges missing); no y or mixed-direction total at the edges has been evaluated, so the final basis (x + y + 45 deg, worst orientation per class) exists at 1550 nm only. 45 degree band-edge cells (sim792, d18, d24, d36 at 1546 and 1554 nm): not computed. Interior channels 1551.604 and 1552.408 nm (sim793): 1 of 4 cells finished (x12_sh0 at 1551.604; x12_sh0 at 1552.408 and both x36_sh0 have no result file); 8 of 11 channels have no lens total. Aperture echo at 1554.017 nm is -20.101 dB (-20.304 dB with the measured x48 zone), above the criterion -20.5 dB on both readings.
5. Quadrant seam loss (Laurent-era 1.46 to 1.61 %) and the seam of the metaatom layer: not computed at any channel (not at 1550 nm with Li either).
6. Channel check of the translation identity (quadrant spacing) and of the rotated-copy argument for the second branch: not simulated at any wavelength.
7. Device-level chain: Eq. (S23) at 3 channels only (A6); receiver-level penalty, isolator, 47.12 % returned toward the fiber: not evaluated (separate TBD).

### A8. PROPOSAL for the two TBD sentences (not a source; wording for the authors to accept or change)

Main text (main_v4_r9.md line 125): replace the TBD by: "At the eleven channels of the 100 GHz grid the bottom metagrating and the eight metaatoms were re-evaluated at the fixed 1550 nm angle (Supplementary Note S12): the phase common to the quadrants and branches, from slab, retroreflective grating and atoms, changes from +6.6 degrees at 1546.003 nm to -15.6 degrees at 1554.017 nm, which the single bias loop absorbs, while the change of the atom phase difference differs between the palette levels by at most 0.7 degrees; the four criteria hold for the bottom metagrating alone from 1546.0 to 1552.4 nm and for all eight atoms only at 1550.0 and 1550.8 nm, and the aperture echo of the lens is -20.1 dB at 1554.017 nm [[decision: whether the criteria apply per channel]]."
SI (si_v2_r8.md line 555): replace the first sentence by "The bottom metagrating and the atoms are computed at the eleven channels (Tables [[numbers to be assigned]]); the lens cells only at 1546.003, 1550 and 1554.017 nm for x ramps (total -0.610 / -0.547 / -0.521 dB, aperture echo -21.34 / -20.82 / -20.10 dB)", and replace "The bottom metagrating, atoms and metalens cells are not computed away from 1550 nm" accordingly; add the extra group delay of 0.062 ps and the 3.65 um trim as a limitation next to the slab thickness.

# (b) SI TBD line 535: device-level interference term per unit fiber power, Eq. (S23)

## B1. Check of each stated assumption against the definitions

| Assumption of Note S12 | Check against the definitions (source) | Verdict |
|---|---|---|
| Factors of the two branches equal | T_x and T_y of the input metagrating: 0.9991374 and 0.9990713 (nG 1793), 0.9990709 and 0.9990147 (nG 2601); difference 6.6e-5 and 5.6e-5, i.e. 0.0003 dB, attributed to solver error because the pattern is exactly sigma_d symmetric (Note S6; sim752 and sim773 input2d.json). At the other wavelengths x - y is -0.0019 to +0.0019 dB (see A5). R_rt, S, C_PD: the second branch is the rotated copy of the first and was not simulated (derived in Note S2). | holds to 0.0003 dB at 1550 nm (0.0019 dB at the band edges); the equality of R_rt, S and C_PD between branches is derived, not simulated |
| S times C_PD does not count the atom transmission twice | S (Eq. (S6), (S14)): eta = 843.167 times abs(t_A) times abs(r_B t_B), the atom amplitudes and the bottom-metagrating reflection are inside S (843.167 = 100 (n_s / cos theta) / S_max with n_s / cos theta = 2.4983, the flux factor of the amplitude coefficients, SI Note S9 after Eq. (S14); identity checked to 7.6e-7 there). C_PD (Eq. (S17) to (S20)): the lens fields are E_p = U gamma_p exp(i phi) with gamma = a_meas / a_LPA, a_LPA the DFT of the complex atom amplitudes (read_class in <WORK_ROOT>\loop_261008\lens_li\lens_total_li.py: amp_t{A,B}_target / lpa a_pred), so the atom amplitudes cancel from gamma; L_tot is referenced to a unit-amplitude ideal lens (E_0, gamma = 1) and C_PD = 0.8243 x 10^(L_tot/10) to the ideal eight-level staircase (82.43 %, which holds the 0.222 dB staircase loss and the 0.54 % beam tail). The retroreflectance is omitted from S by definition (Note S3), so R_rt is not counted twice either. | holds by construction. Residual: the correlation between atom amplitude and phase position inside the LPA (a_LPA is a DFT, not mean amplitude times the ideal staircase) is in neither S_bar nor C_PD; bound 8.3e-4 relative = 0.004 dB from the palette amplitude spread (relative standard deviation of abs(t_A) 2.1 %, of abs(r_B t_B) 0.75 %) and the cosine spread of the phase error (ARITH, c2_s23.py) |
| S_ell is one aperture-averaged signal | The right average is the usage-weighted mean of the atom eta over the palette (the interference overlap is linear in abs(t_A) abs(r_B t_B)). Weighted mean 87.137 %, unweighted 87.203 %, range of the levels 85.169 to 88.515 % (sim783 1550 records); the weights are facts_v3 H6 (tm_focus_summary.json, written by the focusing script that fig7_definition.md 6.3 calls stale, frozen palette); a recomputation for P769 by nearest-level assignment of the hyperbolic lens phase (c2_usage_check.py, rms phase error 14.5 deg, usage within 1.2 percentage points of H6 per level) gives 87.148 % against 87.137 %; any weights would give 85.17 to 88.52 %, i.e. at most 0.08 dB | holds; weights uncertain by at most 0.08 dB, by 0.003 dB between the weighted and unweighted averages, by 0.0005 dB between H6 and the recomputed weights |
| S in the equation and S in Table S23 | Eq. (S23) multiplies P_ell T_ell by sqrt(R_rt) S: S must be the absolute S = sqrt(T(+) R(0) T(-)) = eta x 0.296296 (power fractions), whereas Table S23 lists S relative to the ceiling | wording defect: state S = eta S_max, S_max = (4/9)^1.5 |
| Same cos(Delta phi + beta_q) for both branches (equal bias) | The summed photocurrent keeps the factor F only if the branch biases are equal; with a difference delta between the branches and equal F the sum is multiplied by cos(delta/2): 0.0166 dB at 10 deg and 0.0665 dB at 20 deg (the two branch budgets of Note S12) | not stated in S12; add it (negligible within the budgets) |
| 1/4 quadrant share, P_x + P_y = P_in | each quadrant carries 1/4 of the beam by symmetry (Gaussian centred on the corner); the 0.54 % of the quadrant power outside the aperture is inside C_PD (82.43 % reference includes it) | holds |
| R_rt acts on arm B only | yes: sqrt(R_rt) enters through 2 sqrt(P_A P_B); plane-wave R(-2) = 0.9999530 (Li nG 257); the finite-beam share of the retroreflected order is 0.99993 (Table S23), 2.3e-5 lower, 0.0001 dB in sqrt | holds |

No assumption fails; one has a wording defect (S absolute) and one a missing condition (equal biases). The model can be evaluated.

## B2. Evaluation

F_ell = (1/2) T_ell sqrt(R_rt) S_bar C_PD, so that i_q / P_in = sum over ell of (P_ell / P_in) F_ell cos(Delta phi + beta_q) (the factor 1/2 is 2 x 1/4 of Eq. (S23); F is the interference amplitude of one output quadrant per unit power in that branch, in the same power units as P_in, before the isolator).

Factors at 1550 nm on the final lens basis (x + y + 45 deg, 1/P map, decayed tail), nG 1793 lane mean for T:

| Factor | Value | dB | Source |
|---|---|---|---|
| T, input metagrating branch efficiency (lane mean; x 0.9991374, y 0.9990713) | 0.9991043 | -0.0039 | <DATA_ROOT>\Result\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\input2d.json keys ng1793_eta_x, ng1793_eta_y (facts B3) |
| sqrt(R_rt), R_rt = 0.9999530 | 0.9999765 | -0.0001 | <DATA_ROOT>\Result\261011_scdc_retroli\261011_sim778_retroli_A001_liy_b11_n257\scdc_retro_H525_450.json key retro_R_m2 (facts C3) |
| S_bar / S_max, usage-weighted eta of the eight atoms | 0.87137 (S_bar = 0.25818) | -0.598 | sim783 atom2d_opt.json carrier_eff_pct at 1550.000, eight levels (<DATA_ROOT>\Result\261012_scdc_wlband\261012_sim783_wl_a*_l1550p000\), weights facts H6 |
| C_PD, 0.8243 x 10^(L_tot/10), L_tot = -0.6008 dB | 0.7178 | -1.440 | facts H20 / Note S10 (x + y + 45 deg, decayed); held tail -0.6217 dB gives 0.7144 |
| S_max = (4/9)^1.5 | 0.296296 | -5.283 | Eq. (S6) |
| factor 1/2 (2 x 1/4) | 0.5 | -3.010 | Eq. (S23) |
| product F | 0.09258 | -10.335 | c2_s23.py |

Result at 1550 nm: F_x = 0.09258 and F_y = 0.09257 (decayed tail; 0.09214 and 0.09213 held tail), i.e. 62.5 % (-2.04 dB) of the ceiling-limited value (1/2) S_max = 0.14815 (62.2 %, -2.06 dB, held tail). With nG 2601 for T: F_x = 0.09257, F_y = 0.09257. The two branches differ by 6.6e-5 in F (0.0003 dB, solver error), so the polarization dependence of the device at 1550 nm is 0.0003 dB or less from this term. For P_x + P_y = P_in the term is i_q / P_in = 0.0926 cos(Delta phi + beta_q) for any state of polarization (0.0921 for the held tail). The isolator insertion loss multiplies it if P_in is the fiber power before the isolator (part (c): 0.4 dB gives 0.912, 0.55 dB gives 0.881). Per channel (x-ramp basis) see table A6: 0.0907, 0.0937, 0.0950 for 1546.003, 1550.000, 1554.017 nm.

The factors of this term do not include (separate TBD): the isolator, the 47.12 % returned toward the fiber, the receiver comparison with a waveguide DLI; also not included: absorption, the beam-radius dependence, and the quadrant seams.

# (c) SI TBD line 167: isolator insertion loss in the link budget

Typical 1550 nm values from datasheets (all insertion loss without connectors unless stated):

| Source (model) | Type | Insertion loss | Other | URL |
|---|---|---|---|---|
| Laser Components Germany (Lightel), "Polarization Insensitive Mini Size Isolator", datasheet 03/23 V3, ordering code MIS with S = single stage, 15 = 1550 nm | fiber-pigtailed, polarization insensitive, SMF-28e | single stage at most 0.4 dB at 1550 nm, 23 C, all polarizations; at most 0.55 dB over 1550 +- 20 nm and 0 to 70 C; dual stage at most 0.5 dB and 0.7 dB | isolation at least 30 dB single, 45 dB dual (1550 +- 15 nm, 23 C); PDL at most 0.1 dB; return loss at least 55 dB; no typical column | https://www.lasercomponents.com/fileadmin/user_upload/home/Datasheets/lightel/isolators/mini-size-polarization-insensitive-isolator.pdf (text extracted from my saved copy web_sources\lightel_mini_size_pi_isolator.pdf) |
| Fiber Resources, "High Power Polarization Insensitive Isolator" 1550 nm (page without a model number) | fiber-pigtailed, polarization insensitive | typical 0.35 dB; at most 0.5 dB over 1550 +- 20 nm at 23 C; each added connector adds 0.3 dB | isolation typical peak 35 dB, minimum 28 dB (all polarizations, 23 C); PDL at most 0.15 dB; return loss at least 50 dB (5 dB lower per connector) | https://www.fiber-resources.com/OpticalIsolator/381.html |
| Ascentta, "Free-space isolator: polarization-dependent (1310, 1480, 1550, 1590 nm)", datasheet V.D1 of 15 July 2018, ordering type FSI-PD, code 15 = 1550 nm | free-space, polarizer based (polarization dependent) | single stage typical 0.2 dB, at most 0.25 dB; dual stage typical 0.3 dB, at most 0.4 dB (23 C) | isolation typical 40 dB single, 55 dB dual; minimum 30 and 42 dB | https://www.findlight.net/uploads/products/datasheet/ascentta-polarization-dependent-free-space-isolator-1310nm-1480nm-1550nm-1590nm_1689316006.pdf (text extracted from my saved copy web_sources\ascentta_pd_free_space_isolator.pdf) |
| Coherent, "Free Space Isolator" datasheet (2023), core codes KPGP single stage and KPGPGP dual stage, 1550 nm | free-space, polarizer based | single stage at most 0.4 dB, dual stage at most 0.60 dB (over the wavelength range 1550 +- 20 nm and the operating temperature -5 to 75 C; the datasheet calls these typical specifications) | isolation at least 20 dB single, 40 dB dual | https://www.coherent.com/content/dam/coherent/site/en/resources/datasheet/networking/free-space-isolator-ds.pdf (text extracted from my saved copy web_sources\coherent_free_space_isolator_ds.pdf) |

Reading. The receiver accepts any state of polarization (two polarization branches), so the isolator must be polarization insensitive; the polarizer-based free-space isolators of the two last rows pass one linear polarization and are listed only because they set the low end of what a free-space Faraday core costs (0.2 to 0.4 dB). I found no datasheet of a polarization-insensitive free-space (walk-off crystal) isolator with an insertion-loss figure; the Fiber Resources isolator-core datasheet came through the search in garbled form and is not used. The 11 channels span 1546.0 to 1554.0 nm, inside the 1550 +- 20 nm window of every row.
Recommended value for the link budget: 0.4 dB (fiber-pigtailed polarization-insensitive single-stage isolator at 1550 nm, 23 C, no connectors; Lightel maximum 0.4 dB, Fiber Resources typical 0.35 dB and maximum 0.5 dB), with 0.55 dB as the specified worst case over the band and 0 to 70 C, and +0.3 dB for every mated connector (Fiber Resources). A dual-stage part costs 0.5 to 0.7 dB and gives at least 45 dB isolation instead of 30 dB; the need follows from the 47.12 % returned toward the fiber (Note S8), whose rejection requirement is not set in the manuscript. In the budget the isolator multiplies the whole device term by 0.912 (0.4 dB) or 0.881 (0.55 dB).

# Open issues

1. The common phase of the elements adds a group delay of about 0.062 ps (FSR 99.4 GHz) that Note S12 does not contain; decide whether the slab is trimmed (3.65 um) and whether the sentence on the thickness for 10 ps and the 12 nm statement need the elements added. This is ARITH from the fixed-angle model.
2. Fixed-angle versus matched-angle (CHECK.md D1): all channel phases and efficiencies are of the fixed-angle model; the matched-angle model is not evaluated; before quoting phase numbers in the paper say so, or run it (outside this task: needs solver runs).
3. Per-channel criteria (decision in fig7_definition.md 8.2 item 5): against the 1550 nm criteria the bottom metagrating alone passes 1546.003 to 1552.408 nm (fails purity at 1553.212, echo and purity at 1554.017), all eight atoms pass only at 1550.000 and 1550.802 nm (3, 2, 2, 1, 1, 0, 0, 2, 2, 4, 4 levels fail at 1546.003 ... 1554.017), the lens total passes at the three evaluated channels on the x-ramp basis (margins 0.025 to 0.129 dB) and the aperture echo fails at 1554.017 nm by 0.40 dB (0.20 dB with the measured x48 zone).
4. Lens data: y ramps (y18 missing at both edges), 45 degree edge cells and interior channels are incomplete; the edge verdicts above are x ramps only, and at 1550 nm the y and 45 degree classes lower the total by 0.054 dB decayed / 0.064 dB held (x only -0.5470 / -0.5581, final -0.6008 / -0.6217 dB); the x-ramp margins at 1546.003 nm (0.040 / 0.025 dB) are smaller than that shift, so the final-basis verdict at 1546.003 nm is open and may fail (ARITH illustration, not a computed value).
5. Eq. (S23): define S as eta x S_max, add the equal-bias condition, and quote the usage weights; F at 1550 nm = 0.0926 per unit branch power (62.5 % of the ceiling-limited 0.1481).
6. Isolator: no polarization-insensitive free-space datasheet found; the recommendation is for a fiber-pigtailed polarization-insensitive isolator.
7. The usage weights (facts H6) come from the frozen-palette focusing script; my crude recomputation for P769 (c2_usage_check.py) agrees to 0.011 percentage points in S_bar eta; the stored usage file for P769 was not regenerated.

STATUS: COMPLETE
"""

body = TEXT.replace("@@TABLES@@", RAW.strip()).replace("@@PHASETABLE@@", PH).replace("@@FTABLE@@", FT).replace("@@IMB@@", IMB)
body = body.replace("@@STATUSLINE@@", "STATUS: COMPLETE (was IN PROGRESS while being written; the last line carries the final status)")
open(os.path.join(HERE, "C2_results.md"), "w", encoding="utf-8").write(body)
print(len(body), "chars")
