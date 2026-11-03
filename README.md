# Code availability: slab DLI-DQPSK receiver (inverse-designed metasurface interferometer)

## Overview

This repository holds the simulation drivers, the RCWA solver, the evaluation scripts and the job files of the adopted runs that produced the numbers of the manuscript "Folding the delay line into a slab: an inverse-designed metasurface interferometer for two-dimensional arrays of polarization-diversity DQPSK receiver front ends" (submitted to Nanophotonics, 2026) and its Supplementary Material.

It reproduces: the RCWA design and re-scoring of the input metagrating, the retroreflective grating, the bottom metagrating and the metaatoms; the lens-level (LPA) read-out; the eye-budget and the wavelength-channel chain; the seam, thermal-drift and tiling evaluations; the FDTD verification drivers; and, new in this version, the matched-angle (`kx_hold`) re-evaluation of the elements and of the lens at the outer wavelength channels.

## Repository layout

- `solver/` : RCWA solver (in-house modified copy of grcwa 0.1.2 with Li's inverse rule, skewed-lattice rule and symmetry reduction; autograd 1.9.1 unmodified) and the six RCWA drivers (`solver/Version_unified/RCWA_custom/`).
- `solver/LICENSES/` : licence texts of the vendored grcwa (GPLv3) and autograd (MIT).
- `fdtd/` : drivers and post-processing of the FDTD verification (Keysight RSoft FullWAVE, commercial; the `.ind` files are written by the drivers at run time and are not included).
- `evaluation/` : scripts that compute the reported numbers from stored results (lens level, eye budget, seam, atoms palette, facts and miscellaneous).
- `evaluation/lens_level/` : lens-level evaluators at the original angle, including `edge_readout.py`, `lib_edge.py` and `make_c7_noPD.py`.
- `evaluation/outer_channels_matched_angle/` : the matched-angle chain; `group_A_elements/` (element-level chain, with `CHANGES_group_A.md`) and `group_B_lens_level/` (lens-level read-out and chain, with `CHANGES_group_B.txt`).
- `run_configs/` : job files of the adopted runs, one subfolder per run family. `elements_matched_angle_sim899/` (93 jobs) and `lens_level_matched_angle_sim900/` (110 jobs) are the matched-angle runs.
- `MANIFEST.csv` : every file with class, size, md5 of the published file, md5 of the file as it was run (`md5_as_run`) and status (see "Changes relative to the code as run").

## Requirements and install

Python 3.10.6 (the version used for the runs). Third-party packages are listed in `requirements.txt` with the versions used:

```
pip install -r requirements.txt
```

grcwa and autograd are vendored in `solver/Version_unified/RCWA_custom/_vendor/` and need no installation. The drivers insert their own folder into `sys.path`; scripts that import sibling modules from other folders need the folder of the module on `PYTHONPATH`. The FDTD drivers additionally need a licensed RSoft FullWAVE (`fullwave.exe`).

## Usage

Every driver takes one argument, the path of a job file; the `script` key of the job file names the driver that it was written for:

```
python solver/Version_unified/RCWA_custom/<driver>.py <path/to/job_file.txt>
```

Example (a matched-angle lens-level job; its `script` key is `Main_rcwa_l3_superramp.py`):

```
python solver/Version_unified/RCWA_custom/Main_rcwa_l3_superramp.py run_configs/lens_level_matched_angle_sim900/<folder>/job_<nn>_<folder>.txt
```

Drivers: `Main_rcwa_scd_c.py` (retroreflective grating, bottom metagrating, carrier and extract gratings), `Main_rcwa_input2d.py` (input metagrating), `Main_rcwa_l3_atom2d_opt.py` (metaatoms), `Main_rcwa_l3_superramp.py` and `Main_rcwa_l3_superramp5.py` (lens level), `Main_rcwa_seam_li.py` (seams), and in `fdtd/Version_unified/RCWA_custom/` `Main_fdtd_atom_verify.py`, `Main_fdtd_carrier2d_verify.py`, `Main_fdtd_input2d_verify.py`.

Results are written to `<Output_folder>/<subfolder_name>`; in the job files `Output_folder = results` (relative to the working directory). Evaluation scripts are run as `python <script>.py` after their paths are set (see below).

## Reproduction scope

- Design runs (input metagrating, retroreflective grating, bottom metagrating, metaatoms): optimiser settings, weights and seeds are in the job files. Re-runs of 2026-10-13..15 with the final-version drivers regenerated these designs bit for bit.
- Lens level (`run_configs/lens_level`): superramp / superramp5 evaluations of the lens at the original angle (1550 nm and the earlier channel runs).
- Matched angle (`kx_hold`): `kx_hold = 1` is an opt-in job key (needs `eval_only = 1`). It evaluates every wavelength channel at the incidence angle matched to that channel, sin(theta) = lambda / (Lambda n_sub), for the outer channels 1546.003 and 1554.017 nm. The key is default off; without it the code path is bit-identical to the earlier drivers, and the 1550 nm results are bit-identical (the `ctl_` jobs are the default-path controls). The drivers carrying `kx_hold` are `scd_c_grating.py`, `Main_rcwa_scd_c.py`, `Main_rcwa_l3_atom2d_opt.py`, `Main_rcwa_l3_superramp.py`, `Main_rcwa_l3_superramp5.py`.
- The 45-degree supercells of the M = 36 class are not recomputed at the matched angle.
- Interior channels (between 1546.003 and 1554.017 nm, excluding the 1550 nm anchor) are interpolated, not recomputed.
- Evaluation scripts under `evaluation/outer_channels_matched_angle/` re-assemble the element-level (group A) and lens-level (group B) numbers from the matched-angle results.

## Paths note

Absolute paths of the authors' machines are replaced by placeholder roots; the part of each path after the root is kept, so the input or result folder that was used stays identifiable. Replace the placeholders with your own locations before running:

- `<DATA_ROOT>` : root of the stored results and input arrays (`<DATA_ROOT>\Result\<run folder>\...`); input files that were staged next to job files are under `<DATA_ROOT>\jobs\`.
- `<CODE_ROOT>` : the folder that holds `Version_unified` (`solver/` or `fdtd/` of this repository).
- `<WORK_ROOT>` : working folder of the evaluation scripts (intermediate tables, figures).
- `<PAPER_ROOT>`, `<DOC_ROOT>`, `<HOME>` : manuscript, document and home folders named by a few figure scripts.

Job-file lines that only steered the job queue (staging, host, process-count injection) were removed; every physics, optimiser and input key keeps its value. Optional environment variable: `SIM_MAX_RAM_PCT` (percent) enables the RAM guard of `Main_rcwa_input2d.py`, the FDTD drivers and a few evaluation scripts; it is off by default.

## Changes relative to the code as run

The published files differ from the files that produced the results only in the following ways (per-file status and both md5 values in `MANIFEST.csv`):

- `text_cleaned`: comments, docstrings and printed / report messages were translated into English and freed of internal paths and machine names. The code itself is unchanged: for every such file the syntax tree with all string literals blanked is identical to the file as run.
- `code_cleaned`: code that only served the authors' compute cluster was removed or made machine-independent: hard-coded CPU-affinity pins, import paths to the authors' code tree (modules now resolve from the repository), lock files placed on a shared folder (now in the system temp folder), host-specific RAM guards (now opt-in through `SIM_MAX_RAM_PCT`), and the stripping of queue-injected job-file lines. Computed numbers are unchanged: a regression run of `Main_rcwa_scd_c.py` (sim304, eval-only) reproduced all 1,189 stored values and arrays bit for bit, and `Main_rcwa_l3_superramp.py` (sim780 needle) all 293 values; only timing and memory fields differ.
- `job_cleaned` / `renamed`: job files with path placeholders and cleaned comments; the run-name tokens `ctlSRV3` and `repSRV3` became `ctlB` and `repB` (three job files and their folders).
- `updated` / `new`: this README, `MANIFEST.csv`, the two `CHANGES_group_*` notes, and the root `LICENSE` (full GPLv3 text).

## Data availability

Input and design arrays (the `.npy` files named by the job files) and the raw result files are not included. They are available from the corresponding author upon reasonable request.

## Known gaps

- The job generator for the tmmerge atom variants (`gen_tmmerge.py`) is not included: it was written for the author's job queue and cannot run elsewhere. The six generated job files are in `run_configs/atoms/`.
- The record analysis behind the SI median factor 1.368 (14,435 extract records and 8,835 split records) is not available; only the arithmetic script `scratch_rf_261022/calc.py` is included.
- The 2026-10-13 re-measurement scripts `cap_vs_f_check.py` and `cap_vs_f_check_stdout.txt` are not available; `review_net_f.py` and `capture_vs_f.py` are included.
- The FDTD `.ind` files were not archived; the drivers generate them at run time. RSoft FullWAVE is commercial. FullWAVE log headers were not stored.
- The drivers as they were at run time for the retroreflective grating (2026-08-20) and the bottom metagrating (sim261-sim359) are not kept, only their final versions. The oldest archived `scd_c_grating.py` is the 2026-09-03 backup.
- The run-time driver md5 of the metaatom optimisation for 2026-09-25 .. 2026-10-07 is not recorded; the file is identical from 2026-10-08 21:18. Only the input-metagrating re-run (sim798) recorded its driver md5 (9f1ab1e8); for the other re-runs the driver version is inferred from file modification times.
- The script of the r18 round that recomputed the FDTD power-term differences from the json files is not identified.
- The scripts that measured the bottom-metagrating pattern widths (223.5, 179.8, 280.0 nm) are not available.
- The scripts of the planar TM lamellar test (Li 1.1e-5 at nG 79, Laurent 0.0014 at nG 477) and of the 6.3e-12 reproduction are not available.
- The cases of the rectangular-grating bit-identity check after the lattice-rule installation are not listed.
- The L-shaped FDTD region (5.55e9 cells, 1975 x 632 um2) has no calculation script (arithmetic only).
- The script that read the retroreflective values R_-2 0.999953, phase +0.525 deg and slope -1.075 deg/nm is not available; the sim778 job and the drivers are included.
- The Python and numpy versions of the other machines that ran jobs are not recorded (the machine that ran most of them: Python 3.10.6, numpy 2.2.6).
- `Main_rcwa_seam_li.py` carries the docstring "PROPOSED / NOT INSTALLED", but the file exists and the 32 sim784 seam jobs were run with it.
- The FDTD drivers call RSoft (`fullwave.exe`) through `core/rsoft_utils.py` and cannot be re-run without RSoft.
- Closed in this version: the matched-angle job files (sim899, sim900), the `kx_hold` driver versions and the original-angle scripts `edge_readout.py`, `lib_edge.py`, `make_c7_noPD.py`.
- Still open for the matched-angle chain: the scripts in `evaluation/outer_channels_matched_angle/` read intermediate result files that are not included (available on request), and in `group_B_lens_level/chain/fig7/g3_eye.py` two assertions of the original script are replaced by prints (see `CHANGES_group_B.txt`). The drivers `Main_rcwa_input2d.py`, `Main_fdtd_atom_verify.py` and `Main_fdtd_carrier2d_verify.py` are the packaged versions of 2026-10-25; later versions only add scheduling keys to the job-key whitelist.

## License

The repository is licensed under GPL-3.0 (see `LICENSE`). The vendored grcwa is GPLv3; the vendored autograd 1.9.1 is MIT (`solver/LICENSES/`). Keysight RSoft FullWAVE is a commercial product and is not part of this repository.

## Citation

See `CITATION.cff`. H. G. Kim and G. J. Lee, "Folding the delay line into a slab: an inverse-designed metasurface interferometer for two-dimensional arrays of polarization-diversity DQPSK receiver front ends", submitted to Nanophotonics (2026).

## Contact

Corresponding author: Gil Ju Lee, gjlee0414@pusan.ac.kr (Pusan National University).
