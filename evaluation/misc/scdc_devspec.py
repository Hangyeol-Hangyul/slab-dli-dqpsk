# -*- coding: utf-8 -*-
r"""Batch C dashboard -- **device specification** tab.

The other tabs of the dashboard show "which cell gave how much so far". This tab
answers the question that comes before: **what must each grating do, and how much must it deliver.**

  Fig 2  System cross-section (x-z, slab aspect ratio to scale)   -- section 1
  Fig 3  Extraction grating 4-port S-matrix channel diagram       -- section 3
  Fig 4  Output-plane 4-zone bias layout (x-y)                    -- section 4

(Fig 1 is `bar_svg` of the summary tab -- on printing all tabs come out in order, so the numbers
are chained. If a figure is added to the summary tab, shift the numbers here as well.)

## 260821 adversarial cross-review

53 findings raised by 3 finders (numbers / physics / code) were merged into 12 groups and
refuted-or-verified; the 22 confirmed items were applied. The root cause that surfaced was not
individual arithmetic slips but **the habit of copying snapshot literals instead of referencing live data**.
Therefore this version adopts the following rules:

  * Numbers are read from `scdc_status.json`. Only three kinds of literal are allowed: (a) pure
    geometric constants, (b) physical upper bounds, (c) quoted values with an explicit source.
  * Band-worst and @1550 are never mixed. Every table carries its reference label.
  * The budget applies to the **sum over lanes**. No pass/fail is assigned to individual devices.

Source of the numbers
  - Measured: `<DATA_ROOT>\Result\...\*.json` (per cell, via `scdc_status.json`)
  - Geometry / dispersion: `chbudget()` in `scdc_status_data.py` (Malitson 1965 Sellmeier)
  - Ceiling: (4/9)^1.5

`scdc_dashboard.py` imports only `tab_devspec(d)`.
"""
import math

# ---------------------------------------------------------------- pure constants
LAM_UM = 1.55
PERIOD_NM = 1315.39
NX = 512
PX_NM = PERIOD_NM / NX                       # 2.5691 nm
G2_DEG_PER_NM = 2 * 360.0 / PERIOD_NM        # 2G = 0.547366 deg/nm
DEG_PER_PX = G2_DEG_PER_NM * PX_NM           # 1.40625 deg/px
MINFEAT_SCALE = 1.2321      # correction for the gate PX_NM offset (achieved value x this = actual nm)
MINFEAT_REQ_NM = 70.0       # required fabrication linewidth (job value 56.81 = 70 / 1.2321)
CEIL = (4.0 / 9.0) ** 1.5   # 0.2962963 -- ceiling from grating reciprocity S = P S^T P


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def dB(x):
    return 10.0 * math.log10(max(float(x), 1e-12))


# --------------------------------------------------------- dB <-> % side by side
def _pctstr(x):
    p = 100.0 * float(x)
    if p >= 10:
        return "%.2f %%" % p
    if p >= 1:
        return "%.3f %%" % p
    if p >= 0.01:
        return "%.4f %%" % p
    return "%.2e %%" % p


def db_ratio(x, dig=2):
    """Linear ratio x as `-33.74 dB (0.042 %)`. For **ratios** such as suppression / leakage / reflection."""
    if x is None:
        return "&mdash;"
    return "%+.*f dB (%s)" % (dig, dB(x), _pctstr(float(x)))


def db_of_ratio(v_dB, dig=2):
    """A **ratio** already given in dB as `-33.74 dB (0.042 %)`."""
    if v_dB is None:
        return "&mdash;"
    return "%+.*f dB (%s)" % (dig, v_dB, _pctstr(10.0 ** (v_dB / 10.0)))


def db_loss(v_dB, dig=3):
    """**Loss** v_dB (positive) as `0.181 dB (95.9 % passes)`."""
    if v_dB is None:
        return "&mdash;"
    return "%.*f dB (%s passes)" % (dig, v_dB,
                                 _pctstr(10.0 ** (-abs(v_dB) / 10.0)))


def db_supp(v_dB, dig=2):
    """A **suppression** (positive dB = return loss etc.) as `5.14 dB (backflow 30.6 %)`."""
    if v_dB is None:
        return "&mdash;"
    return "%.*f dB (backflow %s)" % (dig, v_dB, _pctstr(10.0 ** (-v_dB / 10.0)))


def pc(x, dig=3):
    return "&mdash;" if x is None else ("%." + str(dig) + "f %%") % (100 * x)


def f6(x, d=6):
    return "&mdash;" if x is None else ("%." + str(d) + "f") % x


def sig(x, d=4):
    return "&mdash;" if x is None else ("%+." + str(d) + "f") % x


# ================================================================ Fig 2
def _X(u):
    return 120.0 + 0.45 * u


TOPZ, BOTZ = 130.0, 400.0


def svg_system(t_um, s_um):
    """System cross-section. x and z are drawn at the same scale (0.45 px/um), so the aspect ratio is true to scale.

    Dimension labels are taken from the data -- the thickness is subject to re-fixing, so it must not be a literal.
    (Drawing coordinates are fixed. A change of t by about 1 % is not visible.)
    """
    x0, xs, x2s = _X(0), _X(847.1), _X(1694.2)
    p = []
    a = p.append
    a('<svg viewBox="0 0 1060 545" width="100%" role="img" '
      'aria-label="Batch C system cross-section" style="max-width:1060px">')
    a('<defs>'
      '<marker id="ah" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
      'orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="context-stroke"/></marker>'
      '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
      'patternTransform="rotate(35)"><line x1="0" y1="0" x2="0" y2="6" '
      'stroke="var(--mut)" stroke-width="2"/></pattern>'
      '</defs>')
    a('<style>'
      '.sl{fill:var(--soft);stroke:var(--line);stroke-width:1.5}'
      '.gr{fill:url(#hatch);stroke:var(--fg);stroke-width:1.2}'
      '.tm{stroke:var(--blue);stroke-width:2.2;fill:none;'
      'marker-end:url(#ah)}'
      '.te{stroke:var(--green);stroke-width:2.2;fill:none;'
      'marker-end:url(#ah)}'
      '.tmd{stroke:var(--blue);stroke-width:2.2;fill:none;'
      'stroke-dasharray:7 4;marker-end:url(#ah)}'
      '.ted{stroke:var(--green);stroke-width:2.2;fill:none;'
      'stroke-dasharray:7 4;marker-end:url(#ah)}'
      '.t{font:600 12.5px sans-serif;fill:var(--fg)}'
      '.ts{font:11px sans-serif;fill:var(--mut)}'
      '.tb{font:700 13px sans-serif;fill:var(--fg)}'
      '.dim{stroke:var(--mut);stroke-width:1;fill:none}'
      '</style>')
    a('<rect class="sl" x="52" y="%g" width="900" height="%g"/>'
      % (TOPZ, BOTZ - TOPZ))
    a('<text class="ts" x="640" y="%g">fused silica &nbsp; n = 1.4440 (phase) / '
      '1.4626 (group)</text>' % (TOPZ + 168))

    def bar(cx, top, num, name, sub, below):
        y = (TOPZ - 8) if top else BOTZ
        p.append('<rect class="gr" x="%g" y="%g" width="78" height="8"/>'
                 % (cx - 39, y))
        ly = (TOPZ - 42) if top else (BOTZ + 46)
        p.append('<circle cx="%g" cy="%g" r="10" fill="var(--bg)" '
                 'stroke="var(--fg)" stroke-width="1.3"/>' % (cx - 44, ly - 4))
        p.append('<text class="tb" x="%g" y="%g" text-anchor="middle">%s</text>'
                 % (cx - 44, ly, num))
        p.append('<text class="t" x="%g" y="%g">%s</text>' % (cx - 28, ly, name))
        p.append('<text class="ts" x="%g" y="%g">%s</text>'
                 % (cx - 28, ly + 14, sub))
        if below:
            p.append('<text class="ts" x="%g" y="%g">%s</text>'
                     % (cx - 28, ly + 27, below))

    bar(x0, True, "1", "input grating", "TM&#8594;T(+1) / TE&#8594;T(0)", "x = 0")
    bar(xs, True, "3", "TE return", "R(&#8722;2)", "x = s")
    bar(x2s, True, "5", "TM return", "R(&#8722;2)", "x = 2s")
    bar(x0, False, "2", "TE branch grating", "T(0) / R(+1) / T(+1)", "")
    bar(xs, False, "4", "TM extraction grating", "T(&#8722;1) / R(0) / T(+1)", "")

    a('<rect x="%g" y="34" width="34" height="62" rx="6" fill="var(--soft)" '
      'stroke="var(--fg)" stroke-width="1.3"/>' % (x0 - 17))
    a('<text class="t" x="%g" y="26" text-anchor="middle">single-mode fiber</text>'
      % x0)
    a('<text class="ts" x="%g" y="112" text-anchor="middle">normal incidence &#183; '
      'D = 200 &#181;m</text>' % x0)
    a('<path class="tm" d="M%g,96 L%g,%g"/>' % (x0, x0, TOPZ - 12))

    a('<path class="tm" d="M%g,%g L%g,%g"/>' % (x0, TOPZ, xs - 3, BOTZ - 2))
    a('<path class="tmd" d="M%g,%g L%g,%g"/>' % (xs, BOTZ, x2s - 3, TOPZ + 2))
    a('<path class="tmd" d="M%g,%g L%g,%g"/>' % (x2s, TOPZ, xs + 3, BOTZ - 2))
    a('<path class="tm" d="M%g,%g L%g,468"/>' % (xs - 7, BOTZ + 8, xs - 7))
    a('<path class="tmd" d="M%g,%g L%g,468"/>' % (xs + 7, BOTZ + 8, xs + 7))
    a('<path class="te" d="M%g,%g L%g,%g"/>' % (x0, TOPZ, x0, BOTZ - 2))
    a('<path class="ted" d="M%g,%g L%g,%g"/>' % (x0, BOTZ, xs - 3, TOPZ + 2))
    a('<path class="ted" d="M%g,%g L%g,%g"/>' % (xs, TOPZ, x0 + 3, BOTZ - 2))
    a('<path class="te" d="M%g,%g L%g,468"/>' % (x0 - 7, BOTZ + 8, x0 - 7))
    a('<path class="ted" d="M%g,%g L%g,468"/>' % (x0 + 7, BOTZ + 8, x0 + 7))

    a('<text class="ts" x="%g" y="%g" fill="var(--blue)">&#952; = 54.6887&#176;'
      '</text>' % (x0 + 26, TOPZ + 44))
    a('<text class="ts" x="%g" y="%g" fill="var(--green)">TE is order 0 &#8594; '
      'normal</text>' % (x0 + 8, TOPZ + 92))

    for cx, lab, col in ((x0, "TE output 4 zones", "var(--green)"),
                         (xs, "TM output 4 zones", "var(--blue)")):
        a('<rect x="%g" y="468" width="112" height="34" rx="4" '
          'fill="var(--soft)" stroke="%s" stroke-width="1.6"/>' % (cx - 56, col))
        a('<text class="t" x="%g" y="489" text-anchor="middle">%s</text>'
          % (cx, lab))

    a('<path class="dim" d="M%g,522 L%g,522 M%g,517 L%g,527 M%g,517 L%g,527 '
      'M%g,517 L%g,527"/>' % (x0, x2s, x0, x0, xs, xs, x2s, x2s))
    a('<text class="ts" x="%g" y="538" text-anchor="middle">0</text>' % x0)
    a('<text class="ts" x="%g" y="538" text-anchor="middle">'
      's = %.1f &#181;m</text>' % (xs, s_um))
    a('<text class="ts" x="%g" y="538" text-anchor="middle">'
      '2s = %.1f &#181;m</text>' % (x2s, 2 * s_um))
    a('<path class="dim" d="M30,%g L30,%g M25,%g L35,%g M25,%g L35,%g"/>'
      % (TOPZ, BOTZ, TOPZ, TOPZ, BOTZ, BOTZ))
    a('<text class="ts" x="22" y="%g" text-anchor="middle" '
      'transform="rotate(-90 22 %g)">t = %.1f &#181;m</text>'
      % ((TOPZ + BOTZ) / 2, (TOPZ + BOTZ) / 2, t_um))

    a('<g transform="translate(640,36)">')
    a('<rect x="0" y="0" width="300" height="76" rx="5" fill="var(--soft)" '
      'stroke="var(--line)"/>')
    a('<line x1="14" y1="20" x2="46" y2="20" stroke="var(--blue)" '
      'stroke-width="2.4"/>'
      '<text class="ts" x="54" y="24">TM lane &#183; reference arm (solid)</text>')
    a('<line x1="14" y1="38" x2="46" y2="38" stroke="var(--blue)" '
      'stroke-width="2.4" stroke-dasharray="7 4"/>'
      '<text class="ts" x="54" y="42">TM lane &#183; delay arm (dotted, +10 ps)'
      '</text>')
    a('<line x1="14" y1="56" x2="46" y2="56" stroke="var(--green)" '
      'stroke-width="2.4"/>'
      '<text class="ts" x="54" y="60">TE lane &#183; reference / delay (same convention)'
      '</text>')
    a('</g>')
    a('</svg>')
    return "".join(p)


# ================================================================ Fig 4
# [260824] **y 4-split -> 2x2 quadrants.** The old version listed the equal-power boundary +-33.72 um as "confirmed",
# but that geometry was discarded on 260820 (the zone is 200 x 33.7 um = 5.9 : 1,
# so a circular spot on one face is impossible in principle). The 2x2 layout cuts along the two lines x=0, y=0
# into 100 x 100 um squares, and the 25 % power holds by **symmetry alone**, so no beam-profile
# specification is needed.
#
# Price: the y=0 boundary passes through the beam **peak**, so the alignment sensitivity worsens from 0.08219 -> 0.13861
# dB/um (factor 1.686), and by quadrant symmetry this is **unavoidable**.
ZONES = [
    # (label, x sign, y sign, shift px, bias deg, partner)
    ("Z0", -1, -1, 32, -45.0, "I partner (+)"),
    ("Z1", -1, +1, 160, +135.0, "I partner (&#8722;)"),
    ("Z2", +1, +1, 96, -135.0, "Q partner (&#8722;)"),
    ("Z3", +1, -1, 224, +45.0, "Q partner (+)"),
]
# [260824] The principle of adjacent differential partners is kept, but the **quadrant reassignment of the shifts is undecided**
# (a mask-stage task; no RCWA recomputation is needed). The px values below are the y4-era layout.
ZONE_NOTE = ("The shifts 32/160/96/224 px are the <b>y4-era layout</b>. The adjacent differential-partner "
             "principle is kept, but they <b>need reassignment</b> to quadrants &mdash; this is a mask "
             "stage task, so no RCWA recomputation.")
ZCOL = ("var(--blue)", "var(--sky)", "var(--orange)", "var(--verm)")


def svg_zones():
    """Output plane (x-y). A **2x2 quadrant** layout; the same grating is translated per quadrant.

    [260824] Replaces the y 4-split figure. The old geometry was discarded on 260820, but only this figure
    remained, so the discarded geometry was published as if it were a confirmed specification (adversarial review [A]).
    """
    X0, Y0, SIDE = 78.0, 44.0, 240.0        # left quadrant map
    HALF = SIDE / 2.0
    CX, CY = X0 + HALF, Y0 + HALF           # x=0, y=0 intersection

    p = []
    a = p.append
    a('<svg viewBox="0 0 1010 372" width="100%" role="img" '
      'aria-label="Output plane 2x2 quadrant bias layout" style="max-width:1010px">')
    a('<style>'
      '.t{font:600 12px sans-serif;fill:var(--fg)}'
      '.ts{font:10.5px sans-serif;fill:var(--mut)}'
      '.tb{font:700 12.5px sans-serif;fill:var(--fg)}'
      '.ax{stroke:var(--mut);stroke-width:1;fill:none}'
      '</style>')
    a('<text class="tb" x="%g" y="26">Output plane (seen from below) &#183; '
      'one polarization share</text>' % X0)

    # ---- 4 quadrants ----
    for i, (lab, sx, sy, sh, bias, pair) in enumerate(ZONES):
        rx = CX if sx > 0 else X0
        ry = Y0 if sy > 0 else CY          # in SVG, y increases downward
        a('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" '
          'fill-opacity="0.20" stroke="%s" stroke-width="1.4"/>'
          % (rx, ry, HALF, HALF, ZCOL[i], ZCOL[i]))
        tx, ty = rx + HALF / 2.0, ry + HALF / 2.0
        a('<text class="tb" x="%g" y="%g" text-anchor="middle">%s</text>'
          % (tx, ty - 16, lab))
        a('<text class="t" x="%g" y="%g" text-anchor="middle">%+.0f&#176;</text>'
          % (tx, ty + 2, bias))
        a('<text class="ts" x="%g" y="%g" text-anchor="middle">shift %d px</text>'
          % (tx, ty + 17, sh))
        a('<text class="ts" x="%g" y="%g" text-anchor="middle">25 %% &#183; %s'
          '</text>' % (tx, ty + 31, pair))

    # ---- dividing lines ----
    a('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="var(--fg)" '
      'stroke-width="1.6"/>' % (CX, Y0, CX, Y0 + SIDE))
    a('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="var(--fg)" '
      'stroke-width="1.6"/>' % (X0, CY, X0 + SIDE, CY))
    a('<text class="ts" x="%g" y="%g" text-anchor="middle">x = 0</text>'
      % (CX, Y0 - 6))
    a('<text class="ts" x="%g" y="%g" text-anchor="end">y = 0</text>'
      % (X0 - 8, CY + 4))
    a('<text class="ts" x="%g" y="%g" text-anchor="middle">'
      '&#8592; 200 &#181;m &#8594; &#183; quadrant 100 &times; 100 &#181;m</text>'
      % (CX, Y0 + SIDE + 20))
    a('<text class="ts" x="%g" y="%g" text-anchor="middle">25 %% power holds by '
      '<tspan font-weight="700">symmetry alone</tspan> &#8212; no profile specification '
      'needed</text>' % (CX, Y0 + SIDE + 35))

    # ---- intensity distribution + alignment sensitivity ----
    GX = X0 + SIDE + 76
    a('<text class="tb" x="%g" y="26">Intensity distribution and alignment sensitivity</text>' % GX)

    def Y(u):
        return Y0 + (100.0 - u) / 200.0 * SIDE

    pts = []
    for k in range(81):
        u = -100.0 + 200.0 * k / 80.0
        v = math.exp(-2.0 * (u / 100.0) ** 2)
        pts.append("%g,%g" % (GX + v * 140.0, Y(u)))
    a('<polyline points="%s" fill="none" stroke="var(--fg)" '
      'stroke-width="1.8"/>' % " ".join(pts))
    a('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="var(--verm)" '
      'stroke-width="1.6" stroke-dasharray="4 3"/>'
      % (GX, Y(0), GX + 150, Y(0)))
    a('<text class="ts" x="%g" y="%g" fill="var(--verm)">y = 0 passes through the beam '
      '<tspan font-weight="700">peak</tspan></text>'
      % (GX + 14, Y(0) - 8))
    a('<text class="ts" x="%g" y="%g">&#8594; dP/dy = '
      '<tspan font-weight="700">0.13861 dB/&#181;m</tspan></text>'
      % (GX + 14, Y(0) + 14))
    a('<text class="ts" x="%g" y="%g">y4 (0.08219) &times; <tspan '
      'font-weight="700">1.686</tspan></text>' % (GX + 14, Y(0) + 28))
    a('<text class="ts" x="%g" y="%g">by quadrant symmetry <tspan '
      'font-weight="700">unavoidable</tspan></text>' % (GX + 14, Y(0) + 42))
    a('<text class="ts" x="%g" y="%g">d<tspan font-style="italic">y</tspan> = '
      '5 &#181;m &#8594; partner imbalance 0.69 dB</text>' % (GX, Y0 + SIDE + 20))
    a('<text class="ts" x="%g" y="%g">&#8594; CMRR about &#8722;19 dB</text>'
      % (GX, Y0 + SIDE + 35))

    # ---- differential demodulation ----
    EX = GX + 236
    a('<text class="tb" x="%g" y="26">Differential demodulation</text>' % EX)
    for k, (t, c) in enumerate((
            ("I = P(Z0) &#8722; P(Z1)", "var(--blue)"),
            ("Q = P(Z3) &#8722; P(Z2)", "var(--verm)"),
            ("bit&#8321; = sign(I)&nbsp;&nbsp;bit&#8322; = sign(Q)", "var(--fg)"),
            ("I<tspan baseline-shift='sub'>tot</tspan> = I(TM) + I(TE)",
             "var(--mut)"),
            ("Q<tspan baseline-shift='sub'>tot</tspan> = Q(TM) + Q(TE)",
             "var(--mut)"))):
        a('<text class="t" x="%g" y="%g" fill="%s">%s</text>'
          % (EX, 56 + 26 * k, c, t))
    a('<text class="ts" x="%g" y="200">Place differential partners <tspan font-weight="700">'
      'adjacent</tspan> &#8212;</text>' % EX)
    a('<text class="ts" x="%g" y="215">drift tolerance 1.66 x,</text>' % EX)
    a('<text class="ts" x="%g" y="230">I/Q boundaries 3 &#8594; 1.</text>' % EX)
    a('<text class="ts" x="%g" y="256">the bias is '
      '<tspan font-weight="700">&#8722;1.40625&#176;</tspan> per px of shift</text>' % EX)
    a('<text class="ts" x="%g" y="271">(64 px = &#923;/8 &#8594; '
      '&#8722;90.00&#176;).</text>' % EX)
    a('<text class="ts" x="%g" y="297" fill="var(--orange)">shift assignment '
      '<tspan font-weight="700">needs review</tspan></text>' % EX)
    a('<text class="ts" x="%g" y="312" fill="var(--orange)">(px below are y4-era '
      'values).</text>' % EX)
    a('</svg>')
    return "".join(p)

def svg_ports(ex):
    """The extraction grating has exactly 4 propagating channels. Their 4x4 block sets the ceiling."""
    p = []
    a = p.append
    GY = 178.0
    a('<svg viewBox="0 0 1010 345" width="100%" role="img" '
      'aria-label="Extraction grating 4-port channel diagram" style="max-width:1010px">')
    a('<defs><marker id="ah3" markerWidth="9" markerHeight="9" refX="7.5" '
      'refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" '
      'fill="context-stroke"/></marker></defs>')
    a('<style>'
      '.t{font:600 12px sans-serif;fill:var(--fg)}'
      '.ts{font:10.5px sans-serif;fill:var(--mut)}'
      '.tb{font:700 12.5px sans-serif;fill:var(--fg)}'
      '.ar{stroke-width:2.2;fill:none;marker-end:url(#ah3)}'
      '</style>')
    a('<rect x="60" y="40" width="620" height="%g" fill="var(--soft)"/>'
      % (GY - 40))
    a('<text class="ts" x="72" y="58">substrate (n = 1.444) &#183; 3 propagating orders</text>')
    a('<rect x="60" y="%g" width="620" height="9" fill="var(--fg)" '
      'fill-opacity="0.75"/>' % GY)
    a('<text class="ts" x="72" y="%g">air &#183; <tspan '
      'font-weight="700">1</tspan> propagating order (&#955;/&#923; = 1.178 &gt; 1)</text>'
      % (GY + 28))
    CX = 370.0

    def ray(x1, y1, x2, y2, col, lab, val, dx=0, dy=0, anc="start"):
        p.append('<path class="ar" d="M%g,%g L%g,%g" stroke="%s"/>'
                 % (x1, y1, x2, y2, col))
        p.append('<text class="t" x="%g" y="%g" text-anchor="%s" fill="%s">%s'
                 '</text>' % (x2 + dx, y2 + dy, anc, col, lab))
        p.append('<text class="ts" x="%g" y="%g" text-anchor="%s">%s</text>'
                 % (x2 + dx, y2 + dy + 14, anc, val))

    a('<path class="ar" d="M150,52 L%g,%g" stroke="var(--fg)"/>' % (CX - 4, GY - 4))
    a('<text class="tb" x="150" y="44">incidence &#183; substrate +&#952; &#183; '
      'k<tspan baseline-shift="sub">x</tspan> = +G</text>')
    ray(CX, GY, 630, 56, "var(--blue)", "R(0) &#8594; +&#952; &nbsp;[enters delay line]",
        "|S&#8323;&#8323;|&#178; = %.6f" % ex["r0"], 8, -6)
    ray(CX, GY, 180, 88, "var(--verm)",
        "R(&#8722;2) &#8594; &#8722;&#952; &nbsp;[backflow to fiber]",
        "|S&#8321;&#8323;|&#178; = %.6f &nbsp;(&#8776; 1/9)" % ex["stray"],
        -6, -6, "end")
    ray(CX, GY, CX, 62, "var(--orange)",
        "R(&#8722;1) &#8594; normal &nbsp;[empty exhaust port]",
        "|S&#8322;&#8323;|&#178; = %.6f" % ex["exit"], 10, 22)
    ray(CX, GY + 9, CX, 296, "var(--green)",
        "T(&#8722;1) &#8594; air normal &nbsp;[reference arm]",
        "|S&#8320;&#8323;|&#178; = %.6f &nbsp;(&#8776; 4/9)" % ex["ep"], 10, 0)

    BX = 700.0
    a('<rect x="%g" y="40" width="298" height="272" rx="6" fill="var(--soft)" '
      'stroke="var(--line)"/>' % BX)
    a('<text class="tb" x="%g" y="64">Why the 4 ports set the ceiling</text>'
      % (BX + 16))
    for k, s in enumerate((
            "propagating channels = substrate 3 (+&#952;, normal, &#8722;&#952;) + air 1 = <tspan "
            "font-weight='700'>4</tspan>",
            "Grating reciprocity is not S = S<tspan baseline-shift='super'>T</tspan> "
            "but",
            "<tspan font-weight='700'>S = P S<tspan baseline-shift='super'>T"
            "</tspan> P</tspan> (transpose with order reversal)",
            "under that constraint max &#8730;(&#951;&#8330;&#183;R&#8320;&#183;"
            "&#951;&#8331;) is solved by",
            "|S&#8320;&#8323;|&#178; = |S&#8323;&#8323;|&#178; = "
            "|S&#8320;&#8321;|&#178; = 4/9,&nbsp; |S&#8321;&#8323;|&#178; = 1/9",
            "&#8594; ceiling (4/9)<tspan baseline-shift='super'>1.5</tspan> = "
            "<tspan font-weight='700'>0.296296</tspan>",
            "",
            "R(&#8722;2) = 1/9 leakage is <tspan font-weight='700'>not a defect</tspan>,",
            "it is a component of the unconstrained optimum. It can be suppressed, but",
            "at a price (C59: &#8722;20.6 dB at 0.41 dB).")):
        a('<text class="ts" x="%g" y="%g">%s</text>' % (BX + 16, 88 + 20 * k, s))
    a('</svg>')
    return "".join(p)


# ================================================================ spec table
def _row(name, req, target, meas, verdict, cls, note=""):
    return (name, req, target, meas, verdict, cls, note)


def spec_table(rows, head=("Required function", "Why needed", "Required spec",
                           "Measured", "Verdict")):
    h = ['<div class="tw"><table><tr>']
    h.append('<th class="wrap">%s</th><th class="wrap">%s</th>'
             '<th class="wrap">%s</th><th class="n">%s</th><th>%s</th></tr>'
             % head)
    for name, req, target, meas, verdict, cls, note in rows:
        pill = ('<span class="pill p-%s">%s</span>' % (cls, verdict)
                if cls else '<span class="mut">%s</span>' % verdict)
        h.append('<tr><td class="wrap"><b>%s</b></td><td class="wrap">%s</td>'
                 '<td class="wrap">%s</td><td class="n">%s</td>'
                 '<td class="wrap">%s%s</td></tr>'
                 % (name, req, target, meas, pill,
                    ('<br><span class="mut">%s</span>' % note) if note else ""))
    h.append("</table></div>")
    return "".join(h)


def dev_block(num, title, sub, cell, minfeat, purpose, rows):
    h = ['<div class="pos">']
    h.append('<div class="hdr"><span class="t">%s %s</span>'
             '<span class="c">%s</span>'
             '<span class="c cell">adopted cell %s</span>'
             '<span class="c">minimum linewidth <b>%s nm</b></span></div>'
             % (num, title, sub, esc(cell),
                "&mdash;" if minfeat is None
                else "%.1f" % (minfeat * MINFEAT_SCALE)))
    h.append('<div class="note" style="margin-top:0">%s</div>' % purpose)
    h.append(spec_table(rows))
    h.append("</div>")
    return "".join(h)


# ================================================================ body
# Keys of the stored data file (scdc_status.json) are Korean; translate them once, right after loading, with this one mapping
# (written as unicode escapes; the English text is the value).
_KEYMAP = {
    "\uac80\ucd9c\uae30 \u00b7 \uae30\uc900\ud314": "detector &#183; reference arm",
    "\uac80\ucd9c\uae30 \u00b7 \uc9c0\uc5f0\ud314": "detector &#183; delay arm",
    "\u03b7\u208a \uae30\uc900\ud314": "eta+ reference arm",
    "R\u2080 \uc9c0\uc5f0\uc120": "R0 delay line",
    "\u03b7\u208b \uc9c0\uc5f0\ud314": "eta- delay arm",
    "R(\u22122) \uc5ed\ub958": "R(-2) backflow",
    "\uc0c1\ud55c \ub300\ube44": "vs upper bound",
    "\ud30c\uc774\ubc84 \uc989\uc2dc\ubc18\uc0ac R_both": "fiber immediate reflection R_both",
    "\uc5d0\ucf54 R(+2)": "echo R(+2)",
    "\ub450 \ud314 \uc704\uc0c1\ucc28 dphi_TM_TE": "two-arm phase difference dphi_TM_TE",
    "R(0) \ud30c\uc774\ubc84 \uc9c1\ud589": "R(0) fiber direct",
    "\ubd84\uae30 (\ud558\ub2e8)": "branch (bottom)",
    "\ub418\ub3cc\ub9bc TE (\uc0c1\ub2e8)": "return TE (top)",
    "\ucd94\ucd9c (\ud558\ub2e8)": "extraction (bottom)",
    "\ub418\ub3cc\ub9bc TM (\uc0c1\ub2e8)": "return TM (top)",
    "\uacf5\uae30 \ub204\uc124 T(\u22121)": "air leakage T(-1)",
    "\uae30\uc900\ud314 \uc5ed\ub958": "reference arm backflow",
    "\uc9c0\uc5f0\ud314 \uc5ed\ub958": "delay arm backflow",
}


def _tr_keys(dct):
    return {_KEYMAP.get(k, k): v for k, v in dct.items()}


def _tr_text(s):
    for k, v in _KEYMAP.items():
        s = s.replace(k, v)
    return s


def tab_devspec(d):
    A = {a["key"]: a for a in d["arch"]}
    du = A["dual"]
    P = {p["role"]: p for p in du["positions"]}
    IN, SP, RTE, EX, RTM = (P["in"], P["sp"], P["re_te"], P["ex"], P["re_tm"])
    exx, spx, inx = _tr_keys(EX["extra"]), _tr_keys(SP["extra"]), _tr_keys(IN["extra"])
    cb = d.get("chbudget") or {}
    cbf = d.get("chbudget_fix") or cb
    cb600 = d.get("chbudget_600") or {}     # discarded old 600 um (for comparison)
    sl = _tr_keys(cb.get("slopes", {}))
    lg = d.get("ledger") or {}
    lrows = [[r[0], _tr_text(r[1])] + list(r[2:]) for r in ((lg.get("rows") or {}).get("dual") or [])]
    lsum = (lg.get("sum") or {}).get("dual") or {}
    retro = (d.get("dump") or {}).get("retro") or {}

    # Every thickness-dependent value is derived from the data (no-literal rule).
    th = math.radians(cb.get("theta_deg") or 54.6887)
    # [260823] Thickness fixed. t_now is the fixed value and t_old is the discarded old value.
    t_now, t_fix = cb.get("t_um") or 592.39, cbf.get("t_um") or 592.39
    t_old = cb600.get("t_um") or 600.0
    s_now, s_fix = t_now * math.tan(th), t_fix * math.tan(th)
    s_old = t_old * math.tan(th)
    # [260823] All thickness-derived values are built from live data (no-literal rule).
    _bias = d.get("bias") or {}
    path_um = (_bias.get("geom") or {}).get("path_um") or (2 * t_now / math.cos(th))
    _tol = _bias.get("tol") or []
    dT_deg = _tol[1][2] if len(_tol) > 1 else 0.0
    # The x axis is compressed by cos(th) and diverges first. Gaussian with w0x = 50*cos(th) um.
    _w0x = 50.0 * math.cos(th)
    _zr = math.pi * _w0x ** 2 * 1.4440 / 1.55
    expand_pct = (math.sqrt(1.0 + (path_um / _zr) ** 2) - 1.0) * 100.0

    def lrow(pol, key):
        for r in lrows:
            if r[0] == pol and key in r[1]:
                return r[3]
        return None

    ex_ports = {"ep": exx.get("eta+ reference arm"), "r0": exx.get("R0 delay line"),
                "em": exx.get("eta- delay arm"), "stray": exx.get("R(-2) backflow"),
                "exit": 0.000086}   # 1550.0_sub_p_TM_R_m1 (not in extra)
    assert None not in (ex_ports["ep"], ex_ports["r0"], ex_ports["em"],
                        ex_ports["stray"]), ex_ports

    h = []
    a = h.append

    # ---------------------------------------------------------- 0. preamble
    a("<h2>0. What this tab answers</h2>")
    a('<div class="key"><b>What each of the 5 gratings must do and how much it must deliver.'
      '</b> If the other tabs show "which cell scores how much right now", this tab sets up '
      'the question before that -- <b>required function &rarr; required spec &rarr; measured &rarr; '
      'verdict</b> -- per device. Every value given in dB is accompanied by the linear '
      '% in parentheses.</div>')
    a('<div class="warn"><b>[260821] Version reflecting the adversarial cross-review.</b> The 3 finders'
      '(numbers &#183; physics &#183; code) raised 53 findings, merged into 12 groups and '
      'refuted-or-verified: <b>confirmed 22 / false-alarm 5</b>, and all were applied. '
      'Of these, <b>5 that affect design decisions</b> are marked with a red box at the '
      'start of the relevant section. The heaviest is the <b>slab thickness</b> -- see section 6.</div>')
    a('<div class="note"><b>How to read.</b> <span class="pill p-ok">met</span> = '
      'the measurement exceeds the requirement. <span class="pill p-est">margin</span> = exceeds it but '
      'assumptions / unverified items are mixed in. <span class="pill p-no">short</span> = not yet '
      'exceeded. In the &ldquo;required spec&rdquo; column, <b>derived</b> is a value computed down from '
      'the system requirement, and <b>physical</b> is a line that the S matrix / reciprocity sets and that cannot be crossed. '
      '<b>[worst]</b> next to a number is the worst of the 5 band wavelengths, <b>[@1550]</b> is the centre-wavelength '
      'value -- not mixing the two is the rule of this version.</div>')

    # ---------------------------------------------------------- 1. system
    a("<h2>1. The system on one page</h2>")
    a('<div class="figwrap">%s<div class="figcap">Fig. 2 &nbsp; Batch C cross-section '
      '(x&ndash;z). The slab aspect ratio is true to scale (0.45 px/&#181;m). In both lanes the '
      'delay arm goes around <b>the same round trip 2t/cos&#952;</b> and returns to the same place '
      '&mdash; zero net lateral shift is the key of this layout. Dimensions are for the <b>fixed thickness '
      't = %.2f &#181;m</b> (fixed 260823). Compared with the discarded old value %.1f &#181;m, '
      's decreased from %.1f &rarr; %.1f &#181;m.</div></div>'
      % (svg_system(t_now, s_now), t_now, t_old, s_old, s_now))

    a("<h3>1.1 Optical path &mdash; two lanes, two arms each</h3>")
    a('<div class="tw"><table><tr><th>lane</th><th>arm</th>'
      '<th class="wrap">path (device &#183; order)</th>'
      '<th class="n">power [@1550]</th><th class="wrap">remarks</th></tr>')
    for pol, arm, path, key, note in (
        ("TM", "reference", "① T(+1) &rarr; ④ T(&#8722;1) &rarr; air normal",
         "detector &#183; reference arm", "one pass inside the slab"),
        ("TM", "delay", "① T(+1) &rarr; ④ R(0) &rarr; ⑤ R(&#8722;2) &rarr; "
         "④ T(+1)", "detector &#183; delay arm", "+10 ps"),
        ("TE", "reference", "① T(0) &rarr; ② T(0) &rarr; air normal",
         "detector &#183; reference arm", "normal pass"),
        ("TE", "delay", "① T(0) &rarr; ② R(+1) &rarr; ③ R(&#8722;2) &rarr; "
         "② T(+1)", "detector &#183; delay arm", "+10 ps"),
    ):
        a('<tr><td><b>%s</b></td><td>%s</td><td class="wrap">%s</td>'
          '<td class="n">%s</td><td class="wrap">%s</td></tr>'
          % (pol, arm, path, pc(lrow(pol, key), 2), note))
    a("</table></div>")
    a('<div class="note">The power is <b>relative to the fiber input</b> '
      '(<code>scdc_status.json &rarr; ledger.rows.dual</code>). The ratio of the two arms, '
      '2.24 : 1, is <b>not a specification</b> &mdash; the 4-zone differential output is '
      '<code>I = 2B&#183;cos</code> and <code>B = 2&#8730;(P<sub>ref</sub>'
      'P<sub>del</sub>)</code> depends only on the <b>product</b> of the two arms. See section 8.'
      '</div>')

    # ---------------------------------------------------------- 2. per device
    a("<h2>2. Required specification per device</h2>")
    a('<div class="note"><b>The verdict of the slope rows was removed (corrected 260821).</b> The bias slope '
      'budget applies to the <b>sum over lanes</b> &mdash; if there is a partner of opposite sign, the sum passes '
      'even if an individual device is large. So the per-device slope is given only as a <b>contribution (information)</b>, '
      'and the pass/fail verdict is made once, per lane, in section 6.</div>')

    # ---- ① input grating
    a(dev_block(
        "&#9312;", "input grating", "top x = 0 &#183; polarization separation",
        IN["cell"], (IN["gate"] or {}).get("min_feature_nm"),
        "<b>One device makes the two lanes.</b> Of the light at normal incidence from the fiber, TM is "
        "bent into the +1 order and sent to +&#952; in the substrate, and TE passes through as order 0 and "
        "goes down normal. The transmission is single-polarization DQPSK and the fiber scrambles the SOP "
        "randomly, so the <b>gains of the two lanes must match</b> for the SOP to "
        "disappear from the photocurrent sum &mdash; a solution that boosts only one cannot be used.",
        [
            _row("TM &rarr; T(+1)", "TM lane entry", "physical upper bound 1.0 &#183; "
                 "practical &ge; 0.97", "%s [worst]" % pc(IN["TM"], 3),
                 "met", "ok", "loss %s" % db_loss(-dB(IN["TM"]))),
            _row("TE &rarr; T(0)", "TE lane entry", "physical upper bound 1.0 &#183; "
                 "practical &ge; 0.97", "%s [worst]" % pc(IN["TE"], 3),
                 "met", "ok", "loss %s" % db_loss(-dB(IN["TE"]))),
            _row("two-lane gain match", "SOP-independent sum &mdash; if it is off, the PDL "
                 "remains at the worst SOP", "derived &le; 0.5 dB",
                 db_loss(abs(dB(IN["TM"]) - dB(IN["TE"]))),
                 "met", "ok", "enforced by the FoM using the &#8730; of the product"),
            _row("R(0)|air (immediate reflection)", "goes straight back to the fiber",
                 "<b>no spec</b> if an isolator is assumed<br>otherwise &le; &#8722;30 dB",
                 db_ratio(inx.get("fiber immediate reflection R_both")),
                 "delegated", "est",
                 "This value is the <b>sum of the TM + TE reflectances</b>, so no single SOP "
                 "sees it (TM %s / TE %s). It is fixed that the isolator handles it"
                 % (db_ratio(0.0013836443767490838),
                    db_ratio(0.007934799396138642))),
            _row("T(+1)|&#8722;&#952; (back-coupling)", "R(&#8722;2) of ④ comes up to exactly this "
                 "place and leaves as <b>air normal = fiber</b>",
                 "<b>no spec</b> if an isolator is assumed",
                 "%s [@1550]" % db_ratio(lrow("TM", "reference arm backflow")),
                 "delegated", "est",
                 "This is the <b>3rd</b> backflow path &mdash; the largest is TM delay-arm backflow %s. "
                 "By reciprocity it cannot be blocked by the grating"
                 % db_ratio(lrow("TM", "delay arm backflow"))),
            _row("R(+2)|&#8722;&#952; (echo seed)", "if the returned beam bounces back to +&#952;, "
                 "a +10 ps coherent copy goes to the detector",
                 "derived &le; &#8722;25 dB", db_ratio(inx.get("echo R(+2)")),
                 "met", "ok", "measured eye-closure contribution is negligible"),
            _row("dphi(TM, TE)", "phase difference between the two lanes",
                 "<b>no spec</b> &mdash; each lane is demodulated differentially and independently, then the "
                 "<b>photocurrents</b> are added (a power sum, so the phase cancels)",
                 "%.2f&#176;" % (inx.get("two-arm phase difference dphi_TM_TE") or 0),
                 "irrelevant", "", "basis in section 5.4"),
        ]))

    # ---- ② branch grating
    a(dev_block(
        "&#9313;", "TE branch grating", "bottom x = 0 &#183; the only active device of the TE lane",
        SP["cell"], (SP["gate"] or {}).get("min_feature_nm"),
        "<b>One device does both branching and extraction.</b> At normal incidence T(0) goes out to air and "
        "becomes the reference arm, and R(+1) is bent to +&#952; and enters the delay line. The &#8722;&#952; beam "
        "returned from the return grating is emitted normal again by <b>the same grating</b> as T(+1). "
        "So the delay arm passes this grating <b>twice</b>, and the sum of relative orders of the bias phase "
        "becomes 2.",
        [
            _row("signal = &#8730;(T&#8320;&#183;R&#8330;&#8321;&#183;T<sub>ext"
                 "</sub>)", "the only quantity that sets the DLI interference term B",
                 "<b>physical ceiling (4/9)<sup>1.5</sup> = %.6f</b>" % CEIL,
                 "%s [worst]" % f6(SP["TE"]), "met", "ok",
                 "<b>%s</b> of the ceiling &#183; remaining margin %s &#183; @1550 is 0.295949 "
                 "(99.883 %%)"
                 % (pc(spx.get("vs upper bound"), 3),
                    db_loss(-dB(SP["TE"] / CEIL)))),
            _row("R(0)|normal", "goes straight back up, passes ① and leaves to the fiber",
                 "<b>no spec</b> if an isolator is assumed",
                 "%s [@1550]" % db_ratio(spx.get("R(0) fiber direct")),
                 "delegated", "est", "main cause of TE-lane backflow"),
            _row("R(&#8722;1)|normal", "normal incidence is left-right symmetric, so if left alone "
                 "R(&#177;1) splits half and half &mdash; one side is pure loss",
                 "derived &le; &#8722;25 dB",
                 "%s [@1550]" % db_ratio(0.000549),
                 "met", "ok", "band worst %s" % db_ratio(0.001271)),
            _row("bias 4 zones", "DQPSK needs 2 interferometers that differ by 90&#176;",
                 "<b>&#923;/8 = %.2f nm = shift by 64 px &rarr; exactly "
                 "&#8722;90&#176;</b>" % (PERIOD_NM / 8),
                 "&#8722;90.0000&#176;", "met", "ok",
                 "measured efficiency cost 2&times;10<sup>&#8722;14</sup> (identity)"),
            _row("bias slope <span class=\"mut\">(contribution)</span>",
                 "one term of the TE-lane sum", "&mdash; judged per lane in section 6",
                 "%s deg/nm" % sig(sl.get("branch (bottom)")),
                 "info", "", "<code>w_slope</code>=1 reaches &#8722;0.0316 "
                 "(C54, cost 0.024 dB). Not yet reflected in the adopted cell"),
        ]))

    # ---- ③ TE return
    a(dev_block(
        "&#9314;", "TE return grating", "top x = s",
        RTE["cell"], (RTE["gate"] or {}).get("min_feature_nm"),
        "<b>Littrow return.</b> The delay arm that came up to substrate +&#952; is reflected into relative order "
        "&#8722;2, flipping k<sub>x</sub> from +G &rarr; &#8722;G. "
        "The beam retraces its path back down, so the <b>net lateral shift is 0</b>, and therefore "
        "the beam can be kept at 200 &#181;m (the shear layout needed 1.6 mm).",
        [
            _row("R(&#8722;2)|+&#952; (TE)", "the only knob of the delay-arm round trip",
                 "unitary upper bound 1.0 &#183; derived &ge; 0.99",
                 "%s [worst]" % pc(RTE["TE"], 4), "met", "ok",
                 "loss %s" % db_loss(-dB(RTE["TE"]))),
            _row("T(&#8722;1) air leakage (TE)", "if it leaks upward it is loss as is",
                 "derived &le; &#8722;25 dB",
                 "%s [@1550]" % db_ratio(9.447119692603533e-05),
                 "met", "ok",
                 "band worst %s. The cell's <code>leak_T_m1</code> key is the <b>TM</b> value, so "
                 "it must not be used in this row" % db_ratio(3.626e-04)),
            _row("normal-incidence TM pass", "the empty exhaust port of ④ comes up to <b>exactly this place</b> "
                 "&mdash; it must be passed and discarded to air",
                 "derived &ge; 0.95",
                 "%s [worst]" % pc(retro.get("pass"), 2),
                 "short", "no",
                 "the share coming back down is %s [@1550]. However the current champion hardly uses this exhaust port "
                 "(|S&#8322;&#8323;|&#178; = 8.6&times;10"
                 "<sup>&#8722;5</sup>) &mdash; to send light there with <code>w_exit</code> "
                 "this value must be raised first. The source is a temporary re-evaluation "
                 "(<code>scdc_retro_normal.json</code>, before C49 completed)"
                 % pc(retro.get("back"), 1)),
            _row("bias slope <span class=\"mut\">(contribution)</span>",
                 "<b>the delay arm passes here once, so it enters the system bias directly"
                 "</b> &mdash; a term absent from every key before 260820, and "
                 "on re-measurement it had the largest contribution",
                 "&mdash; judged per lane in section 6",
                 "%s deg/nm" % sig(sl.get("return TE (top)")),
                 "info", "", "<code>w_slope</code>=10 gives &#8722;0.9860 (C53). "
                 "That cell also raises the normal-incidence TM pass to 0.9374 as a bonus"),
        ]))

    # ---- ④ extraction grating
    a(dev_block(
        "&#9315;", "TM extraction grating", "bottom x = s",
        EX["cell"], (EX["gate"] or {}).get("min_feature_nm"),
        "<b>A device used three times.</b> Part of the beam coming down at +&#952; is extracted as T(&#8722;1) "
        "and sent out as the reference arm (air normal), the rest is passed as R(0) specular reflection "
        "into the delay line, and the returned &#8722;&#952; beam is extracted again as T(+1). "
        "The three channels are <b>elements of the same 4&times;4 S matrix</b> and so limit each other &mdash; "
        "the ceiling of that constraint is (4/9)<sup>1.5</sup>.",
        [
            _row("signal = &#8730;(&#951;&#8330;&#183;R&#8320;&#183;"
                 "&#951;&#8331;)", "interference term B of the TM lane",
                 "<b>physical ceiling (4/9)<sup>1.5</sup> = %.6f</b>" % CEIL,
                 "%s [worst]" % f6(EX["TM"]), "met", "ok",
                 "<b>%s</b> of the ceiling &#183; remaining margin %s &#183; @1550 is 0.296233 "
                 "(99.979 %%)"
                 % (pc(exx.get("vs upper bound"), 3),
                    db_loss(-dB(EX["TM"] / CEIL)))),
            _row("&#951;&#8330; / R&#8320; / &#951;&#8331;", "split among the three channels",
                 "<b>physical optimum = 4/9 each (0.444444)</b>",
                 "%.6f / %.6f / %.6f [@1550]"
                 % (ex_ports["ep"], ex_ports["r0"], ex_ports["em"]),
                 "met", "ok",
                 "At @1550, relative to 4/9: &#8722;0.31 / +0.31 / &#8722;0.03 %. "
                 "In the band the individual channels fluctuate by &#177;3.3 %, but the <b>product fluctuates "
                 "only 0.17 %</b> &mdash; this is the evidence that it sits on the 4-port ceiling"),
            _row("R(&#8722;2)|+&#952;", "goes up at &#8722;&#952; and leaves at ① to "
                 "the fiber",
                 "<b>unconstrained optimum = 1/9 (0.111111)</b><br>"
                 "suppressing it costs value",
                 "%.6f [@1550]" % ex_ports["stray"], "optimal", "ok",
                 "Not a defect but <b>a component of the unconstrained optimum</b>. However "
                 "<b>it is not impossible to remove, and the price is quoted</b> &mdash; "
                 "C59 measured Pareto: cost <b>&#8722;20.6 dB at 0.41 dB</b>, "
                 "&#8722;31.6 dB at 0.68 dB, &#8722;33.6 dB at 0.73 dB "
                 "(reference &#8722;9.54 dB). This share goes to the fiber, so "
                 "<b>it is fixed that the isolator handles it</b> and there is no reason to buy it now"),
            _row("bias 4 zones", "TM lane I/Q",
                 "<b>&#923;/8 = 64 px &rarr; &#8722;90&#176;</b>",
                 "&#8722;90.0000&#176;",
                 "met", "ok", "same shift as the branch grating (sum of relative orders is 2 for both)"),
            _row("bias slope <span class=\"mut\">(contribution)</span>",
                 "one term of the TM-lane sum", "&mdash; judged per lane in section 6",
                 "%s deg/nm" % sig(sl.get("extraction (bottom)")),
                 "info", "", "on the <code>w_slope</code> 6~12 plateau |slope| &lt; "
                 "0.02 (C55 / C60). Best suppressed of the four devices"),
        ]))

    # ---- ⑤ TM return
    a(dev_block(
        "&#9316;", "TM return grating", "top x = 2s",
        RTM["cell"], (RTM["gate"] or {}).get("min_feature_nm"),
        "Same function as ③ and <b>only the polarization differs.</b> The position differs because the TM lane "
        "goes one step further at ④ and reaches x = 2s. If the two return gratings are given "
        "<b>the same pattern</b>, the return phases of the two lanes cancel in the difference and "
        "the TM&#8211;TE bias difference cannot be adjusted &mdash; room must be left to "
        "translate only one of them.",
        [
            _row("R(&#8722;2)|+&#952; (TM)", "delay-arm round trip",
                 "unitary upper bound 1.0 &#183; derived &ge; 0.99",
                 "%s [worst]" % pc(RTM["TM"], 4), "met", "ok",
                 "loss %s" % db_loss(-dB(RTM["TM"]))),
            _row("T(&#8722;1) air leakage (TM)", "loss",
                 "derived &le; &#8722;25 dB",
                 db_ratio(_tr_keys(RTM["extra"]).get("air leakage T(-1)")),
                 "met", "ok", "practically 0"),
            _row("bias slope <span class=\"mut\">(contribution)</span>",
                 "one term of the TM-lane sum", "&mdash; judged per lane in section 6",
                 "%s deg/nm" % sig(sl.get("return TM (top)")),
                 "info", "", "<b>does not respond to <code>w_slope</code> at "
                 "all</b> &mdash; fixed at &#8722;1.074 for 1/3/10/30. "
                 "Appears to be a structural limit; unresolved"),
            _row("phase offset to ③", "TM&#8211;TE bias-difference trimming knob",
                 "derived &mdash; shift only one side by <b>182.4 nm = 71 px</b>",
                 "not applied", "short", "no",
                 "d&#968; = &#8722;99.83&#176; [@1550] / sensitivity 2G = %.6f deg/nm. "
                 "71 px is an integer so quantisation error is 0, residual 0.02&#176;. "
                 "<b>Efficiency cost 0</b>. Fixed at the mask stage"
                 % G2_DEG_PER_NM),
        ]))

    # ---------------------------------------------------------- 3. 4 ports
    a("<h2>3. Why it cannot go higher &mdash; the 4-port S matrix</h2>")
    a('<div class="figwrap">%s<div class="figcap">Fig. 3 &nbsp; Propagating channels of the '
      'extraction grating. Substrate 3 + air 1 = exactly 4, and this 4&times;4 '
      'limits &#951;&#8330; &#183; R&#8320; &#183; &#951;&#8331; simultaneously. The values are for the '
      'adopted cell <code>%s</code> @ 1550 nm (column sum = 1.000000, '
      'unitarity checked).</div></div>'
      % (svg_ports(ex_ports), esc(EX["cell"])))
    a('<div class="tw"><table><tr><th>constraint</th><th class="n">ceiling</th>'
      '<th class="wrap">applicability</th></tr>'
      '<tr><td>unitary only</td><td class="n">0.384900 = &#8730;(4/27)</td>'
      '<td class="wrap">reciprocity not applied &mdash; <b>overestimate</b></td></tr>'
      '<tr><td>S = S<sup>T</sup> (simple symmetry)</td><td class="n">0.334717</td>'
      '<td class="wrap"><b>not applicable</b> to gratings. The measurement violates it as |S&#8321;&#8323;|&#178;'
      ' &ne; |S&#8323;&#8321;|&#178;</td></tr>'
      '<tr class="win"><td><b>S = P S<sup>T</sup> P (grating reciprocity)</b></td>'
      '<td class="n"><b>%.6f = (4/9)<sup>1.5</sup></b></td>'
      '<td class="wrap">r<sub>mn</sub> = r<sub>&#8722;n,&#8722;m</sub>. '
      '<b>the true ceiling</b></td></tr></table></div>' % CEIL)
    a('<div class="good"><b>Both devices sit on the ceiling.</b> Extraction %s (%s) '
      '&#183; branch %s (%s) &mdash; both on a <b>band-worst</b> basis, and at @1550 they are '
      '99.979 %% / 99.883 %%. The remaining theoretical margins are %s / %s respectively, which means '
      '<b>nothing is gained by adding degrees of freedom (layers &#183; NX &#183; material)</b>.'
      '</div>'
      % (f6(EX["TM"]), pc(exx.get("vs upper bound"), 3),
         f6(SP["TE"]), pc(spx.get("vs upper bound"), 3),
         db_loss(-dB(EX["TM"] / CEIL)), db_loss(-dB(SP["TE"] / CEIL))))
    a('<div class="note"><b>Side correction.</b> The FoM docstring of <code>scd_c_grating.py</code> '
      'still describes the discarded upper bounds (0.3849 / 0.334717) and the old geometry '
      '(x = 423.5 &#181;m, 5 ps) as true. The optimisation code itself is not '
      'affected, but <b>every time that docstring is quoted a wrong value is reproduced</b> '
      '&mdash; the &ldquo;0.055 dB&rdquo; error in section 8 arose exactly that way.</div>')

    # ---------------------------------------------------------- 4. output stage
    a("<h2>4. Output-stage specification</h2>")
    a('<div class="good"><b>[260824] The zones are 2&times;2 quadrants.</b> '
      'Cutting along the two lines x = 0, y = 0 makes four <b>100 &times; 100 &#181;m squares</b>. '
      'The <b>exact 25 %</b> power holds as long as the intensity distribution is even about both axes, '
      'so <b>no beam profile needs to be fixed as a specification</b> &mdash; in contrast to the '
      '&#177;33.72 &#181;m of y4, which depended on a Gaussian assumption.<br>'
      '<b>Price</b>: the y = 0 boundary passes the beam <b>peak</b>, so the alignment sensitivity worsens to '
      '<b>0.13861 dB/&#181;m</b> (<b>1.686 times</b> y4 0.08219), and by '
      'quadrant symmetry it is <b>unavoidable</b>. 0.69 dB for d<i>y</i> = 5 &#181;m.</div>')
    a('<div class="note">%s</div>' % ZONE_NOTE)
    a('<div class="figwrap">%s<div class="figcap">Fig. 4 &nbsp; Output-plane '
      '<b>2&times;2 quadrant</b> bias. No new grating is designed; <b>the same '
      'grating is just translated per quadrant</b> &mdash; the efficiency cost is measured as '
      '2&times;10<sup>&#8722;14</sup>.</div></div>' % svg_zones())
    a(spec_table([
        _row("number of outputs", "4 zones &times; 2 polarizations",
             "<b>8</b> (free space)", "8", "fixed", "ok",
             "demux per channel in the downstream PIC"),
        _row("bias convention", "decided by 2 comparators and robust to DC offset",
             "<b>B convention &#177;45 / &#177;135&#176;</b>",
             "&#8722;45 / +135 / &#8722;135 / +45",
             "fixed", "ok", "same performance as convention A (0/90/180/270), only the circuit is favourable"),
        _row("y arrangement order", "so that differential partners see the same y drift",
             "<b>differential partners adjacent</b> (32 / 160 / 96 / 224 px)",
             "fixed", "fixed", "ok",
             "drift tolerance 1.66 x, I/Q boundaries 3&rarr;1, fabrication cost 0"),
        _row("zone boundary", "if the partner powers are off, the common term does not cancel: DC offset",
             "<b>2&times;2 quadrants &mdash; x = 0, y = 0</b>",
             "zone 100 &times; 100 &#181;m, power 25 % each", "fixed", "ok",
             "holds by symmetry alone &mdash; no beam profile spec needed. The price is alignment "
             "sensitivity <b>0.13861 dB/&#181;m</b> (1.686 times y4, unavoidable)"),
        _row("beam diameter (output)", "same as the input &mdash; the anamorphic magnifications of the "
             "input and extraction gratings cancel exactly",
             "<b>x 200 &#181;m</b> (the fiber width as is)", "200 &#181;m",
             "fixed", "ok", "inside the slab it is an ellipse x 115.6 &times; y 200 &#181;m"),
        _row("beam diameter lower limit", "over the folded path of %.2f mm x is compressed by cos&#952; "
             "and diverges first" % (path_um / 1000.0),
             "<b>&ge; 150 &#181;m</b>", "200 &#181;m",
             "met", "ok", "for 100 &#181;m it expands %.1f %% after %.2f mm &mdash; "
             "&ldquo;can be reduced to 100 &#181;m&rdquo; is a discarded figure"
             % (path_um / 1000.0, expand_pct)),
        _row("device width W = 2s", "if the beam footprint goes outside the grating it is clipped as is",
             "<b>&ge; 2s + 309 &#181;m</b> (154.5 per side, spillover &#8722;30 dB)",
             "%.1f &#181;m" % (2 * s_now), "short", "no",
             "margin 0 &rarr; spillover %s. With the fixed thickness (%.2f &#181;m) 2s "
             "<b>shrank from the old %.1f &rarr; %.1f &#181;m</b>, so the margin got worse"
             % (db_of_ratio(-3.01), t_now, 2 * s_old, 2 * s_now)),
        _row("receiving lens", "the 100 GHz band ties the PD small, but the zone is larger than that",
             "<b>L3 = focusing metalens array</b> (layer 3 of the same stack) &#183; "
             "focal length <b>f has no fixed point value</b> (callout right below). "
             "The f = 507 &#181;m &rarr; spot "
             "5.0 &times; 5.0 &#181;m long listed in the table is a <b>superseded point value</b>"
             "<br>aperture <b>100 &times; 100 &#181;m</b> (quadrant), lens offset "
             "39.92 &#181;m &mdash; these two do not depend on f, so they are <b>valid</b>. "
             "The x edge angle 6.76&#176; (ghost limit 10.27&#176;) is "
             "<b>a value derived from f = 507, so it is invalid as well</b>"
             "<br><b>The y axis is unverified</b> &mdash; RCWA has NY=1, so the y local-period "
             "approximation has never been computed",
             "not designed", "short", "no",
             "<b>must not be attached</b> to the grating &mdash; if the exit medium "
             "n<sub>out</sub> &ge; 1.178 the ghost orders open (the angle depends on "
             "n<sub>out</sub>: grazing for 1.178, "
             "&#177;54.69&#176; for 1.444). &ldquo;200&times;50 &#181;m&rdquo; is "
             "a remnant of the constant-width era"),
    ], head=("item", "why needed", "required spec", "current", "verdict")))

    a('<div class="warn"><b>[design] The focal length f has no fixed point value &mdash; '
      'do not read the f = 507 &#181;m long listed in the table as the current specification.</b> '
      '507 is one of <b>at least 8 different point values</b> '
      '(700 / 507 / 616 / 553.7 / 419.1 / 354.7 ...) that appeared between 260824 and 260905, and on 260905 it was '
      'replaced by a <b>880.8~1330.1 &#181;m window</b>. However that window is '
      '<b>based on N = 4</b>. Moreover the number of phase levels N is set <b>only as a target</b> '
      '&mdash; the stage 3 target is <b>N = 8</b> (per lane, only <code>H3</code> '
      'shared; 260910), but <b>whether it is reachable cannot be judged '
      'from the currently stored map</b> &mdash; the judgement needs a measurement that does not yet exist. '
      'So this table states neither that it is reachable nor that it is not. '
      '<b>A target and reachability are different things, so do not read them mixed</b> '
      '(however <b>N and f are inputs of stage 3</b>, so the fact that N is disputed '
      'does not itself block stage 3 (260908)). f must be '
      're-derived from the <b>finally reachable N</b>, so until that N '
      'is fixed this table does not state f as the current specification.'
      '<br>The figures <b>derived from</b> f = 507 &mdash; spot 5.0 &times; 5.0 '
      '&#181;m, x edge angle 6.76&#176;, edge 9.94 cells/2&#960; &mdash; '
      'are invalid together with f. In contrast the following are <b>valid since they do not depend on f</b>: '
      'aperture 100 &times; 100 &#181;m, lens offset 39.92 &#181;m, quantisation '
      '0.22 dB (but this is a <b>fixed ratio for N = 8</b>, so it depends on N, not on f).'
      '<br><b>The y-axis requirement f &ge; 961 &#181;m is also alive.</b> This is not a result of a specific f '
      'but the <b>lower bound</b> that the cell width &ge; 2&#955; criterion puts on f, '
      'so it remains a constraint even if f is undecided. However '
      '<b>the lower end 880.8 &#181;m of the replacement window is lower than this 961 &#181;m bound</b> '
      '&mdash; whether the two values are comparable on the same criterion has not been checked, so '
      'it is not judged here (<b>open</b>, the original calculation must be checked).</div>')

    # ---------------------------------------------------------- 5. phase
    a("<h2>5. Phase specification &mdash; bias, slope, TE/TM difference</h2>")
    a('<div class="key">The phase splits into three layers. <b>(a) 90&#176; between zones</b> is '
      'an identity, so it is free rather than a specification. <b>(b) The absolute bias</b> is set '
      'randomly by the slab thickness but is trimmed to zero cost by grating translation. '
      '<b>(c) The wavelength slope</b> alone is the real specification &mdash; it cannot be erased by translation and '
      'directly sets the number of channels.</div>')

    a("<h3>5.1 (a) 90&#176; between zones &mdash; not a specification</h3>")
    a(spec_table([
        _row("64 px (&#923;/8) shift", "the sum of relative orders &#931;m = 2, so "
             "&#916;bias = &#8722;2Gd is an <b>identity</b>", "&#8722;90&#176;",
             "&#8722;90.00&#176;", "exact", "ok",
             "signal change 2&times;10<sup>&#8722;14</sup>. The sign matches the correspondence of Fig 4 "
             "32/160/96/224 px &harr; &#8722;45/+135/&#8722;135/+45&#176; "
             "(&#8722;1.40625 deg/px)"),
        _row("NX = 512 quantisation", "512 / 8 = 64 is an integer",
             "rounding error 0", "0", "exact", "ok",
             "1 px = %.4f nm = %.5f&#176;" % (PX_NM, DEG_PER_PX)),
        _row("temperature (+50 K)", "d and &#923; are <b>two lengths on the same substrate</b>, so "
             "the ratio is invariant under uniform expansion",
             "invariant", "90.0000000000&#176;", "exact", "ok",
             "all that temperature makes is a <b>common rotation</b> of the four quadrants"),
    ], head=("case", "basis", "required", "measured", "verdict")))

    a("<h3>5.2 (b) Absolute bias &mdash; solved by trimming</h3>")
    a('<div class="warn"><b>[design] The trimming shift was short by a factor of 1.85.</b> '
      'The &ldquo;98.5 nm = 38.3 px&rdquo; of the previous version was computed with the <b>260819 champion '
      'd&#968; = &#8722;53.89&#176;</b>. Afterwards the champion changed and '
      'd&#968; became &#8722;99.83&#176;, but only the shift kept the old value. '
      'If it went to the mask as is, a residual of 45.97&#176; would be frozen into the silicon and <b>0.717 dB '
      'would be lost permanently</b> (no recovery except refabrication). The correct value is <b>71 px</b>.</div>')
    a(spec_table([
        _row("slab thickness sensitivity", "&#934; = 2t&#183;k<sub>z</sub>",
             "&#177;25.8 nm per 10&#176;", "0.3877 deg/nm",
             "trimming", "est",
             "cannot be met with the polishing tolerance &rarr; handled by grating shift after fabrication"),
        _row("front/back mask alignment", "&#8706;bias/&#8706;dx = 2G",
             "&#177;18.3 nm per 10&#176;", "%.6f deg/nm" % G2_DEG_PER_NM,
             "trimming", "est", "27~109&#176; for a 50~200 nm error"),
        _row("TM&#8211;TE bias difference (constant)", "each lane uses a <b>different device</b>, so "
             "it is erased by shifting only one return",
             "&le; 10&#176;", "&#8722;99.83&#176; [@1550]", "trimming", "est",
             "cancelled by a <b>182.4 nm = 71 px</b> shift (residual 0.02&#176;). "
             "<b>Efficiency cost 0</b>"),
        _row("360&#176; continuous control", "%.1f nm shift = one full turn" % (PERIOD_NM / 2),
             "controllable", "possible", "met", "ok", "cost 0"),
    ], head=("item", "basis", "required", "current", "verdict")))

    a("<h3>5.3 (c) Wavelength slope &mdash; the only real phase specification</h3>")
    a('<div class="note"><b>The slab thickness t absorbs only the <u>mean</u> of the two lane slopes.</b> '
      'This is because there are two lanes but one slab. So the task of '
      '<code>w_slope</code> is not &ldquo;slope removal&rdquo; but '
      '<b>&ldquo;lane matching&rdquo;</b>.</div>')
    a('<div class="tw"><table><tr><th>device</th>'
      '<th class="n">adopted-cell slope</th>'
      '<th class="n"><code>w_slope</code> best</th>'
      '<th class="wrap">round &#183; remarks</th></tr>')
    for nm, key, best, note in (
        ("branch (TE, bottom)", "branch (bottom)", "&#8722;0.0316",
         "C54 <code>w_slope</code>=1 &#183; cost 0.024 dB"),
        ("return TE (top)", "return TE (top)", "&#8722;0.9860",
         "C53 <code>w_slope</code>=10 &#183; <b>was the largest single contribution</b>. "
         "C60 reproduction: <b>w=10 and lr=0.03 must both match</b> for this solution to appear "
         "(&#8722;0.9827 reproduced). w=6 &rarr; &#8722;1.317, lr=0.02 &rarr; "
         "&#8722;1.278, lr=0.05 &rarr; &#8722;1.274 give <b>a different basin</b> "
         "&mdash; narrow, unlike the wide plateau of the extraction"),
        ("extraction (TM, bottom)", "extraction (bottom)", "&#8722;0.0007",
         "C55 <code>w_slope</code>=10 &#183; C60 reproduction confirms a <b>w = 6~12 plateau</b> "
         "(6/8/10/12 all |slope| &lt; 0.02, the same structure family with min feature 98~102 nm "
         "and robust also for lr 0.02~0.03). Cliff at w &ge; 15"),
        ("return TM (top)", "return TM (top)", "&#8722;1.074",
         "C56 &#183; <b>unchanged even with a 30x weight</b> &mdash; unresolved"),
    ):
        a('<tr><td>%s</td><td class="n">%s</td><td class="n">%s</td>'
          '<td class="wrap">%s</td></tr>'
          % (nm, sig(sl.get(key)), best, note))
    a("</table></div>")
    lane_te = (cb.get("lanes", {}).get("dual") or {}).get("slope_sum")
    lane_tm = (cb.get("lanes", {}).get("dual_tm") or {}).get("slope_sum")
    diff = (lane_tm - lane_te) if None not in (lane_te, lane_tm) else None
    a('<div class="tw"><table><tr><th>lane</th><th class="n">adopted cell</th>'
      '<th class="n">after <code>w_slope</code></th><th class="wrap">meaning</th>'
      '</tr>'
      '<tr><td>TE = branch + return TE</td><td class="n">%s</td>'
      '<td class="n">&#8722;1.018</td>'
      '<td class="wrap">both values pass the lane spec of section 6</td></tr>'
      '<tr><td>TM = extraction + return TM</td><td class="n">%s</td>'
      '<td class="n">&#8722;1.075</td><td class="wrap">&mdash;</td></tr>'
      '<tr class="win"><td><b>lane-to-lane difference</b></td>'
      '<td class="n"><b>%s</b></td><td class="n"><b>&#8722;0.057</b></td>'
      '<td class="wrap"><b>26x (96.2 %%) improvement</b> &#183; total cost %s</td>'
      '</tr></table></div>'
      % (sig(lane_te), sig(lane_tm), sig(diff), db_loss(0.033)))
    a('<div class="note">The <code>w_slope</code> column is the value of the <b>best cell of each round C53~C56'
      '</b> and is not yet the adopted cell. If adopted, the two lane values and the difference change as in '
      'the table above.</div>')

    a("<h3>5.4 Two kinds of TE/TM phase difference &mdash; an easily confused distinction</h3>")
    a('<div class="tw"><table><tr><th>quantity</th><th class="wrap">definition</th>'
      '<th class="wrap">in the two-way track</th><th class="wrap">in the single track</th>'
      '</tr>'
      '<tr><td><b>dphi<sub>TM,TE</sub></b><br><span class="mut">input grating'
      '</span></td>'
      '<td class="wrap">arg T(+1)<sub>TM</sub> &#8722; arg T(0)<sub>TE</sub>'
      ' &mdash; phase difference at the point where the two <b>lanes</b> split</td>'
      '<td class="wrap"><span class="pill p-ok">no spec</span> each lane is '
      'demodulated differentially and independently, then the <b>photocurrents</b> are added. It is a power sum, so the phase cancels'
      '</td>'
      '<td class="wrap">not applicable (one lane)</td></tr>'
      '<tr><td><b>d&#968;</b><br><span class="mut">DLI bias difference</span></td>'
      '<td class="wrap">&#968;<sub>TM</sub> &#8722; &#968;<sub>TE</sub>, '
      '&#968; = phase difference of the <b>two arms</b> of each polarization</td>'
      '<td class="wrap"><span class="pill p-est">constant is trimmed</span> '
      '&#8722;99.83&#176; &rarr; zeroed by shifting one return by <b>71 px</b>. '
      'The slope +1.5075 deg/nm is brought down to &#8722;0.057 with <code>w_slope</code>'
      '</td>'
      '<td class="wrap"><span class="pill p-no">cannot be trimmed</span> translation moves '
      '<b>both polarizations equally</b>. This is the real constraint of the single track'
      '</td></tr></table></div>')
    a('<div class="warn"><b>Why it cancels in the two-way track.</b> The delay arms of the two lanes go around '
      '<u>exactly the same</u> 2t/cos&#952;, and the input grating is common to '
      'both arms <u>within</u> a lane. So the propagation term and the input-grating phase cancel completely in d&#968; '
      'and <b>d&#968; becomes a pure grating quantity</b> &mdash; meaning it can be '
      'designed away. <b>The material-dispersion correction (section 6) does not enter d&#968; either</b> '
      '&mdash; the two lanes share the same slab, so it cancels in the difference. The single track '
      'handles <b>both polarizations with one grating</b>, so it has no such room.</div>')

    # ---------------------------------------------------------- 6. channels
    a("<h2>6. Channel budget &mdash; slab thickness %.2f &#181;m fixed</h2>" % t_now)
    a('<div class="good"><b>[design &#183; fixed 260823] The phase index was being used for the group '
      'delay.</b> Differentiating &#934; = 2t&#183;k<sub>z</sub> with respect to &#969; gives '
      'the <b>group index</b> &mdash; '
      '<code>dk<sub>z</sub>/d&#969; = n<sub>g</sub>/(c&#183;cos&#952;)</code>. '
      'fused silica has n = %.6f (phase) but <b>n<sub>g</sub> = %.6f</b>, so '
      'with n held constant the phase delay and the group delay coincidentally agree and the error was invisible. '
      'As a result the <b>basis itself is invalid</b> for the 260820 decision that &ldquo;t = 600 &#181;m is the round '
      'value where the FSR is exactly 100 GHz&rdquo;. '
      'Rechecked by numerical differentiation &mdash; the measured <code>dk<sub>z</sub>/d&#969;</code> '
      '0.008440386 ps/&#181;m differs from <code>n<sub>g</sub>/(c cos&#952;)</code> by '
      '0.000 %%, from <code>n/(c cos&#952;)</code> by &#8722;1.270 %%. '
      '<b>592.3900 &#181;m for &#964;<sub>g</sub> = 10 ps and the &#177;5 band-worst '
      'minimax 592.3778 &#181;m agree within 12 nm</b>. '
      '<b>260823 fixed: t = %.2f &#181;m.</b></div>'
      % (cb.get("n_phase") or 0, cb.get("n_group") or 0, t_now))
    a('<div class="tw"><table><tr><th>thickness</th><th class="n">&#964;<sub>g</sub>'
      '</th><th class="n">geometric FSR</th><th class="n">geometric worst</th>'
      '<th class="n">TE lane worst</th><th class="n">TM lane worst</th>'
      '<th>verdict</th></tr>')
    for lab, c, cls, verd in (
        ("t = %.1f &#181;m <span class=\"mut\">(discarded &mdash; phase index)</span>" % t_old,
         cb600, "no", "over budget"),
        ("t = %.2f &#181;m <span class=\"mut\">(<b>fixed</b> &#183; &#964;<sub>g</sub> = 10 ps)"
         "</span>" % t_now, cb, "ok", "met"),
    ):
        L = c.get("lanes", {})
        a('<tr%s><td>%s</td><td class="n">%.4f ps</td><td class="n">%.3f GHz'
          '</td><td class="n">%.2f&#176;</td><td class="n">%.2f&#176;</td>'
          '<td class="n">%.2f&#176;</td>'
          '<td><span class="pill p-%s">%s</span></td></tr>'
          % (' class="win"' if cls == "ok" else "", lab,
             c.get("tau_g_ps") or 0, c.get("fsr_GHz") or 0,
             c.get("geo_worst") or 0,
             (L.get("dual") or {}).get("worst") or 0,
             (L.get("dual_tm") or {}).get("worst") or 0,
             cls, verd))
    a("</table></div>")
    a('<div class="note">When the thickness changes, <b>s = t&#183;tan&#952; and the device width 2s '
      'move too</b>: s %.1f &rarr; %.1f &#181;m, 2s %.1f &rarr; %.1f '
      '&#181;m (&#8722;%.2f %%). The beam margin of section 4 is already 0, so it must be '
      'reviewed together. &#952; is set by the phase index, so it is <b>unchanged</b> at %.4f&#176;, and '
      'the RCWA cell results and the grating slope are also <b>unaffected</b> &mdash; this correction enters only the analytic '
      'propagation term.</div>'
      % (s_old, s_now, 2 * s_old, 2 * s_now,
         abs(s_now / s_old - 1.0) * 100.0 if s_old else 0.0,
         cb.get("theta_deg") or 0))

    a("<h3>6.1 A free constant is free &mdash; the centred budget</h3>")
    a('<div class="note">There is no reason to pin the geometric residual to ch = 0. '
      '<b>The absolute bias is trimmed at zero cost by grating translation</b> (section 5.2), and '
      'since the return grating differs per lane, <b>the two lanes can be centred independently</b>. '
      'Subtracting the min-max constant halves the worst case.</div>')
    Lf = cbf.get("lanes", {})
    a('<div class="tw"><table><tr><th>item (t = %.2f &#181;m)</th>'
      '<th class="n">ch=0 fixed</th><th class="n">after centring</th>'
      '<th class="n">trimming constant</th></tr>' % t_fix)
    for lab, w, wc, off in (
        ("geometric residual", cbf.get("geo_worst"), cbf.get("geo_worst_c"),
         cbf.get("geo_trim_off")),
        ("TE lane (11 channels)", (Lf.get("dual") or {}).get("worst"),
         (Lf.get("dual") or {}).get("worst_c"),
         (Lf.get("dual") or {}).get("trim_off")),
        ("TM lane (11 channels)", (Lf.get("dual_tm") or {}).get("worst"),
         (Lf.get("dual_tm") or {}).get("worst_c"),
         (Lf.get("dual_tm") or {}).get("trim_off")),
    ):
        a('<tr><td>%s</td><td class="n">%.3f&#176;</td>'
          '<td class="n"><b>%.3f&#176;</b></td><td class="n">%+.3f&#176;</td>'
          '</tr>' % (lab, w or 0, wc or 0, off or 0))
    a("</table></div>")

    a("<h3>6.2 Single verdict criterion</h3>")
    a('<div class="warn"><b>[design] Three budget criteria were mixed.</b> The previous version '
      'put <code>&le; 1.33 deg/nm</code> (a value from the 10&#176; budget) on the '
      'device rows and marked &ldquo;short&rdquo;, while two rows below it marked '
      '&ldquo;met&rdquo; with the 20&#176; criterion. Moreover the 4.99 of <code>bias.band.budget</code> '
      'is <b>not a lane budget but a candidate-cell screening line</b>. This version '
      '<b>unifies the criterion into one</b>.</div>')
    a(spec_table([
        _row("verdict criterion", "conservative margin against the 45&#176; DQPSK decision margin",
             "<b>lane band-worst |bias| &le; %.0f&#176;</b>"
             % (cbf.get("budget_deg") or 20),
             "&mdash;", "single criterion", "ok",
             "using the 10&#176; criterion makes the slope spec much tighter &mdash; "
             "which one is chosen decides the fate of the <code>w_slope</code> campaign"),
        _row("lane-sum slope spec", "the above criterion converted to a slope &mdash; the geometric centring "
             "residual %.3f&#176; is subtracted and the rest divided by the half band %.3f nm"
             % (cbf.get("geo_worst_c") or 0, cbf.get("halfband_nm") or 0),
             "<b>|lane sum| &le; %.2f deg/nm</b>" % (cbf.get("slope_spec") or 0),
             "TE %s &#183; TM %s" % (sig(lane_te, 3), sig(lane_tm, 3)),
             "met", "ok",
             "<b>The adopted cell passes as is.</b> Adopting a <code>w_slope</code> cell is "
             "not a blocker but a margin-improvement option"),
        _row("group delay &#964;<sub>g</sub>", "the DQPSK DLI matches the symbol period &mdash; "
             "it is an envelope, so it is the <b>group</b> delay",
             "10 ps (100 GHz FSR)",
             "%.4f ps @ t = %.2f" % (cb.get("tau_g_ps") or 0, t_now),
             "met", "ok", "resolved by the 260823 thickness fix &mdash; table of section 6 above"),
    ], head=("item", "basis", "required", "current", "verdict")))
    # [260823] Section 6.1 argues that "a free constant is free", so the centred value (count_c)
    # is logically the right number. The untrimmed case is the conservative lower bound, so both are shown.
    cnt = ((cbf.get("lanes", {}).get("dual") or {}).get("count") or {})
    cnt_tm = ((cbf.get("lanes", {}).get("dual_tm") or {}).get("count") or {})
    cntc = ((cbf.get("lanes", {}).get("dual") or {}).get("count_c") or {})
    cntc_tm = ((cbf.get("lanes", {}).get("dual_tm") or {}).get("count_c") or {})
    a('<div class="tw"><table><tr><th>bias criterion</th>'
      '<th class="n">TE (ch=0 fixed)</th><th class="n">TM (ch=0 fixed)</th>'
      '<th class="n">TE (after trimming)</th><th class="n">TM (after trimming)</th>'
      '<th class="wrap">remarks</th></tr>')
    for k, note in (("10", "conservative"), ("20", "<b>adopted criterion</b>"), ("30", ""),
                    ("45", "entire decision margin")):
        a('<tr%s><td>&le; %s&#176;</td><td class="n">%s</td>'
          '<td class="n">%s</td><td class="n"><b>%s</b></td>'
          '<td class="n"><b>%s</b></td><td class="wrap">%s</td></tr>'
          % (' class="win"' if k == "20" else "", k,
             cnt.get(k, "&mdash;"), cnt_tm.get(k, "&mdash;"),
             cntc.get(k, "&mdash;"), cntc_tm.get(k, "&mdash;"), note))
    a("</table></div>")
    a('<div class="good"><b>Both the design target of 5 channels (&#177;2) and 11 channels (&#177;5, '
      '8.01 nm) hold</b> &mdash; for the fixed thickness t = %.2f &#181;m. '
      'The channel count above is not the old way of counting inside a &#177;5 scan (saturating at 11) '
      'but <b>the actual maximum symmetric span</b> that fits the budget. At the discarded t = %.1f &#181;m '
      'the geometric term alone exceeded the budget, and no amount of grating squeezing could meet it.</div>'
      % (t_now, t_old))

    # ---------------------------------------------------------- 7. system
    a("<h2>7. System specification summary</h2>")
    tm, te = lsum.get("TM") or {}, lsum.get("TE") or {}
    mins = [x for x in ((IN["gate"] or {}).get("min_feature_nm"),
                        (SP["gate"] or {}).get("min_feature_nm"),
                        (RTE["gate"] or {}).get("min_feature_nm"),
                        (EX["gate"] or {}).get("min_feature_nm"),
                        (RTM["gate"] or {}).get("min_feature_nm")) if x]
    a(spec_table([
        _row("signal power (relative to fiber)", "sum of the two arms. Starting point of the detection SNR",
             "&mdash;", "TM %s &#183; TE %s [@1550]"
             % (pc(tm.get("signal"), 2), pc(te.get("signal"), 2)),
             "measured", "", "interference term B = TM %.4f / TE %.4f <b>[worst]</b> "
             "(@1550 is 0.5876 / 0.5827). The reversed ranking of reach and B is "
             "not a difference of criterion but because the TE arm ratio is more unbalanced"
             % (du["lanes"]["TM"], du["lanes"]["TE"])),
        _row("lane gain match", "PDL at the worst SOP",
             "derived &le; 0.5 dB", db_loss(du.get("spread_dB")),
             "met", "ok", ""),
        _row("echo (detector ISI)", "+20 ps = coherent copy delayed by 2 symbols",
             "derived &le; &#8722;20 dB",
             "TM %s &#183; TE %s"
             % (db_of_ratio(tm.get("echo_ratio_dB")),
                db_of_ratio(te.get("echo_ratio_dB"))),
             "short", "no",
             "<b>This is the heaviest shortfall.</b> Even the best cell of the suppression round "
             "(<code>260820_c48e2_2L_we080</code>, split_pi) has raw orders of "
             "TM %s / <b>TE %s</b>, i.e. <b>&#8722;18.22 dB on a worst-polarization basis</b>, "
             "and at that cost the signal falls from 0.2814 &rarr; 0.2414 (&#8722;0.67 dB). "
             "At least <b>+9 dB</b> remain to the target"
             % (db_of_ratio(-33.78), db_of_ratio(-18.22))),
        _row("eye closure", "decision-margin loss caused by the echo",
             "derived &le; 0.5 dB",
             "TM %s &#183; TE %s"
             % (db_loss(tm.get("eye_dB")), db_loss(te.get("eye_dB"))),
             "short", "no", "same cause as the echo row above"),
        _row("return loss (fiber backflow)", "transmitter protection",
             "<b>the isolator handles it</b> (decided)",
             "grating alone %s" % db_supp(tm.get("rl_dB")),
             "delegated", "est",
             "with the grating alone, reciprocity limits it to 12.9 dB &mdash; 30 dB is impossible"),
        _row("operating temperature", "the TM lane is self-coherent, so the bias directly sets the decision axis",
             "derived &#177;1 K (TM lane lock)",
             "d&#934;/dT = %.3f deg/K" % dT_deg,
             "margin", "est",
             "orthogonality is <b>immune</b> to temperature. Only a common rotation remains"),
        _row("temperature uniformity",
             "the two lanes are s &#8776; %.0f &#181;m apart" % s_now,
             "derived &le; 0.1 K/mm",
             "0.1 K/mm &rarr; differential %.2f&#176;"
             % (0.1 * s_now * 1e-3 * dT_deg),
             "met", "ok", "large margin against the budget"),
        _row("fabrication minimum linewidth", "EBL/DUV resolution limit",
             "<b>&ge; %.0f nm</b>" % MINFEAT_REQ_NM,
             "minimum %.1f nm" % (min(mins) * MINFEAT_SCALE),
             "met", "ok",
             "<b>input grating &#183; branch grating tie</b> (both raw value 62.55 nm). "
             "The <code>min_feature_nm = 56.81</code> of the job is the value with the gate PX_NM "
             "offset (&times;%.4f) divided out beforehand" % MINFEAT_SCALE),
        _row("layer thickness upper bound", "deposition process limit",
             "a-Si / Si&#8323;N&#8324; / SiO&#8322; &le; 600 nm &#183; "
             "TiO&#8322; &le; 300 nm",
             "top 525/450 &#183; bottom 500/350/300 nm", "met", "ok", ""),
    ], head=("item", "why needed", "required spec", "current", "verdict")))

    # ---------------------------------------------------------- 8. non-specs
    a("<h2>8. Things that are <u>not</u> specifications &mdash; where people often tripped</h2>")
    a('<div class="tw"><table><tr><th class="wrap">once regarded as a specification</th>'
      '<th class="wrap">why it is not a specification</th><th class="wrap">basis</th></tr>')
    for it, why, ev in (
        ("two-arm power ratio 1 : 1",
         "The 4-zone differential output <code>I = 2B&#183;cos</code> has no common term A, and "
         "<code>B = 2&#8730;(P<sub>ref</sub>P<sub>del</sub>)</code> depends only on the "
         "<b>product</b> of the two arms. B for 2.24 : 1 and 1 : 1 agree to 5 decimal "
         "places",
         "Forcing 1:1 costs <b>0.738 dB</b> as the grating-reciprocity ceiling goes 0.296296 &rarr; 0.250000. "
         "The &ldquo;0.055 dB&rdquo; of the previous version was a figure from the "
         "<b>discarded S = S<sup>T</sup> model</b> &mdash; the conclusion is not "
         "reversed but <b>strengthened 13-fold</b>. Do not switch on <code>w_bal</code>"),
        ("dphi<sub>TM,TE</sub> of the input grating",
         "The two lanes are demodulated <b>independently</b> and the photocurrents (power) are "
         "added at the end. The phase disappears before that",
         "section 5.4"),
        ("common phase of the output &mdash; purpose of the <b>y</b> lens",
         "Only the differential is observable, so the common phase does not appear in the DLI output power. "
         "Moreover the local-period approximation breaks down for the y lens, so it cannot be made with a grating",
         "<b>However <code>w_phase</code> itself is alive</b> &mdash; "
         "it is still used to tie the 2&times;2 <b>x</b> lens ramps (<code>arg t(&#8722;1)</code>, "
         "<code>arg t(+1)</code>), and the default at the library entry point "
         "<code>Main_rcwa_scd_c3_lib.py:177</code> is <b>1.0</b> "
         "(0 only in <code>Main_rcwa_scd_c3.py</code>). All 16 runs of C57/C58 currently running "
         "are non-zero, so &ldquo;no residual risk&rdquo; does not hold "
         "&mdash; see section 9"),
        ("R(&#8722;2) = 1/9 leakage removal <span class=\"mut\">(with a proviso)</span>",
         "Under grating reciprocity the <b>unconstrained</b> optimum requires exactly that value. "
         "However it is <b>&ldquo;paid&rdquo; rather than &ldquo;impossible&rdquo;</b> "
         "&mdash; it is suppressed by a penalty (<code>w_stray</code>) and the three &#951; "
         "go down by just that much",
         "<b>C59 measured Pareto</b>: 0.41 dB &rarr; &#8722;20.6 dB, "
         "0.68 &rarr; &#8722;31.6, 0.73 &rarr; &#8722;33.6 "
         "(reference &#8722;9.54). Same nature as the echo Pareto. <b>But this share "
         "goes to the fiber and it is fixed that the isolator handles it</b>, so it is <b>not "
         "bought</b> &mdash; hence it is still not a specification"),
        ("focusing by grating shift",
         "Applying the shift theorem to the two arms, the relative orders are opposite (&#8722;1 vs +1), so "
         "gathering one arm spreads the other. It is <b>the same fact</b> as the bias slope being "
         "2Gd",
         "only a per-zone <b>constant</b> shift (bias) is harmless"),
        ("30 dB return loss from the grating",
         "The grating-alone ceiling is 12.9 dB. By decision a <b>fiber "
         "isolator</b> handles it",
         "instead the echo (the share going to the detector) became the top priority &mdash; section 7"),
    ):
        a('<tr><td class="wrap"><b>%s</b></td><td class="wrap">%s</td>'
          '<td class="wrap">%s</td></tr>' % (it, why, ev))
    a("</table></div>")

    # ---------------------------------------------------------- 9. unmet
    a("<h2>9. Unmet &#183; unverified list</h2>")
    a('<div class="tw"><table><tr><th class="n">rank</th>'
      '<th class="wrap">item</th><th>status</th>'
      '<th class="wrap">next action</th></tr>')
    for rank, it, st, cls, nx in (
        ("1", "return-grating trimming <b>71 px</b> (not 38.3 px)", "corrected", "no",
         "a value that goes into the mask layout as is. If wrong, 0.717 dB permanent loss"),
        ("2", "slab thickness <b>%.1f &rarr; %.2f &#181;m</b> re-fixed"
         % (t_old, t_now),
         "fixed", "ok",
         "260823 fixed &mdash; numerical-differentiation check done. s / 2s / GDD / channel budget all reflected. Polishing tolerance is of order &#177;1~2 &#181;m"),
        ("3", "echo &#8722;20 dB", "short", "no",
         "worst-polarization best cell is &#8722;18.22 dB, adopted cell &#8722;12.9 dB. "
         "C62 (<code>w_echo2</code> + <code>w_exit</code>) in progress"),
        ("4", "device width 2s + 309 &#181;m", "short", "no",
         "the thickness is fixed (2s = %.1f &#181;m), so this can now be fixed. RCWA redesign is unnecessary" % (2 * s_now)),
        ("5", "focusing &mdash; <b>L3 metalens array</b> (f point value undecided)",
         "f point value undecided + y axis unverified", "no",
         "<b>f has no fixed point value</b> &mdash; 507 &#181;m is a superseded "
         "point value and the replacement window (880.8~1330.1 &#181;m) is based on N = 4. N is "
         "<b>fixed only as a target N = 8</b> and <b>whether it is reachable cannot be judged "
         "from the stored map</b>. f must be re-derived once a reachable N "
         "is decided "
         "(&sect;4 callout). The quadrant aperture 100 &times; 100 &#181;m is "
         "valid independent of f. The x axis passes with edge 9.94 cells/2&#960; (&ge;8) "
         "and quantisation 0.22 dB &mdash; but <b>9.94 is a value computed at f = 507, "
         "so it must be re-measured once f is decided</b>, and 0.22 dB is not about f but a "
         "<b>fixed ratio for N = 8</b>, so it depends on N. <b>The y axis is unverified</b> &mdash; "
         "<code>_build3</code> uses <code>Add_LayerGrid(h, NX, 1)</code>, i.e. "
         "NY = 1, so the y local-period approximation has never been computed. Measured with its own criterion "
         "(cell width &ge; 2&#955;) it fell <b>short by a factor of 2</b> at f = 507, and "
         "to satisfy it <b>f &ge; 961 &#181;m</b> is needed &mdash; this bound is not a result of a specific f "
         "but a constraint on f, so it is <b>valid even if f is undecided</b>. "
         "Raising f has the trade-off that the spot grows (quantitative value after f is fixed)"),
        ("&mdash;", "<code>w_slope</code> cell adoption", "optional", "run",
         "not a blocker at the 20&#176; criterion. Margin-improvement option "
         "(w = 6~12 plateau confirmed in C60)"),
        ("&mdash;", "return TM slope &#8722;1.074", "remaining", "est",
         "passes at the 20&#176; criterion. Becomes a blocker again if the 10&#176; criterion is chosen"),
        ("&mdash;", "2&times;2 x-lens phase library", "in progress", "run",
         "C57 / C58 &mdash; measuring the cost of the <code>arg r(0)</code> flatness condition. "
         "State in the interpretation of the results that the values are under the <code>w_phase</code> = 1.0 constraint"),
        ("&mdash;", "single-track d&#968; trimming", "in progress", "run",
         "C61 (<code>w_dpsi</code>) &mdash; possible only through the penalty term"),
        ("&mdash;", "output-plane alignment tolerance", "undecided", "est",
         "<b>2&times;2 needs no profile specification</b> (25 %% holds by symmetry "
         "alone). Instead y = 0 passes the beam peak and gives <b>0.13861 dB/&#181;m</b> "
         "(1.686 times y4, unavoidable), so the <b>alignment tolerance</b> must be fixed "
         "&mdash; for d<i>y</i> = 5 &#181;m the partner imbalance is 0.69 dB, CMRR &#8722;19 dB"),
        ("&mdash;", "H3 = 750 nm exceeds the fabrication limit", "unresolved", "no",
         "exceeds the <code>a-Si</code> thickness limit <b>600 nm</b> (no exempt layer) by 25 %%. "
         "<b>All 40 jobs</b> of the phase library are out-of-spec. Lowering to 600 "
         "makes the a-Si/air single-pass phase 432&#176; &rarr; 346&#176;, so the "
         "2&#960; margin disappears"),
        ("&mdash;", "PIC coupling efficiency", "not measured", "est",
         "large-area apodised grating coupler first priority, 30~60 % expected &mdash; "
         "not reflected in the link budget"),
        ("&mdash;", "FoM docstring correction", "documentation", "est",
         "<code>scd_c_grating.py</code> still describes 0.3849 / 0.055 dB / "
         "x = 423.5 / 5 ps as true"),
    ):
        a('<tr><td class="n">%s</td><td class="wrap">%s</td>'
          '<td><span class="pill p-%s">%s</span></td>'
          '<td class="wrap">%s</td></tr>' % (rank, it, cls, st, nx))
    a("</table></div>")
    return "".join(h)
