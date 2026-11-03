# -*- coding: utf-8 -*-
"""fig7a_no_pd.py (261018) -- scope: "Fig 7(a) table and drawing need not treat the PD -- we do not deal with the PD".
1. table 'f7v panel a table': delete row 10 (Photodiode: 86.75 % / 100 % / -0.617 dB);
2. path: hide the 'Photodiode' block ('f7v block 06'); the metalens output arrow with 42.63 % stays as the output;
3. excess-loss bar (group 'f7v panel a excess loss', ungrouped / regrouped): implementation 1.680 -> 1.063 dB (input 0.004 + DLI
   section 0.230 + metalens 0.829), total 4.320 -> 3.703 dB, same scale (33.67 pt / dB) and top; blue bar, bracket and the two
   labels follow. Record: fig7a_no_pd_record.json (old table row texts, old geometry)."""
import json, time

assert pres.Name == "261023_Figure.pptx"
sl = pres.Slides(7)
rec = dict(when=time.strftime("%H:%M:%S"))
# 1
tb = sl.Shapes("f7v panel a table")
t = tb.Table
assert t.Rows.Count == 10 and t.Cell(10, 7).Shape.TextFrame.TextRange.Text.strip() == u"\u22120.617 dB"
rec["row10"] = [t.Cell(10, j).Shape.TextFrame.TextRange.Text for j in range(1, 8)]
rec["table_old"] = (tb.Left, tb.Top, tb.Width, tb.Height)
t.Rows(10).Delete()
rec["table_new"] = (tb.Left, tb.Top, tb.Width, tb.Height)
# 2
pd = sl.Shapes("f7v block 06")
pd.Visible = 0
rec["pd_block_hidden"] = pd.Name
# 3
g = sl.Shapes("f7v panel a excess loss")
gname = g.Name
r = g.Ungroup()
kids = [r.Item(i).Name for i in range(1, r.Count + 1)]
b1, b2 = sl.Shapes("f7v bar 01"), sl.Shapes("f7v bar 02")
K = b1.Height / 2.640
rec["scale_pt_per_dB"] = K
rec["old"] = {n: (sl.Shapes(n).Left, sl.Shapes(n).Top, sl.Shapes(n).Width, sl.Shapes(n).Height) for n in kids}
b2.Height = 1.063 * K
br = sl.Shapes("f7v bracket 01")
br.LockAspectRatio = 0
br.Height = 3.703 * K
l1 = sl.Shapes("f7v losstext 01")
l1.TextFrame.TextRange.Replace("4.320 dB", "3.703 dB")
l1.Top = b1.Top + 3.703 * K / 2.0 - l1.Height / 2.0
l3 = sl.Shapes("f7v losstext 03")
l3.TextFrame.TextRange.Replace("1.680 dB", "1.063 dB")
l3.Top = b2.Top + b2.Height / 2.0 - l3.Height / 2.0
ng = sl.Shapes.Range(kids).Group()
ng.Name = gname
json.dump(rec, open(r"<WORK_ROOT>\fig7_layout_261018\fig7a_no_pd_record.json", "w", encoding="utf-8"), indent=1)
result = dict(table=[round(v, 2) for v in rec["table_new"]], K=round(K, 3), bar2=(round(b2.Top, 2), round(b2.Height, 2)),
              bracket=round(br.Height, 2), l1=l1.TextFrame.TextRange.Text, l3=l3.TextFrame.TextRange.Text.replace("\r", "/"))
