"""Draw figure.svg from results/results.json. Standard library only.

Presentation only: it reads numbers the analysis already wrote and computes
nothing new. Run `python ecg_age_bias.py` first.
"""
import html
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "results" / "results.json"
OUT = ROOT / "figure.svg"

if not SRC.exists():
    sys.exit("results/results.json not found: run `python ecg_age_bias.py` first")

res = json.loads(SRC.read_text())
W, H = 720, 420
X0, Y0, Y1 = 70, 70, 330          # plot box: left edge, top, baseline
YMAX = 6.0                         # percent axis top
BAR = 56
COLOURS = {"older": "#c2410c", "younger": "#1d4ed8"}
LABEL = {"older": "older-looking (gap > +8 y)", "younger": "younger-looking (gap < -8 y)"}


def y(pct):
    return Y1 - (Y1 - Y0) * pct / YMAX


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    'font-family="system-ui, sans-serif" font-size="13">',
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    '<text x="70" y="28" font-size="16" font-weight="600">Crude death rate by AI ECG age gap group, '
    'before and after correcting for age</text>',
    f'<text x="70" y="48" fill="#444">CODE-15%, n = {res["cohort"]["n_patients"]:,} patients. '
    'Whiskers are 95% intervals.</text>',
]
for tick in range(0, 7):
    parts.append(f'<line x1="{X0}" x2="{W - 30}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#e5e5e5"/>')
    parts.append(f'<text x="{X0 - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#444">{tick}%</text>')

panels = [("raw", "Raw gap", 230), ("corrected", "Corrected for age", 520)]
for key, title, cx in panels:
    parts.append(f'<text x="{cx}" y="{Y1 + 36}" text-anchor="middle" font-weight="600">{title}</text>')
    for i, grp in enumerate(("older", "younger")):
        g = res[key][grp]
        x = cx - BAR - 6 + i * (BAR + 12)
        lo, hi = g["ci95_pct"]
        parts.append(
            f'<rect x="{x}" y="{y(g["death_rate_pct"]):.1f}" width="{BAR}" '
            f'height="{Y1 - y(g["death_rate_pct"]):.1f}" fill="{COLOURS[grp]}"/>'
        )
        parts.append(f'<line x1="{x + BAR / 2}" x2="{x + BAR / 2}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="#111"/>')
        parts.append(
            f'<text x="{x + BAR / 2}" y="{y(hi) - 6:.1f}" text-anchor="middle" font-weight="600">'
            f'{g["death_rate_pct"]:.2f}%</text>'
        )
        parts.append(
            f'<text x="{x + BAR / 2}" y="{Y1 + 16}" text-anchor="middle" fill="#444" font-size="11">'
            f'age {g["mean_age"]:.1f}</text>'
        )
parts.append(f'<line x1="{X0}" x2="{W - 30}" y1="{Y1}" y2="{Y1}" stroke="#111"/>')
for i, grp in enumerate(("older", "younger")):
    lx = 70 + i * 260
    parts.append(f'<rect x="{lx}" y="{H - 22}" width="12" height="12" fill="{COLOURS[grp]}"/>')
    parts.append(f'<text x="{lx + 18}" y="{H - 12}">{html.escape(LABEL[grp])}</text>')
parts.append(
    f'<text x="{W - 30}" y="{Y1 + 52}" text-anchor="end" fill="#444" font-size="11">'
    'age under each bar = group mean chronological age</text>'
)
parts.append("</svg>")
OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT.name}")
