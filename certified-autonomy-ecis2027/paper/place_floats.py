"""Greedy placement of figures and tables: for each float in document order, try moving it 0..3 text blocks later
(never past another float or the references), render with LibreOffice, keep the placement that minimises
total blank space at the bottom of non-final pages (ties: fewer pages, then smaller shift)."""
import os, re, subprocess, sys, shutil, glob
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
V = "/root/.claude/skills/synced/d396c4ca-838a-4fb9-9acf-f22550fd1804_6034123d-bb7d-4bfd-8f72-7da68971c93b/docx/scripts/office"
SRC = open("manuscript.md").read()
head, refs = SRC.split("\n# References", 1)
blocks = [b for b in head.split("\n\n") if b.strip()]
is_float = lambda b: b.startswith("FIGURE:") or b.startswith("TABLE:")
def measure(blks):
    open("manuscript.md", "w").write("\n\n".join(blks) + "\n\n# References" + refs)
    subprocess.run(["python3", "build_docx.py"], capture_output=True)
    subprocess.run(["python3", "../pack_docx.py", "tpl", "ecis2027_certified_autonomy.docx"], capture_output=True)
    shutil.rmtree("render_opt", ignore_errors=True); os.makedirs("render_opt")
    subprocess.run(["python3", f"{V}/soffice.py", "--headless", "--convert-to", "pdf", "--outdir", "render_opt", "ecis2027_certified_autonomy.docx"], capture_output=True)
    subprocess.run(["pdftoppm", "-gray", "-r", "30", "render_opt/ecis2027_certified_autonomy.pdf", "render_opt/g"], capture_output=True)
    gaps = []
    for f in sorted(glob.glob("render_opt/g-*.pgm")):
        a = np.array(Image.open(f)); h = a.shape[0]; body = a[int(h * 0.06):int(h * 0.90)]
        rows = np.where((body < 200).any(1))[0]; last = rows.max() if len(rows) else 0
        gaps.append(1 - (last + 1) / body.shape[0])
    return len(gaps), sum(gaps[:-1]), gaps
def shifted(blks, i, s):
    b = list(blks); j = i
    if s < 0:
        for _ in range(-s):
            if j - 1 < 0 or is_float(b[j - 1]) or b[j - 1].startswith("#") or b[j - 1].startswith("ABSTRACT") or b[j - 1].startswith("KEYWORDS"): return None
            b[j], b[j - 1] = b[j - 1], b[j]; j -= 1
        return b
    for _ in range(s):
        if j + 1 >= len(b) or is_float(b[j + 1]): return None
        b[j], b[j + 1] = b[j + 1], b[j]; j += 1
    return b
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cur = blocks; best = measure(cur); print("start", best[0], round(best[1], 3), flush=True)
floats = [i for i, b in enumerate(cur) if is_float(b)]
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(floats)
for k in range(start, end):
    i = [j for j, b in enumerate(cur) if is_float(b)][k]
    label = cur[i].split("\n")[0][:40]; choice = (best[0], best[1], 0, cur)
    for s in [-2, -1, 1, 2, 3]:
        nb = shifted(cur, i, s)
        if nb is None: continue
        r = measure(nb); print(f"  float {k} {label!r} shift {s}: pages {r[0]} interior gap {r[1]:.3f}", flush=True)
        if (r[0], round(r[1], 3), abs(s)) < (choice[0], round(choice[1], 3), abs(choice[2])): choice = (r[0], r[1], s, nb)
    cur = choice[3]; best = (choice[0], choice[1]); print(f"float {k} {label!r}: keep shift {choice[2]} -> pages {best[0]} gap {best[1]:.3f}", flush=True)
final = measure(cur); print("final", final[0], round(final[1], 3), [round(g, 2) for g in final[2]])
