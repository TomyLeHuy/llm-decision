"""Build the ECIS 2027 submission from the official template and manuscript.md (run paper/build.sh, not this file directly).
Keeps the template's styles, headers, footers, section properties and the mandatory title-page author box.
"""
import re, os, shutil, copy, sys
from lxml import etree
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
import zipfile
TEMPLATE = os.path.join(HERE, "template", "Submission-Template_ECIS-2027.docx")
TPL = os.path.join(HERE, "build", "tpl"); FIGDIR = HERE      # figure paths in manuscript.md are relative to paper/
shutil.rmtree(TPL, ignore_errors=True); os.makedirs(TPL); zipfile.ZipFile(TEMPLATE).extractall(TPL)   # fresh template on every build
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "r": R}
def w(tag): return "{%s}%s" % (W, tag)

TRACK_TITLE = "Exemplars and advancement of design research in information systems"
TEXT_WIDTH_EMU = int(16.0 * 360000)   # 16 cm text width (A4, margins 3/2 cm)
MAX_FIG_WIDTH_EMU = int(15.5 * 360000)

# ---------------- manuscript parsing ----------------
src = open(os.path.join(HERE, "manuscript.md"), encoding="utf-8").read()
meta = {k: re.search(r"^%s:\s*(.+)$" % k, src, re.M).group(1).strip() for k in ["CATEGORY", "SHORT TITLE", "TITLE", "KEYWORDS"]}
abstract = re.search(r"ABSTRACT:\n(.*?)\n\nKEYWORDS", src, re.S).group(1).strip()
body_src = src[src.index("# Introduction"):]

# italicise journal names + volumes in references (APA): ". Journal, 12(3)" and ". Journal, 127, Article"
def apa_italics(line):
    line = re.sub(r"\. (In )?(Proceedings of [^(]+?) \(", lambda m: ". " + (m.group(1) or "") + "*" + m.group(2) + "* (", line)
    line = re.sub(r"\. ([A-Z][^.*]*?), (\d+)\(", r". *\1, \2*(", line)
    line = re.sub(r"\. ([A-Z][^.*]*?), (\d+), (Article|\d)", r". *\1, \2*, \3", line)
    return line

# ---------------- XML helpers ----------------
def el(tag, **attrs):
    e = etree.Element(w(tag))
    for k, v in attrs.items(): e.set(w(k), str(v))
    return e
def run(text, italic=False, bold=False, sub=False, size=None, tab=False):
    r = el("r"); rpr = el("rPr")
    if bold: rpr.append(el("b")); rpr.append(el("bCs"))
    if italic: rpr.append(el("i")); rpr.append(el("iCs"))
    if sub: rpr.append(el("vertAlign", val="subscript"))
    if size: rpr.append(el("sz", val=size)); rpr.append(el("szCs", val=size))
    if len(rpr): r.append(rpr)
    if tab: r.append(el("tab"))
    t = el("t"); t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t)
    return r
def runs_from_markup(text, bold=False, size=None):
    """*italic* markup and X_{sub} subscripts -> runs"""
    out = []
    for seg in re.split(r"(\*[^*]+\*)", text):
        if not seg: continue
        italic = seg.startswith("*") and seg.endswith("*")
        seg = seg[1:-1] if italic else seg
        for piece in re.split(r"(_\{[^}]+\})", seg):
            if not piece: continue
            if piece.startswith("_{"): out.append(run(piece[2:-1], italic=True, sub=True, size=size))
            else: out.append(run(piece, italic=italic, bold=bold, size=size))
    return out
def para(style, text=None, runs=None, jc=None, keep_next=False):
    p = el("p"); ppr = el("pPr"); ppr.append(el("pStyle", val=style))
    if keep_next: ppr.append(el("keepNext"))
    if jc: ppr.append(el("jc", val=jc))
    p.append(ppr)
    for r_ in (runs if runs is not None else (runs_from_markup(text) if text else [])): p.append(r_)
    return p

# ---------------- images ----------------
rels_path = os.path.join(TPL, "word", "_rels", "document.xml.rels")
rels = etree.parse(rels_path); rels_root = rels.getroot()
next_rid = max(int(e.get("Id")[3:]) for e in rels_root) + 1
img_counter = [10]; docpr_id = [100]
def add_image(fname):
    global next_rid
    src_path = os.path.join(FIGDIR, fname); img_counter[0] += 1
    target = "media/image%d.png" % img_counter[0]
    shutil.copy(src_path, os.path.join(TPL, "word", target))
    rid = "rId%d" % next_rid; next_rid += 1
    rel = etree.SubElement(rels_root, "{%s}Relationship" % PR); rel.set("Id", rid); rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"); rel.set("Target", target)
    im = Image.open(src_path); wpx, hpx = im.size; dpi = im.info.get("dpi", (300, 300))[0] or 300
    width_emu = min(MAX_FIG_WIDTH_EMU, int(wpx / dpi * 914400)); height_emu = int(width_emu * hpx / wpx)
    return rid, width_emu, height_emu
def figure_para(fname):
    rid, cx, cy = add_image(fname); docpr_id[0] += 1
    xml = f'''<w:p xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}"><w:pPr><w:pStyle w:val="Figure"/><w:keepNext/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{docpr_id[0]}" name="Picture {docpr_id[0]}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="Picture {docpr_id[0]}"/><pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
    return etree.fromstring(xml)
def caption_para(label, text):
    p = para("Caption", runs=[run(label), run(text, tab=True)]); p[0].insert(1, el("keepLines")); return p

# ---------------- tables ----------------
def table(rows, widths_twips, split=False):
    tbl = el("tbl"); tblpr = el("tblPr")
    tblw = el("tblW", w=str(sum(widths_twips)), type="dxa"); tblpr.append(tblw)
    tblpr.append(el("jc", val="center"))
    borders = el("tblBorders")
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = el(side, val="single", sz="4", space="0", color="auto"); borders.append(b)
    tblpr.append(borders)
    cellmar = el("tblCellMar")
    for side, v in [("left", 57), ("right", 57)]: cellmar.append(el(side, w=str(v), type="dxa"))
    tblpr.append(cellmar)
    tblpr.append(el("tblLook", val="01E0", firstRow="1", lastRow="0", firstColumn="0", lastColumn="0", noHBand="0", noVBand="0"))
    tbl.append(tblpr)
    grid = el("tblGrid")
    for wd in widths_twips: grid.append(el("gridCol", w=str(wd)))
    tbl.append(grid)
    for i, row in enumerate(rows):
        tr = el("tr"); trpr = el("trPr"); trpr.append(el("cantSplit"))
        if i == 0: trpr.append(el("tblHeader"))
        tr.append(trpr)
        for j, cell in enumerate(row):
            tc = el("tc"); tcpr = el("tcPr"); tcpr.append(el("tcW", w=str(widths_twips[j]), type="dxa")); tc.append(tcpr)
            p = el("p"); ppr = el("pPr"); ppr.append(el("pStyle", val="TableText")); ppr.append(el("spacing", before="20", after="20"))
            if (i == 0) if split else (i < len(rows) - 1): ppr.insert(1, el("keepNext"))
            if j > 0 and i > 0 and re.match(r"^[+\-−\d.%]", cell.strip()): ppr.append(el("jc", val="center"))
            if i == 0: ppr.append(el("jc", val="center"))
            p.append(ppr)
            for r_ in runs_from_markup(cell.strip(), bold=(i == 0)): p.append(r_)
            tc.append(p); tr.append(tc)
        tbl.append(tr)
    return tbl
def col_widths(rows, total=9000):
    ncol = len(rows[0]); lens = [max(len(r[j]) for r in rows) for j in range(ncol)]
    lens = [min(max(l, 6), 60) for l in lens]; s = sum(lens)
    widths = [int(total * l / s) for l in lens]; widths[-1] += total - sum(widths); return widths

# ---------------- document assembly ----------------
doc_path = os.path.join(TPL, "word", "document.xml")
doc = etree.parse(doc_path); body = doc.getroot().find(w("body"))
children = list(body)
sectPr = body.find(w("sectPr"))
def pstyle(e):
    ps = e.find("./w:pPr/w:pStyle", NS); return ps.get(w("val")) if ps is not None else None
# keep: title (0), type of paper (1), empty type (2), the author-box paragraph (first Importantremark with drawing)
title_p = children[0]; type_p = children[1]; empty_p = children[2]
box_p = next(c for c in children if pstyle(c) == "Importantremark" and c.find(".//w:drawing", NS) is not None)
# rewrite title and category text: keep pPr, replace runs
def set_text(p, text, keep_rpr_from_first=True):
    first_rpr = None
    for r_ in p.findall("w:r", NS):
        if first_rpr is None: first_rpr = r_.find("w:rPr", NS)
        p.remove(r_)
    for hl in p.findall("w:hyperlink", NS): p.remove(hl)
    r_ = run(text); p.append(r_)
set_text(title_p, meta["TITLE"]); set_text(type_p, meta["CATEGORY"])
for c in children: body.remove(c)
for c in [title_p, type_p, empty_p, box_p]: body.append(c)

# abstract + keywords
body.append(para("Subtitle", "Abstract"))
body.append(para("Abstract", abstract))
body.append(para("Basictext", ""))
kw = para("Abstract", runs=[run("Keywords: ", italic=True), run(meta["KEYWORDS"] + ".")]); body.append(kw)

fig_n = 0; tab_n = 0
lines = body_src.split("\n"); i = 0; in_refs = False
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln: i += 1; continue
    if ln.startswith("# References"):
        in_refs = True; body.append(para("Subtitle", "References")); i += 1; continue
    if in_refs:
        body.append(para("Reference", apa_italics(ln))); i += 1; continue
    if ln.startswith("## "): body.append(para("Heading2", ln[3:])); i += 1; continue
    if ln.startswith("# "): body.append(para("Heading1", ln[2:])); i += 1; continue
    if ln.startswith("FIGURE:"):
        fname, cap = [s.strip() for s in ln[len("FIGURE:"):].split("|", 1)]; fig_n += 1
        body.append(figure_para(fname)); body.append(caption_para("Figure %d." % fig_n, cap)); i += 1; continue
    if ln.startswith("EQUATION:"):
        eq = ln[len("EQUATION:"):].strip()
        p = para("Basictext", runs=runs_from_markup(eq), jc="center"); body.append(p); i += 1; continue
    if ln.startswith("TABLE:"):
        rows = []; i += 1; widths = None; split = False
        if lines[i].startswith("WIDTHS:"):
            widths = [int(x) for x in lines[i][len("WIDTHS:"):].split(",")]; i += 1
        split = False
        if lines[i].startswith("SPLIT:"): split = True; i += 1
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c for c in lines[i].strip().strip("|").split("|")]); i += 1
        cap = lines[i][len("CAPTION:"):].strip() if lines[i].startswith("CAPTION:") else ""; i += 1
        tab_n += 1; body.append(table(rows, [int(9000 * x / sum(widths)) for x in widths] if widths else col_widths(rows), split=split)); body.append(caption_para("Table %d." % tab_n, cap)); continue
    body.append(para("Basictext", ln)); i += 1
body.append(sectPr)
doc.write(doc_path, xml_declaration=True, encoding="UTF-8", standalone=True)
rels.write(rels_path, xml_declaration=True, encoding="UTF-8", standalone=True)

# headers: short title (default) and track title (even)
for fname, text in [("header2.xml", meta["SHORT TITLE"]), ("header1.xml", TRACK_TITLE)]:
    hp = os.path.join(TPL, "word", fname); h = etree.parse(hp); ts = h.getroot().findall(".//w:t", NS)
    for k, t in enumerate(ts): t.text = text if k == 0 else ""
    h.write(hp, xml_declaration=True, encoding="UTF-8", standalone=True)
# footnotes: drop the template's example footnote (id 2)
fp = os.path.join(TPL, "word", "footnotes.xml"); f = etree.parse(fp)
for fn in f.getroot().findall("w:footnote", NS):
    if fn.get(w("id")) == "2": f.getroot().remove(fn)
f.write(fp, xml_declaration=True, encoding="UTF-8", standalone=True)
# metadata: blind
cp = os.path.join(TPL, "docProps", "core.xml"); c = open(cp, encoding="utf-8").read()
c = re.sub(r"<dc:title>.*?</dc:title>", "<dc:title></dc:title>", c); c = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", c)
c = re.sub(r"<dc:description>.*?</dc:description>", "<dc:description></dc:description>", c); open(cp, "w", encoding="utf-8").write(c)
print("figures:", fig_n, "tables:", tab_n)
