#!/usr/bin/env python
"""Build Ramoy_Midterm_Defense.pptx from MIDTERM_DEFENSE_DECK.md.

One slide per `## Slide N` heading. Title / Subtitle / Body / Visual become
slide content; Speaker notes go to the notes pane only.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = "/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL"
SRC = os.path.join(ROOT, "MIDTERM_DEFENSE_DECK.md")
OUT = os.path.join(ROOT, "Ramoy_Midterm_Defense.pptx")
FORMULA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formulas")

# ---------------------------------------------------------------- design tokens
INK = RGBColor(0x14, 0x21, 0x3D)        # titles, primary text
BODY = RGBColor(0x2A, 0x33, 0x45)
ACCENT = RGBColor(0x1D, 0x5B, 0x8F)     # "our model", rules, chips
ACCENT_BG = RGBColor(0xE9, 0xF0, 0xF7)
NEUTRAL = RGBColor(0x5A, 0x64, 0x72)    # "baseline", captions
RULE = RGBColor(0xC9, 0xD2, 0xDC)
ZEBRA = RGBColor(0xF4, 0xF7, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RESULT_RED = RGBColor(0xA8, 0x35, 0x2A)

TITLE_FONT = "Georgia"
TEXT_FONT = "Calibri"

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN_L, MARGIN_R = 0.72, 0.72
CONTENT_W = SLIDE_W - MARGIN_L - MARGIN_R
CONTENT_TOP = 1.80
CONTENT_BOTTOM = 6.92

TAG_STYLE = {
    "[RESULT]": RESULT_RED,
    "[KEY CONCEPT]": ACCENT,
    "[THE CORE CLAIM]": RGBColor(0x7A, 0x2F, 0x6B),
}

SEP_RE = r"^\|[\s:|]*-[\s:\-|]*\|$"
# Figures to place on a slide, keyed by slide number. These come from
# deck_figures/ (extracted from explanation.ipynb, or generated for the deck);
# the Visual: line still supplies any file it names under preliminary_results/.
SLIDE_FIGURES = {
    "4":  ["deck_figures/S04a_ewg_price.png", "deck_figures/S04b_ewg_returns.png"],
    "5":  ["deck_figures/S05_two_investments.png"],
    "6":  ["deck_figures/S06_S07_four_scenarios.png"],
    "7":  ["deck_figures/S06_S07_four_scenarios.png"],
    "8":  ["deck_figures/S08_rolling_window.png"],
    "10": ["deck_figures/S10_direction_vs_size.png"],
    "11": ["deck_figures/S11_forecasting_task.png"],
    "12": ["deck_figures/S12a_covid_all_countries.png"],
    "13": ["deck_figures/S13_transfer_entropy_idea.png"],
    "14": ["deck_figures/S14_gnn_architecture.png"],
    "15": ["deck_figures/S15_alpha_dial.png"],
    "16": ["deck_figures/S16_hurst_regime_detector.png"],
    "21b": ["deck_figures/S21_h1b_walkthrough.png"],
    "28": ["deck_figures/S28_topology_alpha_contrast.png"],
}


FIELD_RE = re.compile(r"^\*\*(Title|Subtitle|Body|Visual|Speaker notes)\:\*\*\s*", re.M)


# ---------------------------------------------------------------- markdown model
@dataclass
class Slide:
    num: str
    heading: str
    act: str
    tag: str | None = None
    title: str = ""
    subtitle: str = ""
    body: str = ""
    visual: str = ""
    notes: str = ""


@dataclass
class Block:
    kind: str                      # para | bullet | number | quote | table | formula | image | caption
    text: str = ""
    level: int = 0
    rows: list = field(default_factory=list)
    paths: list = field(default_factory=list)


def parse_markdown(path: str) -> list[Slide]:
    raw = open(path, encoding="utf-8").read()
    # Everything before the first ACT heading is generator instructions, not slides.
    lines = raw.split("\n")
    slides: list[Slide] = []
    act = ""
    cur_head: str | None = None
    cur_act = ""
    buf: list[str] = []

    def flush():
        nonlocal_unused = None
        if cur_head is None:
            return
        m = re.match(r"##\s+Slide\s+(\S+)\s*(?:[—-]\s*(.*))?$", cur_head)
        num = m.group(1) if m else "?"
        heading = (m.group(2) or "").strip() if m else cur_head
        tag = None
        for t in TAG_STYLE:
            if t in heading:
                tag = t
                heading = heading.replace(t, "").strip().rstrip("`").strip()
        heading = heading.replace("`", "").strip()
        s = Slide(num=num, heading=heading, act=cur_act, tag=tag)
        text = "\n".join(buf).strip()
        # split on the five recognised field markers, keep order-independence
        parts = FIELD_RE.split(text)
        if parts and not FIELD_RE.match(text):
            parts = parts[1:] if len(parts) > 1 else parts
        it = iter(FIELD_RE.findall(text))
        chunks = re.split(FIELD_RE, text)
        # chunks = [pre, name1, body1, name2, body2, ...]
        for i in range(1, len(chunks) - 1, 2):
            name, val = chunks[i], chunks[i + 1].strip()
            val = re.sub(r"\n---\s*$", "", val).strip()
            if name == "Title":
                s.title = val
            elif name == "Subtitle":
                s.subtitle = val
            elif name == "Body":
                s.body = val
            elif name == "Visual":
                s.visual = val
            elif name == "Speaker notes":
                s.notes = val
        if not s.title:
            s.title = s.heading
        slides.append(s)

    for ln in lines:
        if ln.startswith("# ") and not ln.startswith("## "):
            act = ln[2:].strip()
            act = re.sub(r"\s*\(.*\)$", "", act).replace(" Slides", "")
            continue
        if ln.startswith("## Slide"):
            flush()
            cur_head, buf, cur_act = ln, [], act
            continue
        if cur_head is not None:
            buf.append(ln)
    flush()
    return slides


def split_blocks(md: str) -> list[Block]:
    """Turn a Body/Visual markdown chunk into ordered renderable blocks."""
    blocks: list[Block] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s == "---":
            i += 1
            continue
        # display formula
        if s.startswith("$$"):
            tex = s
            while not tex.rstrip().endswith("$$") or len(tex.strip()) < 4:
                i += 1
                tex += "\n" + lines[i]
            blocks.append(Block("formula", tex.strip().strip("$").strip()))
            i += 1
            continue
        # table
        if s.startswith("|") and i + 1 < len(lines) and re.match(SEP_RE, lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip()
                if not re.match(SEP_RE, r):
                    cells = [c.strip() for c in re.split(r"(?<!\\)\|", r.strip("|"))]
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", rows=rows))
            continue
        # blockquote (may span lines)
        if s.startswith(">"):
            group = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                group.append(lines[i].strip().lstrip(">").strip())
                i += 1
            chunk = []
            for q in group + [""]:
                if q:
                    chunk.append(q)
                elif chunk:
                    blocks.append(Block("quote", " ".join(chunk)))
                    chunk = []
            continue
        # bullets / numbers
        m = re.match(r"^(\s*)([-*])\s+(.*)$", ln)
        if m:
            blocks.append(Block("bullet", m.group(3).strip(), level=len(m.group(1)) // 2))
            i += 1
            continue
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", ln)
        if m:
            blocks.append(Block("number", f"{m.group(2)}.  {m.group(3).strip()}", level=len(m.group(1)) // 2))
            i += 1
            continue
        blocks.append(Block("para", s))
        i += 1
    return blocks


# ---------------------------------------------------------------- rich text runs
INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")


def add_runs(p, text, size, color=BODY, bold=False, italic=False, font=TEXT_FONT):
    text = text.replace("\\|", "|").replace("&nbsp;", " ")

    def emit(t, b, it, mono):
        if not t:
            return
        r = p.add_run()
        r.text = t
        r.font.size = Pt(size)
        r.font.bold = b
        r.font.italic = it
        r.font.name = "Consolas" if mono else font
        r.font.color.rgb = color if not b else (INK if color == BODY else color)

    def walk(t, b, it, mono):
        for piece in INLINE_RE.split(t):
            if not piece:
                continue
            if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
                walk(piece[2:-2], True, it, mono)
            elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
                walk(piece[1:-1], b, True, mono)
            elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
                walk(piece[1:-1], b, it, True)
            else:
                emit(piece, b, it, mono)

    walk(text, bold, italic, False)
    return p


def plain(text: str) -> str:
    return re.sub(r"[*`]", "", text).replace("\\|", "|")


# ---------------------------------------------------------------- formula images
def render_formula(tex: str, key: str) -> str | None:
    os.makedirs(FORMULA_DIR, exist_ok=True)
    out = os.path.join(FORMULA_DIR, f"{key}.png")
    if os.path.exists(out):
        return out
    src = tex.replace(r"\text{", r"\mathrm{").replace(r"\widehat", r"\hat")
    src = re.sub(r"\\!", "", src).replace(r"\,", r"\ ")
    src = re.sub(r"\\left|\\right", "", src)
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"tex = r'''{src}'''\n"
        "fig = plt.figure(figsize=(0.01,0.01))\n"
        "fig.text(0,0,'$'+tex+'$', fontsize=26, color='#14213D')\n"
        f"fig.savefig(r'{out}', dpi=260, transparent=True, bbox_inches='tight', pad_inches=0.06)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(f"[formula fallback] {key}: {r.stderr.strip()[-300:]}\n")
        return None
    return out


# ---------------------------------------------------------------- measurement
def est_text_h(text, size, width_in, spacing=0.10):
    chars = max(1, int(width_in / (0.47 * size / 72.0)))
    lines = max(1, math.ceil(len(plain(text)) / chars))
    return lines * (size * 1.28 / 72.0) + spacing


def col_widths(rows, width_in, fs=11.0):
    ncol = max(len(r) for r in rows)
    norm = [r + [""] * (ncol - len(r)) for r in rows]
    char_w = 0.47 * fs / 72.0
    weights, mins = [], []
    for c in range(ncol):
        col = [plain(r[c]) for r in norm]
        longest_word = max((max((len(w) for w in cell.split()), default=1) for cell in col), default=1)
        mins.append(min(width_in * 0.46, longest_word * char_w * 1.18 + 0.22))
        head = len(col[0]) * 1.30
        cells = [len(x) for x in col[1:]] or [1]
        weights.append(max(head, sum(cells) / len(cells) * 1.15, max(cells) * 0.55, 6))
    if sum(mins) > width_in:
        k = width_in / sum(mins)
        return [m * k for m in mins], norm
    slack = width_in - sum(mins)
    tot = sum(weights)
    extra_w = [max(0.0, width_in * w / tot - m) for w, m in zip(weights, mins)]
    etot = sum(extra_w) or 1.0
    return [m + slack * e / etot for m, e in zip(mins, extra_w)], norm


def est_table_h(rows, size, width_in):
    if not any(plain(c).strip() for c in rows[0]):
        rows = rows[1:]
    widths, norm = col_widths(rows, width_in, size)
    h = 0.0
    for ri, r in enumerate(norm):
        fs = size if ri else size + 0.5
        lines = 1
        for c, cell in enumerate(r):
            chars = max(1, int((widths[c] - 0.16) / (0.47 * fs / 72.0)))
            lines = max(lines, math.ceil(max(1, len(plain(cell))) / chars))
        h += lines * (fs * 1.26 / 72.0) + 0.12
    return h + 0.10


def est_block_h(b: Block, scale: float, width_in: float) -> float:
    fs = 15.5 * scale
    if b.kind == "table":
        return est_table_h(b.rows, 11.0 * scale, width_in)
    if b.kind == "formula":
        return 0.78 * scale + 0.18
    if b.kind == "image":
        return 0.0  # handled separately
    if b.kind == "quote":
        return est_text_h(b.text, fs + 0.5, width_in - 0.55, 0.30) + 0.24
    if b.kind == "caption":
        return est_text_h(b.text, 11.5 * scale, width_in, 0.16)
    if b.kind in ("bullet", "number"):
        return est_text_h(b.text, fs, width_in - 0.30 - 0.22 * b.level, 0.13)
    return est_text_h(b.text, fs, width_in, 0.16)



def fit_scale(blocks, width, avail, reserve=0.0, lo=0.52):
    scale = 1.0
    while scale > lo:
        need = sum(est_block_h(b, scale, width) for b in blocks)
        need += 0.14 * sum(1 for b in blocks if b.kind == "table")
        if need + reserve <= avail:
            return scale
        scale -= 0.02
    return lo


def best_two_column(blocks, col_w, avail, img_reserve):
    """Balance blocks across two columns; return (split, scale) or None."""
    best = None
    for k in range(1, len(blocks)):
        left, right = blocks[:k], blocks[k:]
        if not left or not right:
            continue
        s = 1.0
        while s > 0.62:
            hl = sum(est_block_h(b, s, col_w) for b in left) + 0.14 * sum(1 for b in left if b.kind == "table")
            hr = sum(est_block_h(b, s, col_w) for b in right) + 0.14 * sum(1 for b in right if b.kind == "table")
            if max(hl, hr + img_reserve) <= avail:
                break
            s -= 0.02
        hl = sum(est_block_h(b, s, col_w) for b in left)
        hr = sum(est_block_h(b, s, col_w) for b in right)
        score = (round(s, 2), -abs(hl - hr))
        if best is None or score > best[0]:
            best = (score, k, s)
    return (best[1], best[2]) if best else None


# ---------------------------------------------------------------- drawing
def hanging(p, indent_in=0.24):
    """Wrapped lines of a bullet line up under its text, not under the marker."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent_in * 914400)))
    pPr.set("indent", str(-int(indent_in * 914400)))


def add_box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def rect(slide, x, y, w, h, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def draw_table(slide, rows, x, y, w, size):
    headless = not any(plain(c).strip() for c in rows[0])
    if headless:
        rows = rows[1:]
    widths, norm = col_widths(rows, w, size)
    ncol, nrow = len(widths), len(norm)
    h = est_table_h(rows, size, w)
    gt = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w), Inches(h)).table
    gt.first_row = not headless
    try:
        gt.horz_banding = False
    except Exception:
        pass
    for c, cw in enumerate(widths):
        gt.columns[c].width = Inches(cw)
    for ri, r in enumerate(norm):
        for ci in range(ncol):
            cell = gt.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0 and not headless:
                cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.fore_color.rgb = ZEBRA if (ri % 2 == (0 if headless else 1)) else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            txt = r[ci]
            if ri == 0 and not headless:
                add_runs(p, plain(txt), size + 0.5, WHITE, bold=True)
            else:
                emph = ci == 0 or txt.startswith("**")
                add_runs(p, txt, size, INK if emph else BODY, bold=(ci == 0 and "**" not in txt))
    return h


def draw_blocks(slide, blocks, x, top, w, avail, scale):
    y = top
    for b in blocks:
        fs = 15.5 * scale
        if b.kind == "table":
            h = draw_table(slide, b.rows, x, y, w, 11.0 * scale)
            y += h + 0.14
        elif b.kind == "formula":
            img = render_formula(b.text, b.key) if hasattr(b, "key") else None
            if img:
                from PIL import Image

                iw, ih = Image.open(img).size
                hh = min(0.62 * scale + 0.10, avail)
                ww = hh * iw / ih
                if ww > w * 0.92:
                    ww = w * 0.92
                    hh = ww * ih / iw
                slide.shapes.add_picture(img, Inches(x + (w - ww) / 2), Inches(y), Inches(ww), Inches(hh))
                y += hh + 0.18
            else:
                _, tf = add_box(slide, x, y, w, 0.4)
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER
                add_runs(p, b.text, fs, ACCENT, italic=True)
                y += est_text_h(b.text, fs, w) + 0.10
        elif b.kind == "quote":
            h = est_text_h(b.text, fs + 0.5, w - 0.55, 0.30)
            rect(slide, x, y, w, h + 0.20, ACCENT_BG)
            rect(slide, x, y, 0.055, h + 0.20, ACCENT)
            _, tf = add_box(slide, x + 0.26, y + 0.10, w - 0.45, h)
            p = tf.paragraphs[0]
            add_runs(p, b.text, fs + 0.5, INK, bold=False)
            y += h + 0.20 + 0.14
        elif b.kind == "caption":
            h = est_text_h(b.text, 11.5 * scale, w, 0.16)
            _, tf = add_box(slide, x, y, w, h)
            p = tf.paragraphs[0]
            add_runs(p, b.text, 11.5 * scale, NEUTRAL, italic=True)
            y += h
        elif b.kind in ("bullet", "number"):
            ind = 0.22 * b.level
            h = est_block_h(b, scale, w)
            _, tf = add_box(slide, x + ind, y, w - ind, h)
            p = tf.paragraphs[0]
            hanging(p, 0.26 if b.kind == "bullet" else 0.30)
            if b.kind == "bullet":
                r = p.add_run()
                r.text = "▪  "
                r.font.size = Pt(fs)
                r.font.color.rgb = ACCENT
                r.font.name = TEXT_FONT
            add_runs(p, b.text, fs, BODY)
            y += h
        else:
            h = est_block_h(b, scale, w)
            _, tf = add_box(slide, x, y, w, h)
            add_runs(tf.paragraphs[0], b.text, fs, BODY)
            y += h
    return y


def chrome(slide, s: Slide, total: int, idx: int):
    # title
    _, tf = add_box(slide, MARGIN_L, 0.44, CONTENT_W - 1.6, 1.08)
    tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    p = tf.paragraphs[0]
    add_runs(p, s.title, 30, INK, bold=True, font=TITLE_FONT)
    if s.subtitle:
        p2 = tf.add_paragraph()
        p2.space_before = Pt(6)
        add_runs(p2, s.subtitle, 16, NEUTRAL, font=TEXT_FONT)
    # accent rule
    rect(slide, MARGIN_L, 1.62, 1.15, 0.045, ACCENT)
    rect(slide, MARGIN_L + 1.15, 1.636, CONTENT_W - 1.15, 0.013, RULE)
    # tag chip
    if s.tag:
        label = s.tag.strip("[]")
        w = 0.20 + 0.088 * len(label)
        rect(slide, SLIDE_W - MARGIN_R - w, 0.62, w, 0.30, TAG_STYLE[s.tag])
        _, tf2 = add_box(slide, SLIDE_W - MARGIN_R - w, 0.665, w, 0.26)
        p = tf2.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_runs(p, label, 10.5, WHITE, bold=True)
    # footer
    _, tf3 = add_box(slide, MARGIN_L, 7.02, CONTENT_W, 0.28)
    p = tf3.paragraphs[0]
    add_runs(p, s.act, 9.5, NEUTRAL)
    _, tf4 = add_box(slide, SLIDE_W - MARGIN_R - 2.0, 7.02, 2.0, 0.28)
    p = tf4.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    add_runs(p, f"Slide {s.num}  ·  {idx}/{total}", 9.5, NEUTRAL)


def set_notes(slide, text):
    if not text:
        return
    tf = slide.notes_slide.notes_text_frame
    tf.text = ""
    first = True
    for para in [x for x in text.split("\n") if x.strip()]:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = plain(para).strip()
        for r in p.runs:
            r.font.size = Pt(12)


# ---------------------------------------------------------------- visual field
IMG_RE = re.compile(r"`?(?:preliminary_results/)?(F\d_[A-Za-z0-9_]+\.png)`?")


def visual_blocks(s: Slide) -> tuple[list[str], str]:
    paths, missing = [], []
    for name in IMG_RE.findall(s.visual):
        p = os.path.join(ROOT, "preliminary_results", name)
        (paths if os.path.exists(p) else missing).append(p if os.path.exists(p) else name)
    caption = re.sub(r"\s+", " ", plain(s.visual)).strip()
    return paths, caption


def place_images(slide, paths, x, y, w, avail):
    from PIL import Image

    if not paths or avail < 0.80:
        return 0.0
    gap = 0.18
    dims = [Image.open(p).size for p in paths]
    ars = [iw / ih for iw, ih in dims]

    def side_by_side():
        each = (w - gap * (len(paths) - 1)) / len(paths)
        h = min(min(each / a for a in ars), avail)
        return h, [(min(each, h * a), h) for a in ars]

    def stacked():
        slot = (avail - gap * (len(paths) - 1)) / len(paths)
        boxes = []
        for a in ars:
            hh = min(slot, w / a)
            boxes.append((hh * a, hh))
        return sum(b[1] for b in boxes) + gap * (len(paths) - 1), boxes

    h1, b1 = side_by_side()
    area1 = sum(bw * bh for bw, bh in b1)
    h2, b2 = stacked()
    area2 = sum(bw * bh for bw, bh in b2)
    vertical = len(paths) > 1 and area2 > area1 * 1.12

    if vertical:
        cy = y + max(0.0, (avail - h2) / 2)
        for p, (bw, bh) in zip(paths, b2):
            slide.shapes.add_picture(p, Inches(x + (w - bw) / 2), Inches(cy), Inches(bw), Inches(bh))
            cy += bh + gap
        return h2

    total_w = sum(bw for bw, _ in b1) + gap * (len(paths) - 1)
    cx = x + (w - total_w) / 2
    for p, (bw, bh) in zip(paths, b1):
        slide.shapes.add_picture(p, Inches(cx), Inches(y + (h1 - bh) / 2), Inches(bw), Inches(bh))
        cx += bw + gap
    return h1


# ---------------------------------------------------------------- build
def build():
    slides = parse_markdown(SRC)
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    total = len(slides)

    for idx, s in enumerate(slides, start=1):
        sl = prs.slides.add_slide(blank)
        set_notes(sl, s.notes)

        # ---- title slide gets its own treatment
        if idx == 1:
            rect(sl, 0, 0, SLIDE_W, SLIDE_H, RGBColor(0xF7, 0xFA, 0xFC))
            rect(sl, 0, 0, 0.30, SLIDE_H, ACCENT)
            _, tf = add_box(sl, 1.35, 2.05, 10.6, 1.5)
            add_runs(tf.paragraphs[0], s.title, 46, INK, bold=True, font=TITLE_FONT)
            _, tf = add_box(sl, 1.35, 3.62, 10.2, 0.8)
            add_runs(tf.paragraphs[0], s.subtitle, 19, NEUTRAL, font=TEXT_FONT)
            rect(sl, 1.35, 4.62, 1.4, 0.045, ACCENT)
            y = 4.95
            for b in split_blocks(s.body):
                h = est_text_h(b.text, 14.5, 10.2, 0.16)
                _, tf = add_box(sl, 1.35, y, 10.2, h)
                add_runs(tf.paragraphs[0], b.text, 14.5, BODY)
                y += h
            continue

        chrome(sl, s, total, idx)

        blocks = split_blocks(s.body)
        for i, b in enumerate(blocks):
            if b.kind == "formula":
                b.key = f"s{s.num}_{i}"
        paths, caption = visual_blocks(s)
        extra = [p for p in (os.path.join(ROOT, f) for f in SLIDE_FIGURES.get(s.num, []))
                 if os.path.exists(p)]
        if extra:
            paths = extra          # an explicit mapping replaces the Visual: file
        elif s.visual and not paths:
            blocks.append(Block("caption", f"Visual — {caption}"))

        avail_total = CONTENT_BOTTOM - CONTENT_TOP
        img_reserve = 0.0
        if paths:
            img_reserve = 1.20 if len(blocks) > 3 else 2.9

        wide = False
        if paths:
            from PIL import Image
            ars = [Image.open(q).size[0] / Image.open(q).size[1] for q in paths]
            wide = len(paths) == 1 and ars[0] >= 2.2

        # a figure reads better beside the text than squeezed under it
        if paths and not wide:
            gap = 0.46
            text_w = CONTENT_W * 0.50 - gap / 2
            img_w = CONTENT_W - text_w - gap
            s_side = fit_scale(blocks, text_w, avail_total, 0.0, lo=0.76)
            if s_side >= 0.78:
                y_end = draw_blocks(sl, blocks, MARGIN_L, CONTENT_TOP, text_w, avail_total, s_side)
                place_images(sl, paths, MARGIN_L + text_w + gap, CONTENT_TOP, img_w, avail_total)
                continue

        if wide:
            img_reserve = min(3.3, CONTENT_W / ars[0] + 0.10)

        scale = fit_scale(blocks, CONTENT_W, avail_total, img_reserve, lo=0.72)
        if not wide and scale <= 0.80 and len(blocks) > 1:
            gap = 0.44
            col_w = (CONTENT_W - gap) / 2
            two = best_two_column(blocks, col_w, avail_total, img_reserve)
            if two and two[1] > scale + 0.02:
                k, s2 = two
                draw_blocks(sl, blocks[:k], MARGIN_L, CONTENT_TOP, col_w, avail_total, s2)
                x2 = MARGIN_L + col_w + gap
                y_end = draw_blocks(sl, blocks[k:], x2, CONTENT_TOP, col_w, avail_total, s2)
                if paths:
                    room = CONTENT_BOTTOM - y_end - 0.08
                    if not place_images(sl, paths, x2, y_end + 0.12, col_w, max(room, 0)):
                        place_images(sl, paths, MARGIN_L, CONTENT_BOTTOM - 1.25, CONTENT_W, 1.20)
                continue

        scale = fit_scale(blocks, CONTENT_W, avail_total, img_reserve, lo=0.60)
        need = sum(est_block_h(b, scale, CONTENT_W) for b in blocks)
        need += 0.14 * sum(1 for b in blocks if b.kind == "table")
        top = CONTENT_TOP
        if not paths and need < avail_total * 0.80:
            top = CONTENT_TOP + min(0.55, (avail_total - need) * 0.34)
        y_end = draw_blocks(sl, blocks, MARGIN_L, top, CONTENT_W, avail_total, scale)
        if paths:
            room = CONTENT_BOTTOM - y_end - 0.08
            place_images(sl, paths, MARGIN_L, y_end + 0.10, CONTENT_W, max(room, 0))

    prs.save(OUT)
    print(f"wrote {OUT}  ({total} slides)")
    return slides


if __name__ == "__main__":
    build()
