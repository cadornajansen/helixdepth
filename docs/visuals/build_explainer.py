"""Build the editable, evidence-based HelixDepth demo diagram using SVG."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INK = "#152A35"
MUTED = "#5F737E"
BLUE = "#4778D5"
TEAL = "#008779"
parts: list[str] = []


def text(x: float, y: float, value: str, size: int = 22, color: str = INK,
         weight: int = 400, anchor: str = "start") -> None:
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def box(x: float, y: float, w: float, h: float, fill: str = "#FFFFFF",
        stroke: str = "#DCE5E8", radius: int = 18) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#A9BAC2",
         arrow: bool = False, dashed: bool = False, width: int = 2) -> None:
    parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"'
                 + (' marker-end="url(#arrow)"' if arrow else '')
                 + (' stroke-dasharray="6 7"' if dashed else '') + '/>')


def section(y: int, number: str, title: str, subtitle: str) -> None:
    line(70, y, 1530, y, "#DCE5E8")
    text(70, y + 52, number, 20, TEAL, 700)
    text(128, y + 54, title, 31, INK, 650)
    text(128, y + 90, subtitle, 21, MUTED)


def node(x: int, y: int, w: int, title: str, detail: str, fill: str = "#FFFFFF") -> None:
    box(x, y, w, 100, fill)
    text(x + 22, y + 40, title, 25, INK, 600)
    text(x + 22, y + 74, detail, 20, MUTED)


parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="2520" viewBox="0 0 1600 2520" role="img" aria-labelledby="title desc">')
parts.append('<title id="title">HelixDepth: from text to evidence</title><desc id="desc">A complete 2D explainer of data preparation, shared Transformer blocks, next-token training, matched evaluation and measured results at approximately 300 million training targets per model.</desc>')
parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#A9BAC2" stroke-width="1.5"/></marker></defs>')
parts.append('<g font-family="Segoe UI, Arial, sans-serif">')
box(0, 0, 1600, 2520, "#F7FAFA", "#F7FAFA", 0)
text(70, 75, "HELIXDEPTH / THE EXPERIMENT", 18, TEAL, 700)
text(70, 145, "More processing. Reused weights.", 54, INK, 700)
text(70, 195, "Can a small language model improve by reusing its Transformer blocks?", 26, MUTED)
box(70, 224, 1460, 56, "#EAF3F1", "#EAF3F1", 12)
text(95, 260, "Trained from scratch  ·  ~41M parameters  ·  One RTX 4090  ·  Matched data and token budgets", 23, TEAL, 600)

section(320, "01", "Turn text into a learning problem", "A language model learns to predict the next token — a numbered piece of text.")
for x, title, detail in [(70, "Source text", "FineWeb-Edu · pinned version"), (448, "Filter + separate", "Exclude exact prior duplicates"), (826, "Tokenizer", "Text → IDs · 16,384 entries"), (1204, "Packed sequences", "512 tokens per context")]:
    node(x, 435, 326, title, detail)
for x in (396, 774, 1152):
    line(x + 7, 485, x + 42, 485, arrow=True)
text(70, 582, "TRAINING TEXT", 17, BLUE, 700)
text(245, 582, "Used to update weights", 22, MUTED)
text(765, 582, "VALIDATION TEXT", 17, TEAL, 700)
text(964, 582, "Held out · used to measure improvement", 22, MUTED)
text(70, 620, "Tokenizer and validation stay fixed. Exact deduplication does not guarantee removal of all near duplicates.", 20, MUTED)

section(660, "02", "The architectural idea", "A block is a bundle of calculations. A parameter is one learned number inside the model.")
box(70, 782, 700, 290)
box(800, 782, 730, 290, "#F0F8F5", "#CFE6DD")
text(98, 827, "BASELINE", 19, BLUE, 700)
text(98, 866, "Six stored blocks · six stages", 27, INK, 600)
text(828, 827, "HELIXDEPTH", 19, TEAL, 700)
text(828, 866, "Six stored blocks · twelve stages", 27, INK, 600)
for start, color, y in [(98, BLUE, 910), (828, TEAL, 910), (828, TEAL, 982)]:
    for i in range(6):
        x = start + i * 105
        box(x, y, 82, 50, "#EFF3FB" if color == BLUE else "#DCEEE7", "none", 10)
        text(x + 41, y + 33, str(i + 1), 23, color, 700, "middle")
        if i < 5:
            line(x + 85, y + 25, x + 102, y + 25, arrow=True)
text(98, 1018, "40,968,320 parameters", 23, BLUE, 600)
parts.append('<path d="M 1444 958 L 1444 972 L 820 972 L 820 1007 L 826 1007" fill="none" stroke="#A9BAC2" stroke-width="2" marker-end="url(#arrow)"/>')
text(1245, 1054, "Same blocks, reused", 18, TEAL)
text(828, 1114, "41,184,432 parameters · small stage-dependent changes", 23, TEAL, 600)
text(98, 1114, "Each block is used once", 23, MUTED)
box(70, 1145, 1460, 98, "#FFFFFF")
text(95, 1184, "Stage identity", 23, INK, 600)
line(257, 1177, 317, 1177, arrow=True)
text(339, 1184, "Small conditioning network", 23, INK, 600)
line(664, 1177, 724, 1177, arrow=True)
text(746, 1184, "Coefficients", 23, INK, 600)
line(909, 1177, 969, 1177, arrow=True)
text(991, 1184, "Small low-rank weight changes", 23, TEAL, 600)
text(95, 1219, "The experiment combines weight reuse and stage conditioning. Extra stages also require extra computation.", 20, MUTED)

section(1280, "03", "One training step: predict → measure → adjust", "16 sequences × 512 token targets = 8,192 predictions per optimizer update.")
for x, title, detail in [(70, "Predict", "Read context; score next tokens"), (448, "Measure loss", "How wrong were the predictions?"), (826, "Backpropagate", "Calculate gradients of the loss"), (1204, "Update weights", "AdamW makes small adjustments")]:
    node(x, 1400, 326, title, detail)
for x in (396, 774, 1152):
    line(x + 7, 1450, x + 42, 1450, arrow=True)
parts.append('<path d="M 1367 1500 L 1367 1530 L 233 1530 L 233 1503" fill="none" stroke="#A9BAC2" stroke-width="2" marker-end="url(#arrow)"/>')
text(800, 1566, "Repeat with the next batch", 21, MUTED, 500, "middle")
text(70, 1612, "Gradient = direction to change a weight", 22, MUTED)
text(765, 1612, "Learning rate = size of that change", 22, MUTED)

section(1650, "04", "Compare quality and cost", "Same tokenizer, data, target budget and evaluation. Similar parameter counts; different compute.")
for x, title, detail, color in [(70, "Validation / perplexity", "How well does it predict unseen text?", TEAL), (566, "Benchmark accuracy", "How often does it choose correctly?", BLUE), (1062, "Time + memory", "What resources does it need?", MUTED)]:
    box(x, 1770, 468, 105)
    text(x + 23, 1810, title, 25, color, 650)
    text(x + 23, 1848, detail, 20, MUTED)
text(70, 1915, "Checkpoints save weights and recovery state. Full validation selects the best candidate, not just the latest one.", 22, MUTED)

section(1950, "05", "The final matched comparison", "Full frozen-validation loss · lower is better · ~300M targets completed per model.")
box(70, 2070, 770, 290)
text(95, 2107, "LOSS", 16, MUTED, 600)
for value in (3.7, 3.8, 3.9, 4.0):
    y = 2310 - (value - 3.65) / 0.35 * 160
    line(151, y, 799, y, "#E4ECEF")
    text(127, y + 6, f"{value:.1f}", 17, MUTED, anchor="end")
for values, color in [((3.9726352411, 3.8224460835, 3.7476057395), BLUE), ((3.9181522866, 3.7680845546, 3.6934699383), TEAL)]:
    ys = [2310 - (v - 3.65) / 0.35 * 160 for v in values]
    line(210, ys[0], 430, ys[1], color, width=4)
    line(430, ys[1], 650, ys[2], color, width=4)
    for x, y, v in zip((210, 430, 650), ys, values):
        parts.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{color}"/>')
        text(x + 15, y - 10, f"{v:.4f}", 20, color, 650)
text(210, 2335, "~100M targets", 19, MUTED, anchor="middle")
text(430, 2335, "~200M targets", 19, MUTED, anchor="middle")
text(650, 2335, "~300M targets", 19, MUTED, anchor="middle")
box(872, 2070, 658, 290, "#FFFFFF")
text(896, 2115, "1.26% lower WikiText perplexity", 28, TEAL, 650)
text(896, 2152, "HelixDepth vs baseline after matched ~300M training", 20, MUTED)
line(896, 2176, 1506, 2176, "#DCE5E8")
text(896, 2215, "Final ~300M external evaluation", 23, INK, 600)
text(896, 2250, "WikiText perplexity: 86.51 → 85.42", 23, TEAL)
text(896, 2284, "Reasoning: mixed · one seed per model", 23, MUTED)
text(896, 2318, "HelixDepth training time: approximately 2.1×", 23, MUTED)
line(100, 2389, 125, 2389, BLUE, width=4)
text(137, 2396, "Baseline", 19, BLUE)
line(300, 2389, 325, 2389, TEAL, width=4)
text(337, 2396, "HelixDepth", 19, TEAL)
text(605, 2396, "10/10 evaluations passed · models and evidence backed up", 20, MUTED)
text(70, 2440, "Evidence supports a quality–compute tradeoff. One seed cannot establish universal superiority.", 22, INK, 600)
text(70, 2483, "Results snapshot: October 2, 2026  ·  Different datasets have different losses  ·  No claim of guaranteed reasoning gains", 18, MUTED)
parts.append('</g></svg>')
ROOT.mkdir(parents=True, exist_ok=True)
svg = "\n".join(parts)
(ROOT / "helixdepth-explained.svg").write_text(svg, encoding="utf-8")
html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HelixDepth explained</title><style>body{margin:0;background:#eaf0f0;font:16px 'Segoe UI',sans-serif}header{position:sticky;top:0;z-index:1;display:flex;gap:12px;align-items:center;padding:12px 24px;background:#ffffffed;border-bottom:1px solid #dce5e8}button,a{font:inherit;color:#152a35;background:#fff;border:1px solid #dce5e8;border-radius:8px;padding:8px 12px;text-decoration:none;cursor:pointer}header span{margin-right:auto;font-weight:600}main{overflow:auto;padding:24px}.sheet{width:min(1600px,100%);margin:auto;box-shadow:0 8px 32px #152a3510}.sheet svg{width:100%;height:auto;display:block}@media print{header{display:none}main{padding:0}.sheet{width:100%;box-shadow:none}}</style><header><span>HelixDepth / visual study guide</span><button onclick="zoom(-.2)" aria-label="Zoom out">−</button><button onclick="zoom(.2)" aria-label="Zoom in">+</button><button onclick="location.reload()">Fit</button><a href="helixdepth-explained.svg" download>SVG</a><a href="helixdepth-explained.png" download>PNG</a></header><main><div class="sheet">''' + svg + '''</div></main><script>let scale=1;function zoom(delta){scale=Math.max(.4,Math.min(2.5,scale+delta));document.querySelector('.sheet').style.width=1600*scale+'px';}</script></html>'''
(ROOT / "helixdepth-explained.html").write_text(html, encoding="utf-8")
print(ROOT / "helixdepth-explained.svg")
