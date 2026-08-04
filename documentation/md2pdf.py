# -*- coding: utf-8 -*-
"""Markdown -> PDF, Korean-safe, no LaTeX toolchain.

    python docs/md2pdf.py docs/some_report.md docs/some_report.pdf

Renders the markdown to styled HTML and prints it with headless Chrome. That route is
used because pandoc/weasyprint/wkhtmltopdf are not installed here and reportlab's
built-in fonts have no Hangul glyphs, whereas Chrome picks up Apple SD Gothic Neo from
the system. Tables, code blocks and blockquotes are styled for A4 print; the status
emoji used in these reports are swapped for coloured glyphs that render in print.

Needs: pip install markdown, and Google Chrome at the standard macOS location.
"""
import html as _html
import re
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
:root { --ink:#16191d; --mute:#6b7480; --line:#d8dde3; --accent:#1a4f8a;
        --bg-soft:#f5f7f9; --warn:#b3421a; }
* { box-sizing: border-box; }
body { font-family: "Apple SD Gothic Neo","AppleGothic","Helvetica Neue",sans-serif;
       font-size: 9.6pt; line-height: 1.55; color: var(--ink); margin: 0;
       -webkit-font-smoothing: antialiased; }
h1 { font-size: 19pt; margin: 0 0 2mm; letter-spacing: -0.02em; }
h1 + p { color: var(--mute); font-size: 9pt; margin-top: 0; }
h2 { font-size: 13pt; margin: 9mm 0 2.5mm; padding-bottom: 1.6mm;
     border-bottom: 1.6px solid var(--accent); letter-spacing: -0.01em;
     break-after: avoid; }
h3 { font-size: 10.8pt; margin: 6mm 0 2mm; color: var(--accent); break-after: avoid; }
h4 { font-size: 9.8pt; margin: 4mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.4mm; }
ul, ol { margin: 0 0 2.6mm; padding-left: 5.5mm; }
li { margin-bottom: 1mm; }
hr { border: 0; border-top: 1px solid var(--line); margin: 7mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm;
        font-size: 8.7pt; break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 1.5mm 2.2mm; text-align: left;
         vertical-align: top; }
th { background: var(--bg-soft); font-weight: 600; }
tbody tr:nth-child(even) { background: #fbfcfd; }
code { font-family: "SF Mono","Menlo",monospace; font-size: 8.4pt;
       background: var(--bg-soft); padding: 0.4mm 1mm; border-radius: 2px; }
pre { background: var(--bg-soft); border: 1px solid var(--line); border-radius: 3px;
      padding: 2.5mm 3mm; overflow-x: auto; break-inside: avoid; margin: 2.5mm 0 4mm; }
pre code { background: none; padding: 0; font-size: 7.9pt; line-height: 1.42; }
blockquote { margin: 2.5mm 0; padding: 2mm 3mm; border-left: 3px solid var(--warn);
             background: #fdf6f3; color: #4a2f26; break-inside: avoid; }
blockquote p { margin: 0; }
strong { font-weight: 650; }
h2, h3 { break-inside: avoid; }
"""


def build(md_path: Path, out_pdf: Path):
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    # keep the emoji status marks legible in print
    body = (body.replace("🔴", '<span style="color:#c0392b">●</span>')
                .replace("🟡", '<span style="color:#c8901a">●</span>')
                .replace("🟢", '<span style="color:#1e8449">●</span>')
                .replace("⚠️", '<span style="color:#b3421a">▲</span>')
                .replace("✅", '<span style="color:#1e8449">✔</span>')
                .replace("❌", '<span style="color:#c0392b">✘</span>'))
    title = _html.escape(re.sub(r"^#\s*", "", text.splitlines()[0]).strip())
    doc = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
           f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")
    tmp_html = out_pdf.with_suffix(".html")
    tmp_html.write_text(doc, encoding="utf-8")

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
         f"--print-to-pdf={out_pdf}", tmp_html.as_uri()],
        check=True, capture_output=True, timeout=180)
    tmp_html.unlink(missing_ok=True)
    return out_pdf


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    p = build(src, dst)
    print(f"{p}  ({p.stat().st_size/1024:.0f} KB)")
