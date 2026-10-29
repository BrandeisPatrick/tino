#!/usr/bin/env python
"""Build the article page from paper/article.md and the figures in paper/figures/.

Writes docs/pages/article/index.html, a standalone page with every figure inlined and
recoloured through CSS tokens so it reads in light and dark themes. With --fragment PATH
it also writes the same page without the document skeleton, which is what the published
artifact takes.

    uv run python scripts/figures.py && uv run python scripts/build_article.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "paper" / "article.md"
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "docs" / "pages" / "article" / "index.html"
TITLE = "Small Decoders as Decision Models"
FONTS = ("https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400"
         "&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap")

# Exact colours written by scripts/figures.py -> the CSS tokens defined in STYLE.
TOKENS = {
    "#1b1b1b": "--ink", "#5c5b57": "--ink2", "#8a8882": "--muted", "#e3e2dc": "--grid",
    "#c4c3bb": "--axis", "#ffffff": "--surface", "#f3f2ee": "--tint", "#6f6d67": "--teacher",
    "#2a78d6": "--c-adapter", "#eb6834": "--c-finetuned", "#1baf7a": "--c-distil",
    "#eda100": "--c-distil-a1", "#e87ba4": "--c-adapter-post", "#6250d6": "--c-thinking",
    "#86b6ef": "--ord1", "#256abf": "--ord2", "#104281": "--ord3",
}

STYLE = """
/* Layout: one centred reading column of about 65 characters; figures run wider on the same axis. */
:root{
  --bg:#ffffff; --fg:#1b1b1b; --fg2:#5c5b57; --accent:#1c5cab; --rule:#dcdbd5;
  --ink:#1b1b1b; --ink2:#5c5b57; --muted:#8a8882; --grid:#e3e2dc; --axis:#c4c3bb; --surface:#ffffff;
  --tint:#f3f2ee; --teacher:#6f6d67;
  --c-adapter:#2a78d6; --c-finetuned:#eb6834; --c-distil:#1baf7a; --c-distil-a1:#eda100;
  --c-adapter-post:#e87ba4; --c-thinking:#6250d6; --ord1:#86b6ef; --ord2:#256abf; --ord3:#104281;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"Source Sans 3", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --figfont:"Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono:ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#17181b; --fg:#e6e5e1; --fg2:#b4b3ad; --accent:#86b6ef; --rule:#2e2f34;
  --ink:#e6e5e1; --ink2:#b4b3ad; --muted:#8e8d88; --grid:#2c2d31; --axis:#44454a; --surface:#17181b;
  --tint:#202126; --teacher:#a09e97;
  --c-adapter:#3987e5; --c-finetuned:#d95926; --c-distil:#199e70; --c-distil-a1:#c98500;
  --c-adapter-post:#d55181; --c-thinking:#9085e9; --ord1:#b7d3f6; --ord2:#6da7ec; --ord3:#2a78d6;
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#17181b; --fg:#e6e5e1; --fg2:#b4b3ad; --accent:#86b6ef; --rule:#2e2f34;
  --ink:#e6e5e1; --ink2:#b4b3ad; --muted:#8e8d88; --grid:#2c2d31; --axis:#44454a; --surface:#17181b;
  --tint:#202126; --teacher:#a09e97;
  --c-adapter:#3987e5; --c-finetuned:#d95926; --c-distil:#199e70; --c-distil-a1:#c98500;
  --c-adapter-post:#d55181; --c-thinking:#9085e9; --ord1:#b7d3f6; --ord2:#6da7ec; --ord3:#2a78d6;
  color-scheme:dark;
}
body{background:var(--bg);color:var(--fg);font-family:var(--serif);font-size:17px;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.page{max-width:64rem;margin:0 auto;padding-block:2.5rem 4rem;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.page>*{max-width:41rem;margin-inline:auto;min-width:0}
.page>figure,.page>.table{max-width:none}
header.title{max-width:54rem;margin-bottom:2rem}
h1{font-family:var(--sans);font-weight:700;font-size:clamp(1.55rem,3.2vw,2.25rem);line-height:1.15;
  letter-spacing:-0.012em;text-wrap:balance;margin:0 0 .8rem}
.byline{font-family:var(--sans);color:var(--fg2);font-size:.95rem;margin:0}
.abstract{border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:1.1rem 0;margin-bottom:2.2rem}
.abstract h2{font-family:var(--sans);font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--fg2);margin:0 0 .5rem;font-weight:600}
.abstract p{margin:0;font-size:1.02rem}
h2{font-family:var(--sans);font-weight:600;font-size:1.35rem;line-height:1.2;margin:2.6rem auto .7rem;text-wrap:balance}
h3{font-family:var(--sans);font-weight:600;font-size:1.05rem;margin:1.8rem auto .4rem}
p{margin:0 auto .95rem}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
ul,ol{padding-left:1.3rem;margin:0 auto 1rem} li{margin-bottom:.35rem}
code{font-family:var(--mono);font-size:.84em;background:var(--tint);padding:.1em .3em;border-radius:3px}
figure{margin:2.4rem auto 2.6rem}
.fig{overflow-x:auto;padding-block:.2rem}
.fig svg{display:block;width:100%;height:auto;min-width:640px}
.fig svg text{font-family:var(--figfont) !important}
figcaption{font-family:var(--sans);font-size:.9rem;line-height:1.45;color:var(--fg2);max-width:41rem;margin:.7rem auto 0}
figcaption b{color:var(--fg);font-weight:600}
.table{overflow-x:auto;margin:1.4rem auto 1.8rem}
table{border-collapse:collapse;font-family:var(--sans);font-size:.88rem;line-height:1.35;min-width:100%;
  font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--fg2);border-bottom:1px solid var(--fg2)}
hr{border:0;border-top:1px solid var(--rule);margin:2rem auto}
footer{font-family:var(--sans);font-size:.85rem;color:var(--fg2);border-top:1px solid var(--rule);
  padding-top:1rem;margin-top:3rem}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:640px){ body{font-size:16px} .fig svg{min-width:600px} }
"""


def inline_svg(path: Path, idx: int) -> str:
    s = path.read_text(encoding="utf-8")
    s = re.sub(r"<\?xml[^>]*\?>\s*", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>\s*", "", s)
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.S)
    for _ in range(2):  # drop the fixed size, keep the viewBox
        s = re.sub(r'(<svg\b[^>]*?)\s(?:width|height)="[^"]*"', r"\1", s, count=1)
    s = re.sub(r'\bid="([^"]+)"', lambda m: f'id="f{idx}-{m.group(1)}"', s)
    s = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#f{idx}-{m.group(1)})", s)
    s = re.sub(r'href="#([^"]+)"', lambda m: f'href="#f{idx}-{m.group(1)}"', s)
    for hexv, tok in TOKENS.items():
        s = re.sub(re.escape(hexv), f"var({tok})", s, flags=re.I)
    return s.replace("<svg ", '<svg role="img" ', 1)


def build() -> tuple[str, str]:
    text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "toc", "attr_list"], output_format="html5")
    m = re.match(r"\s*<h1[^>]*>(.*?)</h1>\s*<p>(.*?)</p>", body, re.S)
    if not m:
        raise SystemExit("article.md must open with a title heading and a byline paragraph")
    header = f'<header class="title"><h1>{m.group(1)}</h1><p class="byline">{m.group(2)}</p></header>'
    body = body[m.end():]
    body = re.sub(r'<h2 id="abstract">Abstract</h2>\s*(.*?)(?=<h2)',
                  r'<section class="abstract"><h2>Abstract</h2>\1</section>', body, count=1, flags=re.S)
    counter = iter(range(1, 100))

    def figure(mm):
        name, caption = mm.group(1), mm.group(2)
        idx = next(counter)
        num = re.match(r"Figure (\d+)\.", caption)
        cap = re.sub(r"^(Figure \d+\.)", r"<b>\1</b>", caption)
        fid = f"fig{num.group(1)}" if num else f"figure-{idx}"
        return f'<figure id="{fid}"><div class="fig">{inline_svg(FIG / name, idx)}</div><figcaption>{cap}</figcaption></figure>'

    body = re.sub(r'<p><img alt="[^"]*" src="figures/([^"]+)"\s*/?></p>\s*<p><em>(.*?)</em></p>', figure, body, flags=re.S)
    body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    head = (f"<title>{TITLE}</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
            f'<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="stylesheet" href="{FONTS}">\n'
            f"<style>{STYLE}</style>\n")
    return head, f'<main class="page">\n{header}\n{body}\n</main>\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="also write the page without the document skeleton, for publishing")
    args = ap.parse_args()
    head, main_html = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
                   f"{head}</head>\n<body>\n{main_html}</body>\n</html>\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), f"({OUT.stat().st_size // 1024} KB)")
    if args.fragment:
        Path(args.fragment).write_text(head + main_html, encoding="utf-8")
        print("wrote", args.fragment)


if __name__ == "__main__":
    main()
