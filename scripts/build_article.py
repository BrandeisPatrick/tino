#!/usr/bin/env python
"""Build the article page from paper/article.md and the figures in paper/figures/.

The page follows the layout of a research-lab article: a masthead, a sticky contents
rail on wide screens, numbered citations that resolve to a references list, figures with
numbered captions, and a citation block with BibTeX. Every figure is inlined and its
colours are mapped to CSS tokens so it reads in light and dark themes.

Writes docs/pages/article/index.html (standalone). With --fragment PATH it also writes
the same page without the document skeleton, which is what the published artifact takes.

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
REPO = "https://github.com/BrandeisPatrick/tino"
FONTS = ("https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400"
         "&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400&display=swap")

# Exact colours written by scripts/figures.py -> the CSS tokens defined in STYLE.
TOKENS = {
    "#1b1b1b": "--ink", "#5c5b57": "--ink2", "#8a8882": "--muted", "#e3e2dc": "--grid",
    "#c4c3bb": "--axis", "#ffffff": "--surface", "#f3f2ee": "--tint", "#6f6d67": "--teacher",
    "#2a78d6": "--c-adapter", "#eb6834": "--c-finetuned", "#1baf7a": "--c-distil",
    "#eda100": "--c-distil-a1", "#e87ba4": "--c-adapter-post", "#6250d6": "--c-thinking",
    "#86b6ef": "--ord1", "#256abf": "--ord2", "#104281": "--ord3",
}

STYLE = """
/* Layout: a research-lab article. Masthead on top, a sticky contents rail at the left on wide
   screens, one reading column of about 65 characters, figures running wider on the same axis. */
:root{
  --bg:#faf9f6; --fg:#1c1a17; --fg2:#5e5a53; --muted:#8b867d; --rule:#e6e2da; --accent:#c4613a;
  --accent-ink:#a84e2d; --tint:#f1efe9;
  --ink:#1c1a17; --ink2:#5e5a53; --grid:#e6e2da; --axis:#c9c4ba; --surface:#faf9f6; --teacher:#6f6d67;
  --c-adapter:#2a78d6; --c-finetuned:#eb6834; --c-distil:#1baf7a; --c-distil-a1:#eda100;
  --c-adapter-post:#e87ba4; --c-thinking:#6250d6; --ord1:#86b6ef; --ord2:#256abf; --ord3:#104281;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"Instrument Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --figfont:"Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --col:42rem; --wide:62rem; --rail:13rem; --gutter:3.5rem;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#1b1a18; --fg:#ece9e2; --fg2:#b5b0a6; --muted:#8b867d; --rule:#34322d; --accent:#e08a64;
  --accent-ink:#e9a083; --tint:#262420;
  --ink:#ece9e2; --ink2:#b5b0a6; --grid:#33312c; --axis:#4a4842; --surface:#1b1a18; --teacher:#a09e97;
  --c-adapter:#3987e5; --c-finetuned:#d95926; --c-distil:#199e70; --c-distil-a1:#c98500;
  --c-adapter-post:#d55181; --c-thinking:#9085e9; --ord1:#b7d3f6; --ord2:#6da7ec; --ord3:#2a78d6;
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#1b1a18; --fg:#ece9e2; --fg2:#b5b0a6; --muted:#8b867d; --rule:#34322d; --accent:#e08a64;
  --accent-ink:#e9a083; --tint:#262420;
  --ink:#ece9e2; --ink2:#b5b0a6; --grid:#33312c; --axis:#4a4842; --surface:#1b1a18; --teacher:#a09e97;
  --c-adapter:#3987e5; --c-finetuned:#d95926; --c-distil:#199e70; --c-distil-a1:#c98500;
  --c-adapter-post:#d55181; --c-thinking:#9085e9; --ord1:#b7d3f6; --ord2:#6da7ec; --ord3:#2a78d6;
  color-scheme:dark;
}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{background:var(--bg);color:var(--fg);font-family:var(--serif);font-size:18px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.topbar{border-bottom:1px solid var(--rule)}
.topbar-inner,.page{max-width:84rem;margin:0 auto;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.topbar-inner{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;padding-block:.85rem;
  font-family:var(--sans);font-size:.9rem}
.brand{font-weight:600;letter-spacing:.01em;color:var(--fg);text-decoration:none}
.brand small{font-weight:400;color:var(--fg2);margin-left:.6rem;font-size:.9rem}
.topbar-inner .kind{color:var(--fg2)}
.page{padding-block:3rem 5rem}
.masthead,.layout{display:grid;grid-template-columns:minmax(0,1fr);column-gap:var(--gutter)}
.masthead{margin-bottom:2.8rem;row-gap:0}
.masthead>*{min-width:0;max-width:56rem}
.eyebrow{font-family:var(--sans);font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--fg2);margin:0 0 1.2rem}
h1{font-family:var(--serif);font-weight:500;font-size:clamp(2rem,4.2vw,3.15rem);line-height:1.1;letter-spacing:-0.018em;
  text-wrap:balance;margin:0 0 1.3rem}
.authors{font-family:var(--sans);font-size:1.02rem;font-weight:500;margin:0 0 .4rem}
.meta{font-family:var(--sans);font-size:.9rem;color:var(--fg2);margin:0}
.meta a{color:var(--fg2)}
.toc{font-family:var(--sans);font-size:.86rem;line-height:1.4;min-width:0}
.toc-title{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .7rem}
.toc ol{list-style:none;margin:0;padding:0}
.toc ol ol{padding-left:.9rem;margin:.25rem 0 .45rem}
.toc li{margin:0 0 .4rem}
.toc a{color:var(--fg2);text-decoration:none;display:inline-block;border-left:2px solid transparent;padding-left:.6rem;margin-left:-.7rem}
.toc a:hover{color:var(--fg)}
.toc a.active{color:var(--fg);border-left-color:var(--accent)}
.body{min-width:0}
.body>*{max-width:var(--col);min-width:0}
.body>figure,.body>.table{max-width:var(--wide)}
@media (min-width:1100px){
  .masthead,.layout{grid-template-columns:var(--rail) minmax(0,var(--wide))}
  .masthead>*{grid-column:2}
  .toc{position:sticky;top:1.5rem;align-self:start;max-height:calc(100vh - 3rem);overflow:auto;padding-right:.5rem}
}
@media (max-width:1099px){
  .toc{border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:1rem 0 .6rem;margin-bottom:2rem}
  .toc ol{columns:2;column-gap:2rem}
  .toc ol ol{display:none}
  .toc li{break-inside:avoid}
}
.abstract{margin:0 0 2.6rem;padding-bottom:1.8rem;border-bottom:1px solid var(--rule)}
.abstract .label{font-family:var(--sans);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .7rem}
.abstract p{font-size:1.1rem;line-height:1.55;margin:0}
h2{font-family:var(--sans);font-weight:600;font-size:1.45rem;line-height:1.2;letter-spacing:-0.012em;margin:3.2rem 0 .9rem;text-wrap:balance}
h3{font-family:var(--sans);font-weight:600;font-size:1.08rem;margin:2.1rem 0 .5rem}
p{margin:0 0 1.1rem}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:3px;
  text-decoration-color:color-mix(in srgb,var(--accent) 45%,transparent)}
a:hover{text-decoration-color:var(--accent)}
ul,ol{padding-left:1.4rem;margin:0 0 1.1rem}
li{margin-bottom:.45rem}
strong{font-weight:600}
code{font-family:var(--mono);font-size:.8em;background:var(--tint);padding:.1em .35em;border-radius:3px}
pre{font-family:var(--mono);font-size:.78rem;line-height:1.55;background:var(--tint);padding:1rem 1.1rem;border-radius:6px;
  overflow-x:auto;margin:0 0 1.1rem;color:var(--fg)}
pre code{background:none;padding:0;font-size:inherit}
figure{margin:2.6rem 0 2.9rem}
.fig{overflow-x:auto;padding-block:.2rem}
.fig svg{display:block;width:100%;height:auto;min-width:640px}
.fig svg text{font-family:var(--figfont) !important}
figcaption{font-family:var(--sans);font-size:.9rem;line-height:1.5;color:var(--fg2);max-width:var(--col);margin:.85rem 0 0}
figcaption b{color:var(--fg);font-weight:600}
.table{overflow-x:auto;margin:1.4rem 0 1.8rem}
table{border-collapse:collapse;font-family:var(--sans);font-size:.88rem;line-height:1.4;min-width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.5rem .65rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--fg2);border-bottom:1px solid var(--fg2)}
.table + p em,.caption{font-family:var(--sans);font-size:.9rem;color:var(--fg2);font-style:normal}
sup{vertical-align:baseline;position:relative;top:-.3em;font-size:.72em;line-height:0}
a.footnote-ref{color:var(--accent-ink);text-decoration:none;font-family:var(--sans);font-weight:500}
a.footnote-ref::before{content:"["}
a.footnote-ref::after{content:"]"}
.refs ol{padding-left:1.7rem;font-family:var(--sans);font-size:.88rem;line-height:1.5;color:var(--fg2)}
.refs li{margin-bottom:.65rem;padding-left:.25rem}
.refs li p{margin:0}
.refs li:target{color:var(--fg)}
.refs a{color:var(--fg2)}
.bib{position:relative}
.bib button{position:absolute;top:.6rem;right:.6rem;font:500 .74rem var(--sans);color:var(--fg2);background:var(--bg);
  border:1px solid var(--rule);border-radius:4px;padding:.32rem .6rem;cursor:pointer}
.bib button:hover{color:var(--fg)}
footer{font-family:var(--sans);font-size:.85rem;color:var(--fg2);border-top:1px solid var(--rule);padding-top:1.2rem;margin-top:3.5rem}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:640px){body{font-size:17px} .fig svg{min-width:600px} .page{padding-block:2rem 4rem} h2{margin-top:2.6rem}}
"""

SCRIPT = """
(function(){
  var links=[].slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  var heads=[].slice.call(document.querySelectorAll('.body h2[id], .body h3[id]'));
  function update(){
    var act=null;
    for(var i=0;i<heads.length;i++){ if(heads[i].getBoundingClientRect().top<=130) act=heads[i].id; }
    links.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href')==='#'+act); });
  }
  var ticking=false;
  window.addEventListener('scroll',function(){ if(!ticking){ ticking=true; requestAnimationFrame(function(){ update(); ticking=false; }); } },{passive:true});
  update();
  [].slice.call(document.querySelectorAll('.bib button')).forEach(function(b){
    var pre=b.parentNode.querySelector('pre');
    function select(){ var r=document.createRange(); r.selectNodeContents(pre); var s=window.getSelection(); s.removeAllRanges(); s.addRange(r); }
    function done(ok){ b.textContent=ok?'Copied':'Selected, press copy'; setTimeout(function(){ b.textContent='Copy BibTeX'; },1800); }
    b.addEventListener('click',function(){
      var t=pre.textContent;
      if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(t).then(function(){ done(true); },function(){ select(); done(false); }); }
      else { select(); done(false); }
    });
  });
})();
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


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def build() -> tuple[str, str]:
    text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "toc", "attr_list", "footnotes", "fenced_code", "smarty"],
                             output_format="html5")

    # Masthead: title, an authors line, a meta line.
    m = re.match(r"\s*<h1[^>]*>(.*?)</h1>\s*<p>(.*?)</p>\s*<p>(.*?)</p>", body, re.S)
    if not m:
        raise SystemExit("article.md must open with a title, an authors paragraph and a meta paragraph")
    title_html, authors, meta = m.group(1), m.group(2), m.group(3)
    body = body[m.end():]

    # Abstract as the lede.
    body = re.sub(r'<h2 id="abstract">Abstract</h2>\s*(.*?)(?=<h2)',
                  r'<section class="abstract"><p class="label">Abstract</p>\1</section>', body, count=1, flags=re.S)

    # Figures: an image paragraph followed by an italic "Figure N." paragraph.
    counter = iter(range(1, 100))

    def figure(mm):
        name, caption = mm.group(1), mm.group(2)
        idx = next(counter)
        num = re.match(r"Figure (\d+)\.", caption)
        cap = re.sub(r"^(Figure \d+\.)", r"<b>\1</b>", caption)
        fid = f"fig{num.group(1)}" if num else f"figure-{idx}"
        return (f'<figure id="{fid}"><div class="fig">{inline_svg(FIG / name, idx)}</div>'
                f"<figcaption>{cap}</figcaption></figure>")

    body = re.sub(r'<p><img alt="[^"]*" src="figures/([^"]+)"\s*/?></p>\s*<p><em>(.*?)</em></p>', figure, body, flags=re.S)
    body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")

    # Footnotes become the numbered references list, in order of first citation.
    fn = re.search(r'<div class="footnote">(.*?)</div>\s*$', body, re.S)
    if fn:
        refs = fn.group(1)
        refs = re.sub(r"<hr\s*/?>", "", refs)
        refs = re.sub(r"(&#160;|\s)*<a class=\"footnote-backref\"[^>]*>.*?</a>", "", refs, flags=re.S)
        body = body[:fn.start()] + f'<section class="refs" id="references"><h2 id="references-h">References</h2>{refs}</section>\n'

    # BibTeX block gets a copy button.
    body = re.sub(r'<pre><code class="language-bibtex">(.*?)</code></pre>',
                  r'<div class="bib"><button type="button" id="copy-bibtex">Copy BibTeX</button><pre id="bibtex"><code>\1</code></pre></div>',
                  body, count=1, flags=re.S)

    # Contents rail from the headings that remain.
    items = []
    for level, hid, label in re.findall(r"<h([23]) id=\"([^\"]+)\">(.*?)</h[23]>", body, flags=re.S):
        items.append((int(level), hid, strip_tags(label)))
    toc = ['<nav class="toc" aria-label="Contents"><p class="toc-title">Contents</p><ol>']
    open_sub = False
    for level, hid, label in items:
        if level == 2:
            if open_sub:
                toc.append("</ol></li>")
                open_sub = False
            else:
                if len(toc) > 1:
                    toc.append("</li>")
            toc.append(f'<li><a href="#{hid}">{label}</a>')
        else:
            if not open_sub:
                toc.append("<ol>")
                open_sub = True
            toc.append(f'<li><a href="#{hid}">{label}</a></li>')
    toc.append("</ol></li>" if open_sub else "</li>")
    toc.append("</ol></nav>")
    toc_html = "".join(toc)

    head = (f"<title>{TITLE}</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
            f'<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="stylesheet" href="{FONTS}">\n'
            f"<style>{STYLE}</style>\n")
    main_html = (
        '<div class="topbar"><div class="topbar-inner">'
        f'<a class="brand" href="{REPO}">tino<small>decision models on a laptop</small></a>'
        '<span class="kind">Research</span></div></div>\n'
        '<main class="page">\n'
        '<header class="masthead">'
        '<p class="eyebrow">Research · Decision models · <time datetime="2026-10-28">October 28, 2026</time></p>'
        f"<h1>{title_html}</h1>"
        f'<p class="authors">{authors}</p><p class="meta">{meta}</p></header>\n'
        f'<div class="layout">{toc_html}\n<article class="body">\n{body}\n</article></div>\n'
        f'<footer>Draft. Every number on this page is read from the tracked runs in <a href="{REPO}">the repository</a>; '
        "the figures regenerate with <code>scripts/figures.py</code> and this page with <code>scripts/build_article.py</code>.</footer>\n"
        f"</main>\n<script>{SCRIPT}</script>\n"
    )
    return head, main_html


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
