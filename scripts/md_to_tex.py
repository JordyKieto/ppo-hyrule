"""Convert `paper/paper.md` into the LaTeX this author's other papers use, then build it with tectonic.

There is no pandoc on this machine and the two sibling reports (`neural-simulators`,
`carla-imitation-learning`) ship a hand-written `paper.tex` in a specific house style: article 11pt, newtx,
booktabs, fancyhdr, a title block carrying the repository and video links, and an `abstract` environment.
This script reproduces that preamble and translates the markdown body into it.

It is deliberately narrow. It handles exactly the constructs `paper/paper.md` uses -- headings, paragraphs,
bold, italics, inline code, links, bare URLs, bullet lists, block quotes, fenced code and pipe tables -- and
raises on anything it does not recognise rather than silently dropping it, because a paper that quietly loses
a table is worse than one that fails to build.

    .venv/bin/python scripts/md_to_tex.py              # writes paper/paper.tex and builds paper/paper.pdf
    .venv/bin/python scripts/md_to_tex.py --no-build   # tex only

EXPERIMENT-CONTRACT: exempt -- a document converter. It opens no environment, trains nothing and writes no
checkpoint.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GH = "https://github.com/JordyKieto/ppo-hyrule"
HF = "https://huggingface.co/jordykieto/ppo-hyrule"

PREAMBLE = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{enumitem}
\usepackage[font=small,labelfont=bf,skip=6pt]{caption}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{fancyvrb}
\usepackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black}

\setlist{itemsep=2pt,topsep=4pt}
\emergencystretch=3em          % long \texttt{} paths cannot hyphenate; let the line stretch instead
\usepackage{seqsplit}
\renewcommand{\arraystretch}{1.15}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small\scshape Technical Report}
\fancyhead[R]{\small\scshape The Legend of PPO}
\fancyfoot[C]{\small\thepage}
\fancypagestyle{firstpage}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[C]{\small\thepage}}

\title{\LARGE\bfseries The Legend of PPO:\\ Learning to Fight in Hyrule}
\author{%
  \textbf{Jordy Kieto}\\[4pt]
  \small Code, claim audit and provenance: \url{GHURL}\\
  \small Models: \url{HFURL}\\
  \small Video: \url{https://youtu.be/FBD1wEsdY3E}
}
\date{}

\begin{document}

\maketitle
\thispagestyle{firstpage}
""".replace("GHURL", GH).replace("HFURL", HF)

SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(t: str) -> str:
    out = []
    for ch in t:
        out.append(SPECIALS.get(ch, ch))
    return "".join(out).replace("\\", "\\") if False else "".join(out)


def inline(t: str) -> str:
    """Markdown inline to LaTeX. Code spans and links are protected before anything is escaped."""
    slots: list[str] = []

    def park(s: str) -> str:
        slots.append(s)
        return f"\x00{len(slots) - 1}\x00"

    def tt(x: str) -> str:
        """Long paths inside \texttt cannot hyphenate and overflow the margin; let them break on / and _."""
        x = (x.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("{", r"\{")
              .replace("}", r"\}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
              .replace("#", r"\#").replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))
        if len(x) > 24:
            x = x.replace("/", "/\\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        return r"\texttt{" + x + "}"

    # {{fig:name}} becomes a real cross-reference, so a figure number cannot drift from the
    # prose the way a hand-typed "Figure 3" does when a figure is inserted above it. Parked
    # before escaping, like any other raw LaTeX.
    t = re.sub(r"\{\{fig:(\w+)\}\}", lambda m: park(r"Figure~\ref{fig:" + m.group(1) + "}"), t)
    t = re.sub(r"`([^`]+)`", lambda m: park(tt(m.group(1))), t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: park(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{"
                              + esc(m.group(1)) + "}"), t)
    t = re.sub(r"<(https?://[^>]+)>", lambda m: park(r"\url{" + m.group(1) + "}"), t)
    t = re.sub(r"(?<![\w/])(https?://[^\s,)]+)", lambda m: park(r"\url{" + m.group(1).rstrip('.') + "}"), t)

    t = esc(t)
    # Bold FIRST and lazily across asterisks: the subtitle is a bold span containing an italic one, and
    # [^*]+ refuses to cross it, which printed the literal ** markers on page 1.
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t, flags=re.S)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", t)
    t = t.replace("±", r"$\pm$").replace("→", r"$\rightarrow$").replace("×", r"$\times$")
    t = t.replace("“", "``").replace("”", "''").replace("’", "'").replace("‘", "`")
    # Restore until nothing is left. A link's TEXT can itself be a parked code span, so a single pass
    # leaves a placeholder buried inside another slot's content and XeTeX halts on the NUL byte.
    for _ in range(8):
        if "\x00" not in t:
            break
        for i, s in enumerate(slots):
            t = t.replace(f"\x00{i}\x00", s)
    if "\x00" in t:
        raise ValueError(f"unresolved inline placeholder: {t[:120]!r}")
    return t


def table(rows: list[list[str]]) -> str:
    """Pipe table to booktabs. A column whose cells are mostly numeric is right-aligned, the rest wrap."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    numeric = []
    for c in range(ncol):
        cells = [b[c].strip() for b in body if b[c].strip()]
        hits = sum(1 for x in cells if re.fullmatch(r"[-+~<>$\\\d.,%/ ±a-z]{0,24}", x) and re.search(r"\d", x))
        numeric.append(bool(cells) and hits >= 0.7 * len(cells))
    spec = "".join("r" if numeric[c] else "X" for c in range(ncol))
    if "X" not in spec:                       # all-numeric: let it size naturally
        spec = "l" + "r" * (ncol - 1)
        env, arg = "tabular", ""
    else:
        env, arg = "tabularx", r"{\linewidth}"
    # Eight columns overflow at \small; the per-start profile table is the one that forced this.
    size = r"\footnotesize" if ncol >= 7 else r"\small"
    out = [r"\begin{center}" + size, f"\\begin{{{env}}}{arg}{{@{{}}{spec}@{{}}}}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + inline(h.strip().strip("*")) + "}" for h in head) + r" \\")
    out.append(r"\midrule")
    for b in body:
        out.append(" & ".join(inline(c.strip()) for c in b) + r" \\")
    out += [r"\bottomrule", f"\\end{{{env}}}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, in_list, in_abstract = 0, False, False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                                   # fenced code
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out += [r"\begin{Verbatim}[fontsize=\footnotesize,frame=single,framesep=4pt]", *buf, r"\end{Verbatim}"]
            continue
        m = re.match(r"^!\[(.*)\]\(([^)]+)\)\s*$", ln)             # figure
        if m:
            close_list()
            caption, path = m.group(1), m.group(2)
            key = re.sub(r"\W+", "", os.path.splitext(os.path.basename(path))[0])
            # A figure wider than it is tall is set across the full column; the strip-shaped
            # teaser would otherwise be set at a size nobody can read.
            out += [
                r"\begin{figure}[" + ("!ht" if "teaser" in path else "tbp") + "]",
                r"  \centering",
                r"  \includegraphics[width=\linewidth]{" + path + "}",
                r"  \caption{" + inline(caption) + "}",
                r"  \label{fig:" + key + "}",
                r"\end{figure}",
            ]
            i += 1
            continue
        if re.match(r"^\s*\|", ln):                                # pipe table
            close_list()
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                cells = [c for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[\s|:-]+", lines[i]):
                    rows.append(cells)
                i += 1
            out.append(table(rows))
            continue
        if ln.startswith("# "):
            i += 1; continue                                       # title handled by \maketitle
        if ln.startswith("## "):
            close_list()
            name = re.sub(r"^##\s*(?:\d+\.?\s*)?", "", ln).strip()
            if name.lower() == "abstract":
                out.append(r"\begin{abstract}\noindent"); in_abstract = True
            else:
                if in_abstract:
                    out.append(r"\end{abstract}"); in_abstract = False
                out.append(r"\section{" + inline(name) + "}")
            i += 1; continue
        if ln.startswith("### "):
            close_list()
            out.append(r"\subsection{" + inline(re.sub(r"^###\s*[\d.]*\s*", "", ln).strip()) + "}")
            i += 1; continue
        if ln.startswith("> "):
            close_list()
            buf = []
            while i < len(lines) and lines[i].startswith("> "):
                buf.append(lines[i][2:]); i += 1
            out += [r"\begin{quote}", inline(" ".join(buf)), r"\end{quote}"]
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            if not in_list:
                out.append(r"\begin{itemize}"); in_list = True
            item = re.sub(r"^\s*[-*]\s+", "", ln)
            while i + 1 < len(lines) and re.match(r"^\s{2,}\S", lines[i + 1]) and not re.match(r"^\s*[-*]\s", lines[i + 1]):
                i += 1; item += " " + lines[i].strip()
            out.append(r"\item " + inline(item))
            i += 1; continue
        if re.match(r"^\d+\.\s", ln):
            if not in_list:
                out.append(r"\begin{itemize}"); in_list = True
            item = re.sub(r"^\d+\.\s", "", ln)
            while i + 1 < len(lines) and re.match(r"^\s{2,}\S", lines[i + 1]):
                i += 1; item += " " + lines[i].strip()
            out.append(r"\item " + inline(item))
            i += 1; continue
        if ln.strip() in ("---", "***"):
            close_list(); i += 1; continue
        if not ln.strip():
            close_list(); out.append(""); i += 1; continue
        close_list()
        para = [ln]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||>|```|\s*[-*]\s|\d+\.\s|---$)", lines[i + 1]):
            i += 1; para.append(lines[i])
        out.append(inline(" ".join(x.strip() for x in para)))
        i += 1
    close_list()
    if in_abstract:
        out.append(r"\end{abstract}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    md = (REPO / "paper" / "paper.md").read_text()
    tex = PREAMBLE + "\n" + convert(md) + "\n\n\\end{document}\n"
    out = REPO / "paper" / "paper.tex"
    out.write_text(tex)
    print(f"wrote {out} ({len(tex.splitlines())} lines)")
    if a.no_build:
        return
    r = subprocess.run(["tectonic", "-X", "compile", str(out), "--outdir", str(REPO / "paper")],
                       capture_output=True, text=True)
    pdf = REPO / "paper" / "paper.pdf"
    if r.returncode != 0 or not pdf.exists():
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"tectonic failed (exit {r.returncode})")
    print(f"built {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
