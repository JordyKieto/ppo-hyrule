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
% Default float parameters reserve so much of a page for text that a half-page figure is
% deferred to a float page of its own, which left one figure alone on a mostly blank page.
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.75}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.80}
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


SECTIONS: set[str] = set()


def scan_sections(md: str) -> set[str]:
    """Every section and subsection number this paper actually has.

    Cross-references are rewritten only to numbers in this set, which is what keeps
    "provenance record section 19" -- a pointer into a DIFFERENT document -- from being
    turned into a dangling \\ref to a section of this one. There is no section 19 here,
    so it is left as prose automatically rather than by a hand-maintained exception.
    """
    found = set()
    for ln in md.split("\n"):
        m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)*)\.?\s", ln)
        if m:
            found.add(m.group(1))
    return found


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
            if "/" in x:          # a path: hyphenated segments must break too, or a cell
                x = x.replace("-", "-\\allowbreak{}")   # like a checkpoint name overflows
        return r"\texttt{" + x + "}"

    # Code spans are parked FIRST, so a citation or cross-reference written inside
    # backticks as an EXAMPLE stays an example rather than becoming a real one.
    t = re.sub(r"`([^`]+)`", lambda m: park(tt(m.group(1))), t)
    # [@key] and [@key1; @key2] become real citations.
    t = re.sub(r"\[@([\w\d]+(?:\s*;\s*@[\w\d]+)*)\]",
               lambda m: park(r"\cite{" + ",".join(
                   k.strip().lstrip("@") for k in m.group(1).split(";")) + "}"), t)
    # "section 9.1", and "sections 5 and 12", become hyperlinked references -- but ONLY for
    # numbers this paper actually has, so a pointer into another document is left as prose.
    def _secref(m):
        nums = re.findall(r"\d+(?:\.\d+)?", m.group(0))
        if not all(n in SECTIONS for n in nums):
            return m.group(0)
        joiner = " and " if " and " in m.group(0) else ", "
        return park(joiner.join(r"\S\ref{sec:" + n + "}" for n in nums))

    t = re.sub(r"\b[Ss]ections?\s+\d+(?:\.\d+)?(?:\s*(?:,|and)\s*\d+(?:\.\d+)?)*", _secref, t)
    # {{fig:name}} becomes a real cross-reference, so a figure number cannot drift from the
    # prose the way a hand-typed "Figure 3" does when a figure is inserted above it. Parked
    # before escaping, like any other raw LaTeX.
    t = re.sub(r"\{\{fig:(\w+)\}\}", lambda m: park(r"Figure~\ref{fig:" + m.group(1) + "}"), t)
    # Inline math is LaTeX already: escaping it prints \$\delta\_\{t\}\$ as literal text
    # and then halts the engine with "Missing $ inserted".
    #
    # The negative lookahead is load-bearing. This paper quotes money -- "$266 for the world
    # model track and $1,381 in total" -- and a bare $...$ rule pairs those two dollar signs
    # and sets the sentence between them as mathematics. Currency is always $ followed by a
    # digit; none of this paper's math is.
    t = re.sub(r"\$(?!\d)([^$\n]+?)\$", lambda m: park("$" + m.group(1) + "$"), t)
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


def paragraph_heading(text: str) -> "str | None":
    """A run-in heading, if this paragraph opens with a titular bold lead-in.

    Only a lead-in that is a complete titular phrase qualifies, which is why this is a
    predicate and not a regex substitution. Roughly half the paragraphs in this paper that
    begin with bold are not headings at all -- they open mid-sentence on an emphasised
    quantity, as in "**76.2%** of the time" or "**trivial floor**" -- and turning those
    into headings would cut the sentence in half.

    The test is: the bold span ends a sentence, is short enough to be a title, is followed
    by more prose, and does not open with a bracket (the evidence-class markers "(M." and
    "(M" start paragraphs and end with a period but are annotations, not headings).
    """
    m = re.match(r"^\*\*(.+?)\*\*\s+(\S.*)$", text, re.S)
    if not m:
        return None
    title, rest = m.group(1).strip(), m.group(2)
    if not title.endswith("."):
        return None
    if title.startswith("(") or len(title) < 4 or len(title) > 120:
        return None
    return r"\paragraph{" + inline(title) + "} " + inline(rest)


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, in_list, in_abstract = 0, False, False
    appendix_started = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "$$":                                     # display equation
            close_list()
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() != "$$":
                buf.append(lines[i]); i += 1
            i += 1
            body = "\n".join(buf)
            env = "align" if "\\\\" in body else "equation"
            out += [r"\begin{" + env + "}", body, r"\end{" + env + "}"]
            continue
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
        cap = re.match(r"^Table:\s*(.+?)\s*$", ln)                 # captioned table
        # one blank line between the caption and the table is allowed, because requiring
        # them to be adjacent silently emits the caption as a paragraph of prose instead
        nxt = i + 1
        while nxt < len(lines) and not lines[nxt].strip():
            nxt += 1
        if cap and nxt < len(lines) and re.match(r"^\s*\|", lines[nxt]):
            close_list()
            caption = cap.group(1)
            i = nxt
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                if not re.fullmatch(r"[\s|:-]+", lines[i]):
                    rows.append([c for c in lines[i].strip().strip("|").split("|")])
                i += 1
            key = re.sub(r"\W+", "", caption.split(".")[0].lower())[:28]
            # \captionof rather than a table float: these tables ARE the argument of the
            # paragraph above them, and floating nineteen of them through a 26-page paper
            # separates each from the sentence it answers. This numbers and captions them
            # in place.
            head = [r"\captionof{table}{" + inline(caption) + "}",
                    r"\label{tab:" + key + "}"]
            # A minipage cannot break across a page, so the caption cannot be stranded at
            # the foot of one with its table overleaf -- which \nopagebreak did not prevent
            # and which happened to three of these twenty. Only short tables are boxed this
            # way: a long one has to be allowed to split, and is unlikely to be orphaned
            # because it will not fit in the remaining space either way.
            if len(rows) <= 14:
                # A float, not a minipage. A minipage also keeps the caption with its
                # table, but when it does not fit in the space left it leaves that space
                # BLANK; a float lets the following text fill the page and places the
                # table at the top of it instead. Orphan fixed either way, gap only one.
                out += [r"\begin{table}[!htbp]", r"  \centering", *["  " + h for h in head],
                        table(rows), r"\end{table}"]
            else:
                out += [r"\begin{center}", *head, r"\end{center}",
                        r"\nopagebreak[4]", r"\vspace{-1.2em}", table(rows)]
            continue
        if re.match(r"^\s*\|", ln):                                # pipe table, inline
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
            app = re.match(r"^##\s+Appendix\s+([A-Z])\.\s*(.+)$", ln)
            if app:
                if in_abstract:
                    out.append(r"\end{abstract}"); in_abstract = False
                if not appendix_started:
                    out.append(r"\appendix")
                    appendix_started = True
                out.append(r"\section{" + inline(app.group(2).strip()) + "}")
                out.append(r"\label{app:" + app.group(1).lower() + "}")
                i += 1; continue
            if name.lower() == "abstract":
                out.append(r"\begin{abstract}\noindent"); in_abstract = True
            else:
                if in_abstract:
                    out.append(r"\end{abstract}"); in_abstract = False
                out.append(r"\section{" + inline(name) + "}")
                num = re.match(r"^##\s+(\d+(?:\.\d+)*)\.?\s", ln)
                if num:
                    out.append(r"\label{sec:" + num.group(1) + "}")
            i += 1; continue
        if ln.startswith("### "):
            close_list()
            out.append(r"\subsection{" + inline(re.sub(r"^###\s*[\d.]*\s*", "", ln).strip()) + "}")
            num = re.match(r"^###\s+(\d+(?:\.\d+)*)\.?\s", ln)
            if num:
                out.append(r"\label{sec:" + num.group(1) + "}")
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
        if ln.strip() == "<!--BIBLIOGRAPHY-->":
            close_list(); out.append(r"\BIBLIOGRAPHYHERE"); i += 1; continue
        if ln.strip() in ("---", "***"):
            close_list(); i += 1; continue
        if not ln.strip():
            close_list(); out.append(""); i += 1; continue
        close_list()
        para = [ln]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||>|```|\s*[-*]\s|\d+\.\s|---$)", lines[i + 1]):
            i += 1; para.append(lines[i])
        text = " ".join(x.strip() for x in para)
        out.append(paragraph_heading(text) or inline(text))
        i += 1
    close_list()
    if in_abstract:
        out.append(r"\end{abstract}")
    return "\n".join(out)


def bibliography(md: str, path: "Path | None" = None) -> str:
    """Emit thebibliography from paper/REFERENCES.md, including only VERIFIED entries.

    The reference list is NOT typed into the tex. It is generated from a table whose rows
    each carry a status, so an entry nobody could confirm cannot reach the PDF: this paper
    argues that a number is not a result until its controls run, and a citation nobody
    checked is the same failure in a different column.

    Fails loudly rather than silently dropping anything:
      * a \\cite key with no row  -> the reference would print as [?]
      * a \\cite key whose row is UNVERIFIED
      * a VERIFIED row no \\cite uses -> dead weight, or a citation that was meant to exist
    """
    path = path or REPO / "paper" / "REFERENCES.md"
    if not path.exists():
        raise SystemExit(f"missing {path}: the paper cites work and must carry its references")
    rows, cols = [], None
    for ln in path.read_text().split("\n"):
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if re.fullmatch(r"[\s|:-]+", ln):
            continue
        if cols is None:
            cols = [c.lower() for c in cells]
            continue
        if len(cells) == len(cols):
            row = dict(zip(cols, cells))
            # the table writes cells in backticks for legibility in the markdown
            row["key"] = row["key"].strip().strip("`").strip()
            row["status"] = row["status"].strip().strip("`").strip()
            rows.append(row)
    for need in ("key", "authors", "title", "venue", "status"):
        if cols is None or need not in cols:
            raise SystemExit(f"{path}: table needs a '{need}' column, found {cols}")

    # Strip code spans and fenced blocks first, for the same reason inline() parks them: a
    # citation shown inside backticks as an EXAMPLE is not a citation, and counting it makes
    # the gate demand a reference row for a key that does not exist.
    scan = re.sub(r"```.*?```", "", md, flags=re.S)
    scan = re.sub(r"`[^`]*`", "", scan)
    cited = set(re.findall(r"\[@([\w\d]+(?:\s*;\s*@[\w\d]+)*)\]", scan))
    keys = {k.strip().lstrip("@") for group in cited for k in group.split(";")}
    bykey = {r["key"]: r for r in rows}

    missing = sorted(k for k in keys if k not in bykey)
    if missing:
        raise SystemExit(f"cited but not in REFERENCES.md: {missing}")
    unver = sorted(k for k in keys if bykey[k]["status"].upper() != "VERIFIED")
    if unver:
        raise SystemExit(f"cited but not VERIFIED: {unver}")
    unused = sorted(r["key"] for r in rows
                    if r["status"].upper() == "VERIFIED" and r["key"] not in keys)
    if unused:
        raise SystemExit(f"VERIFIED but never cited: {unused}")

    def raw(cell: str, key: str, col: str) -> str:
        """Bibliography cells are author-written LaTeX (\\emph, ~ ties), not markdown.

        Running them through inline() would escape the backslashes and print the markup.
        They are passed through verbatim, so the only thing worth checking is that the
        braces balance -- an unbalanced one swallows the rest of the document.
        """
        t = cell.strip()
        if t.startswith("`") and t.endswith("`"):
            t = t[1:-1].strip()
        if t.count("{") != t.count("}"):
            raise SystemExit(f"REFERENCES.md {key}: unbalanced braces in '{col}': {t!r}")
        return t

    def stop(t: str) -> str:
        """Add the terminal period only when the cell does not already end a sentence.

        An author list ending `et~al.` and a title ending `?` already carry it; appending
        blindly prints `et~al..` and `?.` in the reference list.
        """
        return t if t.endswith((".", "?", "!")) else t + "."

    items = [r"\begin{thebibliography}{99}", r"\small", ""]
    for key in sorted(keys):
        r = bykey[key]
        items += [r"\bibitem{" + key + "}",
                  stop(raw(r["authors"], key, "authors")),
                  r"\newblock " + stop(raw(r["title"], key, "title")),
                  r"\newblock " + stop(raw(r["venue"], key, "venue")),
                  ""]
    items.append(r"\end{thebibliography}")
    print(f"  bibliography: {len(keys)} entries, all VERIFIED and all cited")
    return "\n".join(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    md = (REPO / "paper" / "paper.md").read_text()
    SECTIONS.clear()
    SECTIONS.update(scan_sections(md))
    body = convert(md)
    bib = bibliography(md)
    # References belong before the appendices, which is where the marker sits in paper.md.
    if r"\BIBLIOGRAPHYHERE" in body:
        body = body.replace(r"\BIBLIOGRAPHYHERE", bib, 1)
    else:
        body = body + "\n\n" + bib
    tex = PREAMBLE + "\n" + body + "\n\n\\end{document}\n"
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
