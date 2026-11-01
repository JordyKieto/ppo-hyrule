"""The markdown-to-LaTeX converter's cross-references, citations and bibliography gate.

Every assertion here was written by breaking the thing it checks and watching it fail
first. The bibliography checks matter most: this paper argues that a number is not a
result until its controls run, so a citation nobody verified must not be able to reach
the PDF by accident.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import md_to_tex as M  # noqa: E402

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}\n      got  {got!r}\n      want {want!r}")
    else:
        print(f"  ok  {name}")


def main():

    M.SECTIONS.clear()
    M.SECTIONS.update({"4.1", "5", "7", "8", "9.1", "12"})

    # --- cross-references ---------------------------------------------------------------
    check("section ref", M.inline("see section 9.1 now"), r"see \S\ref{sec:9.1} now")
    check("two sections", M.inline("see sections 5 and 12"),
          r"see \S\ref{sec:5} and \S\ref{sec:12}")
    check("capitalised", M.inline("Section 7 says"), r"\S\ref{sec:7} says")
    # A number this paper does not have points at a DIFFERENT document and must stay prose.
    # Without the SECTIONS guard this produced a dangling \ref that prints as "??".
    check("foreign section untouched",
          M.inline("the provenance record section 19 carries it"),
          "the provenance record section 19 carries it")

    # --- citations ----------------------------------------------------------------------
    check("one citation", M.inline("PPO [@schulman2017ppo] is used"),
          r"PPO \cite{schulman2017ppo} is used")
    check("two citations", M.inline("both [@a1; @b2] apply"), r"both \cite{a1,b2} apply")

    # --- display equations ---------------------------------------------------------------
    tex = M.convert("$$\nx = y\n$$\n")
    check("equation env", "\\begin{equation}" in tex and "x = y" in tex, True)

    # --- inline math -------------------------------------------------------------------
    # Escaping this does not render wrong, it halts XeTeX with "Missing $ inserted".
    check("inline math passes through", M.inline(r"the term $\gamma\lambda$ here"),
          r"the term $\gamma\lambda$ here")
    check("math with braces", M.inline(r"$\delta_{t+l}$ and `code`"),
          r"$\delta_{t+l}$ and \texttt{code}")

    # Two dollar signs in a sentence about money are not a math span. This exact sentence
    # silently became mathematics and overflowed the margin.
    check("currency is not math",
          "$" in M.inline("$266 for the world model track and $1,381 in total")
          and "\\ensuremath" not in M.inline("$266 and $1,381"), True)
    check("currency untouched", M.inline("it cost $1,381 in total"), r"it cost \$1,381 in total")

    # --- captioned tables ----------------------------------------------------------------
    # The caption is a plain line of text. If it does not attach to the table it does not fail
    # -- it renders as a stray paragraph reading "Table: ..." in the middle of the paper, which
    # is exactly what happened to the appendix table.
    for gap in ("", "\n"):
        tex = M.convert(f"Table: A caption.{gap}\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        label = "blank line" if gap else "adjacent"
        check(f"captioned table, {label}",
              "\\captionof{table}{A caption.}" in tex and "Table:" not in tex, True)

    # --- run-in headings ---------------------------------------------------------------
    # Roughly half the paragraphs in the paper that open with bold are NOT headings: they
    # open mid-sentence on an emphasised quantity. Promoting one of those to a heading cuts
    # the sentence in half, and it still compiles, so nothing else would catch it.
    for text, want in [
        ("**Outcomes.** Three of them.", True),
        ("**The stun chain.** A stun must be manufactured.", True),
        ("**76.2%** of the time it reads zero", False),      # mid-sentence emphasis
        ("**Twenty terms are zeroed** in this preset", False),  # no terminal period
        ("**(M.** Recomputed for this paper.", False),       # an evidence-class marker
        ("**M**", False),                                    # too short, nothing follows
    ]:
        got = M.paragraph_heading(text)
        check(f"run-in heading: {text[:32]!r}", got is not None, want)

    # --- the bibliography gate -----------------------------------------------------------
    TBL = """| key | authors | title | venue | used for | status |
    |---|---|---|---|---|---|
    | good1 | A. One | A Title | Venue, 2020. arXiv:1 | s1 | VERIFIED |
    | shaky | B. Two | B Title | Venue, 2021. arXiv:2 | s2 | UNVERIFIED |
    """


    def bib(md, table=TBL, tmp=REPO / "tests" / ".refs_tmp.md"):
        tmp.write_text(table)
        try:
            return M.bibliography(md, tmp), None
        except SystemExit as e:
            return None, str(e)
        finally:
            tmp.unlink(missing_ok=True)


    out, err = bib("text [@good1] more")
    check("a verified, cited entry is emitted", out is not None and "\\bibitem{good1}" in out, True)

    # positive control: a key with no row would print as [?] in the PDF and nothing else warns
    out, err = bib("text [@missing] more")
    check("cited but absent is refused", err is not None and "missing" in err, True)

    # positive control: the whole point of the status column
    out, err = bib("text [@good1] and [@shaky]")
    check("cited but UNVERIFIED is refused", err is not None and "shaky" in err, True)

    # a verified row nothing cites is either dead weight or a citation that was meant to exist
    out, err = bib("text with no citations at all")
    check("verified but uncited is refused", err is not None and "good1" in err, True)

    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  " + f)
        raise SystemExit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
