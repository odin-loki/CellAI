"""Build docs/audit/CELLAI_AUDIT_PAPER.pdf from the Markdown source.
pandoc (MathML, self-contained) -> HTML -> Chromium (Playwright) print-to-PDF."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); AUDIT = os.path.dirname(HERE)
src = os.path.join(AUDIT, "CELLAI_AUDIT_PAPER.md"); out_pdf = os.path.join(AUDIT, "CELLAI_AUDIT_PAPER.pdf")
md = open(src, encoding="utf-8").read()
# The YAML front matter carries the title; drop the duplicate H1 used for GitHub rendering.
md = re.sub(r"^# From Cellular Memory.*\n", "", md, count=1, flags=re.M)
tmp_md = os.path.join(AUDIT, "_build_tmp.md"); open(tmp_md, "w", encoding="utf-8").write(md)
html = os.path.join(HERE, "CELLAI_AUDIT_PAPER.html")
try:
    subprocess.run(["pandoc", tmp_md, "-f", "markdown+pipe_tables+tex_math_dollars+yaml_metadata_block-implicit_figures",
                    "-t", "html5", "--standalone", "--embed-resources", "--mathml",
                    "--css", os.path.join(HERE, "paper.css"), "--resource-path", AUDIT,
                    "--metadata", "lang=en-GB", "-o", html], check=True, cwd=AUDIT)
finally:
    os.remove(tmp_md)
# PDF via Node Playwright (pre-installed Chromium).
subprocess.run(["node", os.path.join(HERE, "print_pdf.js"), html, out_pdf], check=True)
print("wrote", out_pdf)
