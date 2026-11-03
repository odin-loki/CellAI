"""Build the supplementary archive CellAI-audit.zip cited by the paper.

Usage:
  python docs/audit/build/make_archive.py --ideas PATH_TO_IDEAS_CLONE --cypha PATH_TO_CYPHA_CLONE [--out DIR]

The Ideas clone must contain commit 7e24624 (raw results). The Cypha clone needs docs/history/.
Layout of the archive (top folder CellAI-audit/):
  README.md, SHA256SUMS
  paper/                      CELLAI_AUDIT_PAPER.pdf and .md
  repository/CellAI/          working tree of the CellAI repository at HEAD
  repository/CellAI.bundle    full git history (git clone CellAI.bundle CellAI-git)
  provenance/PROVENANCE.md    source repository, commit, original path and SHA-256 of every file below
  provenance/raw_results/     arch-search raw results (Rounds 1-4)
  provenance/original_cell_ai_docs/   original Cell AI design documents (HTML)
  provenance/design_history/  design-history notes and the v2 design dialogue
Not included: the Kukushkin et al. (2024) article (CC BY-NC-ND 4.0; cite the DOI) and third-party
datasets (WikiText-103, CodeSearchNet; obtain from HuggingFace).
"""
import argparse, hashlib, os, re, shutil, subprocess, tempfile, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXED_TIME = (2026, 11, 3, 0, 0, 0)

IDEAS_RAW_COMMIT = "7e24624e0868def9240af40c8863e64682a80e86"
IDEAS_RAW = ["results.json", "results_v2.json", "results_v3.json", "results_v4.json"]
IDEAS_RAW_DIR = "Cell AI/data/local/arch_search/"
IDEAS_MM = "Cell AI/data/local/multimodal_results.json"
CYPHA_FILES = ["docs/history/eras/00-prototypes.md", "docs/history/TIMELINE.md",
               "docs/history/archive/prototypes/Cell AI v2 Thinking Convo Log.txt"]


def sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True).stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideas", required=True); ap.add_argument("--cypha", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, ".."))
    a = ap.parse_args()
    md = open(os.path.join(REPO, "docs/audit/CELLAI_AUDIT_PAPER.md"), encoding="utf-8").read()
    version = re.search(r'^version:\s*"([^"]+)"', md, re.M).group(1)
    name = "CellAI-audit"
    tmp = tempfile.mkdtemp(); root = os.path.join(tmp, name)
    os.makedirs(os.path.join(root, "repository")); os.makedirs(os.path.join(root, "paper"))
    prov = os.path.join(root, "provenance")
    for d in ("raw_results", "original_cell_ai_docs", "design_history"):
        os.makedirs(os.path.join(prov, d))

    # 1. Repository: working tree at HEAD + full history bundle.
    head = sh("git", "rev-parse", "HEAD", cwd=REPO).decode().strip()
    tar = sh("git", "archive", "--format=tar", "--prefix=CellAI/", "HEAD", cwd=REPO)
    subprocess.run(["tar", "-x", "-C", os.path.join(root, "repository")], input=tar, check=True)
    bundle = os.path.join(root, "repository", "CellAI.bundle")
    sh("git", "bundle", "create", bundle, "HEAD", cwd=REPO)
    sh("git", "bundle", "verify", bundle, cwd=REPO)

    # 2. Paper.
    for f in ("CELLAI_AUDIT_PAPER.pdf", "CELLAI_AUDIT_PAPER.md"):
        shutil.copy(os.path.join(REPO, "docs/audit", f), os.path.join(root, "paper", f))

    # 3. Provenance.
    rows = []
    ideas_head = sh("git", "rev-parse", "HEAD", cwd=a.ideas).decode().strip()
    cypha_head = sh("git", "rev-parse", "HEAD", cwd=a.cypha).decode().strip()
    for f in IDEAS_RAW + [None]:
        src = IDEAS_RAW_DIR + f if f else IDEAS_MM
        dst = os.path.join(prov, "raw_results", os.path.basename(src))
        open(dst, "wb").write(sh("git", "show", f"{IDEAS_RAW_COMMIT}:{src}", cwd=a.ideas))
        rows.append((dst, "author's research-notes repository (Ideas)", IDEAS_RAW_COMMIT, src))
    site = os.path.join(a.ideas, "docs/site/Cell AI")
    for dp, _, fs in os.walk(site):
        for f in sorted(fs):
            s = os.path.join(dp, f); rel = os.path.relpath(s, site)
            dst = os.path.join(prov, "original_cell_ai_docs", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(s, dst)
            rows.append((dst, "author's research-notes repository (Ideas)", ideas_head, "docs/site/Cell AI/" + rel))
    s = os.path.join(a.ideas, "docs/topics/cell-ai.html"); dst = os.path.join(prov, "original_cell_ai_docs", "topic_index_cell-ai.html")
    shutil.copy(s, dst); rows.append((dst, "author's research-notes repository (Ideas)", ideas_head, "docs/topics/cell-ai.html"))
    for f in CYPHA_FILES:
        dst = os.path.join(prov, "design_history", os.path.basename(f))
        shutil.copy(os.path.join(a.cypha, f), dst)
        rows.append((dst, "author's Cypha repository", cypha_head, f))

    with open(os.path.join(prov, "PROVENANCE.md"), "w", encoding="utf-8") as fh:
        fh.write("# Provenance of archived source material\n\n"
                 "Every file under `provenance/` is copied unchanged from the author's own repositories. "
                 "Commit hashes identify the exact source revision.\n\n"
                 "| Archive path | Source | Commit | Original path | SHA-256 |\n|---|---|---|---|---|\n")
        for dst, src, commit, orig in rows:
            fh.write(f"| `{os.path.relpath(dst, root)}` | {src} | `{commit[:12]}` | `{orig}` | `{sha256(dst)[:16]}…` |\n")
        fh.write(f"\nCellAI repository: HEAD `{head}`; the bundle contains the full history, "
                 "including commits a109790, eb18003, 429615e and 7be5113 cited in the paper.\n")

    # 4. README.
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# {name}

Supplementary archive for "From Cellular Memory to Language Modelling: An Independent Profile,
Biological Assessment and Audit of CellAI" (version {version}). The paper cites files by their paths in
this archive.

| Path | Content |
|---|---|
| `paper/` | The paper (PDF and Markdown source) |
| `repository/CellAI/` | The CellAI repository at commit `{head[:7]}`: code under audit, the original papers, and `docs/audit/` (this study's scripts, data and figures) |
| `repository/CellAI.bundle` | Full git history. Restore with `git clone CellAI.bundle CellAI-git` |
| `provenance/raw_results/` | Raw architecture-search results for Rounds 1-4 (E4-E26) |
| `provenance/original_cell_ai_docs/` | Original Cell AI design documents |
| `provenance/design_history/` | Design-history notes and the v2 design dialogue |
| `provenance/PROVENANCE.md` | Source revision and checksum of every provenance file |
| `SHA256SUMS` | Checksums of every file (`sha256sum -c SHA256SUMS`) |

Reproduction: see Appendix A of the paper; run commands from `repository/CellAI/`.

Not included: the Kukushkin et al. (2024) article (CC BY-NC-ND 4.0; doi:10.1038/s41467-024-53922-x)
and third-party datasets (WikiText-103, CodeSearchNet; available from HuggingFace).

Licences: the CellAI code is AGPL-3.0-or-later with a commercial option (`repository/CellAI/LICENSE`,
`COMMERCIAL-LICENCE.md`). The paper text and figures: see the paper's Statements section.
""")

    # 5. Checksums and deterministic zip.
    files = sorted(os.path.relpath(os.path.join(dp, f), root) for dp, _, fs in os.walk(root) for f in fs)
    with open(os.path.join(root, "SHA256SUMS"), "w") as fh:
        for f in files:
            fh.write(f"{sha256(os.path.join(root, f))}  {f}\n")
    files.append("SHA256SUMS")
    out = os.path.join(os.path.abspath(a.out), name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            zi = zipfile.ZipInfo(f"{name}/{f}", date_time=FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, open(os.path.join(root, f), "rb").read())
    shutil.rmtree(tmp)
    print(out, os.path.getsize(out), "bytes,", len(files), "files")


if __name__ == "__main__":
    main()
