# Zenodo upload checklist

**Files to upload (both)**
- `CELLAI_AUDIT_PAPER.pdf`: the paper (build with `python docs/audit/build/build_pdf.py`).
- `CellAI-audit-v1.1.zip`: the supplementary archive the paper cites (build with
  `python docs/audit/build/make_archive.py --ideas <Ideas clone> --cypha <Cypha clone>`). It contains the
  repository with full git history, this study's scripts and data, and provenance material, so the record
  does not depend on any GitHub repository.

**Suggested metadata** (machine-readable copy: `zenodo_metadata.json`)

| Field | Value |
|---|---|
| Resource type | Publication → Report (or Preprint) |
| Title | From Cellular Memory to Language Modelling: An Independent Profile, Biological Assessment and Audit of CellAI |
| Creators | **Fill in yourself** (name, affiliation, ORCID). See the paper's "Authorship and AI assistance" statement and disclose AI assistance per your policy |
| Description | Use the paper's abstract |
| Version | 1.1 |
| Publication date | 2026-11-03 |
| Language | English |
| Keywords | CellAI; biologically inspired AI; cellular memory; massed-spaced effect; spacing effect; Hebbian plasticity; BCM; reaction-diffusion; language modelling; n-gram baselines; evaluation leakage; reproducibility |
| Licence | CC BY 4.0 (suggested for text and figures; code in the repository stays AGPL-3.0-or-later) |
| Related identifiers | `references` https://doi.org/10.1038/s41467-024-53922-x (Kukushkin et al., 2024). No GitHub links: the code and data are in the zip |

**Before uploading:** Zenodo DOIs are permanent. Re-read the Revision note and the Statements section, and check the archive with `sha256sum -c SHA256SUMS` after unzipping.
