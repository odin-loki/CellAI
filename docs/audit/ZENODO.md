# Zenodo upload checklist

**Files to upload**
- `CELLAI_AUDIT_PAPER.pdf`: the paper (build with `python docs/audit/build/build_pdf.py`).
- Optional: a zip of `docs/audit/` (Markdown source, figures, data, scripts) as supplementary material.

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
| Related identifiers | `isSupplementedBy` https://github.com/odin-loki/CellAI ; `references` https://doi.org/10.1038/s41467-024-53922-x (Kukushkin et al., 2024) |

**Before uploading:** Zenodo DOIs are permanent. Re-read the Revision note and the Statements section, and decide whether to also link a GitHub release of the exact commit that contains `docs/audit/`.
