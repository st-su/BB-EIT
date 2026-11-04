---
name: bb-eit
description: Predict protein adsorption (ng/cm²) on polymer brush surfaces from monomer SMILES plus protein/surface descriptors, or inverse-design monomers that hit a target adsorption, using the BB-EIT model. Use for antifouling coatings, biosensor surfaces, polymer–protein interaction screening, or any "how much protein will adsorb on this polymer" question.
---

# BB-EIT: protein adsorption on polymer brushes

BB-EIT is a ChemBERTa-based 5-fold ensemble that predicts how much protein adsorbs on a polymer brush.

## Inputs

| Field | Meaning | Training range |
| --- | --- | --- |
| `smiles` | Monomer SMILES, e.g. `C=C(C)C(=O)OCCO` (HEMA) | methacrylates/acrylamides etc. |
| `pi` | Protein isoelectric point | 4.7–11 |
| `thickness` | Brush thickness (nm) | 0.6–91 |
| `ca` | Surface water contact angle (deg) | 12–83 |
| `zeta` | Surface zeta potential (mV) | -57–48 |
| `mw` | Protein molecular weight (kDa) | 14–340 |

Common proteins: BSA pI 4.7 / 66.5 kDa, lysozyme pI 11 / 14 kDa, fibrinogen pI 5.5 / 340 kDa.

Output is adsorption in ng/cm² (training data 0–1000). Flag predictions outside the ranges above as extrapolation, and report the ensemble std as uncertainty.

## How to call

**Preferred — MCP tools** (server `bb-eit`, configured in `.mcp.json`):

- `predict_adsorption(smiles: list[str], pi, thickness, ca, zeta, mw)` → mean and std per SMILES.
- `design_polymer(target, pi, thickness, ca, zeta, mw, seed_smiles=None, population=50, generations=30, top_k=5)` → best candidate monomers.

**Fallback — Python** (run from the repo root):

```bash
python -c "from mcp_server import predict_adsorption as p; print(p(['C=C(C)C(=O)OCCO'], pi=4.7, thickness=17, ca=48, zeta=-12, mw=66.5))"
```

```bash
python inverse_design.py --strategy genetic --target 20 --pi 4.7 --thickness 17 --ca 48 --zeta -12 --mw 66.5 --output results.csv
```

## Notes

- First call loads ChemBERTa (Hugging Face) and five ~177 MB weight files; it takes a while.
- Weights are stored with Git LFS. If loading fails with an unpickling error, run `git lfs pull`.
- When the user lacks `ca`/`zeta`, ask for them or state the values you assumed.
- `design_polymer` candidates are model suggestions; check synthesizability before recommending them.
