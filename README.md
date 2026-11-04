# BB-EIT

**A generalized model for predicting protein adsorption on polymer brushes using augmented chemical embeddings.**

*ACS Applied Materials & Interfaces* (2026) 18 (14): 20976–20988.. DOI: [10.1021/acsami.5c25223](https://pubs.acs.org/doi/10.1021/acsami.5c25223)

BB-EIT (Bio-interface BERT Encoder for Interaction Translation) embeds monomer SMILES with the pretrained [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) and combines them with surface and protein descriptors to predict adsorbed protein amount (ng/cm²), generalizing to unseen polymers and proteins.

**Inputs:** monomer SMILES, protein pI, brush thickness (nm), water contact angle (°), zeta potential (mV), protein MW (kDa).

## Setup

```bash
git lfs pull                      # fetch the 5-fold ensemble weights
pip install -r requirements.txt
```

## Use

**Inverse design (CLI)**

```bash
python inverse_design.py --strategy genetic --target 50 \
    --pi 5.5 --thickness 20 --ca 40 --zeta -10 --mw 66 --output results.csv
```

`--strategy genetic` evolves valid SMILES; `--strategy latent_gradient` optimizes in embedding space and returns a latent vector.

**From an AI agent (MCP)**

`mcp_server.py` exposes two tools, `predict_adsorption` and `design_polymer`. Claude Code picks it up automatically from `.mcp.json`; for other clients, register:

```json
{ "mcpServers": { "bb-eit": { "command": "python", "args": ["/path/to/BB-EIT/mcp_server.py"] } } }
```

The agent skill in `.claude/skills/bb-eit/` tells agents when and how to call the model.

## Repository

| Path | Contents |
| --- | --- |
| `BB-EIT.ipynb` | Training and evaluation |
| `BB-EIT_Feature_Importance.ipynb` | SHAP analysis |
| `BB-EIT_UMAP_FC1.ipynb` | UMAP of the FC1 embedding |
| `inverse_design.py` | Inverse design CLI |
| `mcp_server.py`, `.mcp.json` | MCP server |
| `.claude/skills/bb-eit/` | Agent skill |
| `src/` | Model and data utilities |
| `data/`, `models/` | Datasets and pretrained weights |

## Related work

- Protein adsorption ML on polymer brushes — *Langmuir* 2025, 41, 7534. [DOI](https://doi.org/10.1021/acs.langmuir.4c05151)
- Hydrophilicity prediction of polymer brushes — *J. Phys. Chem. B* 2024, 128, 6589. [DOI](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c08422)
