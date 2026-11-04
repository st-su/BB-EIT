"""
mcp_server.py
=============
MCP server exposing BB-EIT to AI agents (stdio transport).

Tools
-----
  predict_adsorption  — predict protein adsorption (ng cm⁻²) for polymer SMILES.
  design_polymer      — genetic-algorithm inverse design toward a target adsorption.

Usage
-----
python mcp_server.py
"""

import csv
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

mcp = MCPServer("bb-eit")


# ---------------------------------------------------------------------------
# Lazy model loading (heavy imports deferred so the server starts instantly)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ensemble():
    from src.model import load_ensemble_models
    # stdout is the MCP channel; route any model-loading output to stderr
    with redirect_stdout(sys.stderr):
        tokenizer, _, models = load_ensemble_models()
    return tokenizer, models


@lru_cache(maxsize=1)
def _training_monomers() -> List[str]:
    path = ROOT / "data" / "Proteins_Data_w_Predicted_Value_Origin.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return sorted({row["SMILES"] for row in csv.DictReader(fh) if row["SMILES"]})


def _is_valid(smiles: str) -> bool:
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_adsorption(
    smiles: List[str],
    pi: float,
    thickness: float,
    ca: float,
    zeta: float,
    mw: float,
) -> List[dict]:
    """Predict protein adsorption on a polymer brush with the BB-EIT 5-fold ensemble.

    Args:
        smiles: Monomer SMILES of the polymer brush, e.g. "C=C(C)C(=O)OCCO" (HEMA).
        pi: Protein isoelectric point (training range 4.7–11).
        thickness: Brush thickness in nm (0.6–91).
        ca: Surface water contact angle in degrees (12–83).
        zeta: Surface zeta potential in mV (-57–48).
        mw: Protein molecular weight in kDa (14–340).

    Returns one entry per SMILES: ensemble-mean adsorption and the std across
    the 5 folds (ng/cm²), or an error for invalid SMILES.
    """
    import numpy as np
    import torch
    from src.model import device

    tokenizer, models = _ensemble()
    features = torch.tensor([[pi, thickness, ca, zeta, mw]], dtype=torch.float32, device=device)

    results = []
    for smi in smiles:
        if not _is_valid(smi):
            results.append({"smiles": smi, "error": "invalid SMILES"})
            continue
        enc = tokenizer(smi, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            x = models[0](enc["input_ids"], enc["attention_mask"], features,
                          extract_layer="concatenate")
            preds = [m.fc2(m.relu(m.fc1(x))).item() for m in models]
        results.append({
            "smiles": smi,
            "adsorption_ng_cm2": round(float(np.mean(preds)), 3),
            "std_ng_cm2": round(float(np.std(preds)), 3),
        })
    return results


@mcp.tool()
def design_polymer(
    target: float,
    pi: float,
    thickness: float,
    ca: float,
    zeta: float,
    mw: float,
    seed_smiles: Optional[List[str]] = None,
    population: int = 50,
    generations: int = 30,
    top_k: int = 5,
) -> dict:
    """Inverse design: evolve monomer SMILES toward a target protein adsorption.

    Runs a genetic algorithm over SMILES, scored by the BB-EIT ensemble, under
    fixed protein/surface conditions (same units as predict_adsorption).
    Takes roughly a minute on CPU with the defaults.

    Args:
        target: Target adsorption in ng/cm² (training data spans 0–1000).
        pi, thickness, ca, zeta, mw: Protein/surface conditions.
        seed_smiles: Starting monomers; defaults to the monomers in the training set.
        population: GA population size.
        generations: Maximum number of GA generations.
        top_k: Number of distinct best candidates to return.
    """
    from inverse_design import ProteinCondition, genetic_algorithm

    seeds = [s for s in (seed_smiles or _training_monomers()) if _is_valid(s)]
    if not seeds:
        raise ValueError("No valid seed SMILES provided.")

    tokenizer, models = _ensemble()
    with redirect_stdout(sys.stderr):
        history = genetic_algorithm(
            seed_smiles=seeds,
            protein_condition=ProteinCondition(pI=pi, thickness=thickness, ca=ca, zeta=zeta, mw=mw),
            tokenizer=tokenizer,
            ensemble_models=models,
            target=target,
            population_size=population,
            generations=generations,
            verbose=False,
        )

    best = {}
    for r in sorted(history, key=lambda r: r.loss):
        best.setdefault(r.smiles, r)
    return {
        "target_ng_cm2": target,
        "generations_run": len(history),
        "candidates": [
            {
                "smiles": r.smiles,
                "predicted_ng_cm2": round(r.predicted_adsorption, 3),
                "abs_error": round(abs(r.predicted_adsorption - target), 3),
            }
            for r in list(best.values())[:top_k]
        ],
    }


if __name__ == "__main__":
    mcp.run()
