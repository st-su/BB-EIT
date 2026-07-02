"""
reverse_design.py
=================
Inverse design of polymer SMILES for a target protein adsorption amount.

Two strategies are exposed:
  1. latent_gradient  — Adam-based optimisation in the concatenated feature
                        space (ChemBERTa CLS token + protein features).
                        Fast; produces latent vectors that must be decoded.
  2. genetic          — Genetic algorithm over discrete SMILES strings.
                        Slower; yields directly valid, evaluatable molecules.

Usage (CLI)
-----------
python inverse_design.py \
    --strategy genetic \
    --target 50.0 \
    --pi 5.5 --thickness 20 --ca 40 --zeta -10 --mw 66 \
    --population 100 --generations 200 \
    --seed_smiles seeds.txt \
    --output results.csv
"""

import argparse
import random
import copy
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel

from model import (
    ChemBERTRegressor,
    ensemble_predictor_for_features,
    load_ensemble_models,
    CHEMBERT_MODEL,
    device,
)

# ---------------------------------------------------------------------------
# Protein condition container
# ---------------------------------------------------------------------------

@dataclass
class ProteinCondition:
    """Five protein/surface descriptors consumed by ChemBERTRegressor."""
    pI: float           # Protein charge (isoelectric point)
    thickness: float    # Surface thickness (nm)
    ca: float           # Predicted contact angle (deg)
    zeta: float         # Predicted zeta potential (mV)
    mw: float           # Protein MW (kDa)

    def to_array(self) -> np.ndarray:
        return np.array([self.pI, self.thickness, self.ca, self.zeta, self.mw],
                        dtype=np.float32)


# ---------------------------------------------------------------------------
# Ensemble scoring helpers
# ---------------------------------------------------------------------------

def score_smiles_batch(
    smiles_list: List[str],
    protein_condition: ProteinCondition,
    tokenizer,
    ensemble_models: List[ChemBERTRegressor],
) -> np.ndarray:
    """
    Return predicted adsorption (ng cm⁻²) for each SMILES in *smiles_list*.
    Invalid SMILES receive a penalty score of +inf.
    """
    scores = np.full(len(smiles_list), np.inf, dtype=np.float64)
    protein_arr = protein_condition.to_array()

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            encoding = tokenizer(
                smi,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            protein_t = torch.tensor(protein_arr, dtype=torch.float32).to(device)

            # Obtain CLS-pooled embedding
            with torch.no_grad():
                chembert_out = ensemble_models[0].chembert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_emb = chembert_out.last_hidden_state[:, 0, :].squeeze(0)

            concat_vec = torch.cat([cls_emb, protein_t]).unsqueeze(0).cpu().numpy()
            pred = ensemble_predictor_for_features(concat_vec, ensemble_models)
            scores[idx] = float(np.squeeze(pred))
        except Exception:
            pass

    return scores


def objective(score: float, target: float) -> float:
    """Scalar loss: squared deviation from target adsorption."""
    return (score - target) ** 2


# ---------------------------------------------------------------------------
# Strategy 1 — Latent-space gradient optimisation
# ---------------------------------------------------------------------------

def latent_gradient_optimise(
    seed_smiles: str,
    protein_condition: ProteinCondition,
    tokenizer,
    ensemble_models: List[ChemBERTRegressor],
    target: float,
    lr: float = 1e-3,
    steps: int = 500,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[float]]:
    """
    Optimise a continuous latent vector (CLS embedding || protein features)
    toward *target* using Adam.

    Returns
    -------
    optimised_vector : np.ndarray, shape (773,)
        The first 768 dims correspond to the ChemBERTa embedding space;
        the last 5 are protein features.  Feed into a latent decoder /
        nearest-neighbour retrieval over a SMILES library.
    loss_curve : list[float]
    """
    protein_arr = protein_condition.to_array()
    encoding = tokenizer(
        seed_smiles, return_tensors="pt",
        padding="max_length", truncation=True, max_length=128,
    )
    with torch.no_grad():
        cls_emb = ensemble_models[0].chembert(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        ).last_hidden_state[:, 0, :].squeeze(0)

    # Only the SMILES-derived portion is a free variable; protein features fixed
    z = cls_emb.detach().clone().requires_grad_(True)
    protein_t = torch.tensor(protein_arr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([z], lr=lr)
    target_t = torch.tensor(target, dtype=torch.float32, device=device)

    loss_curve = []
    for step in range(steps):
        optimizer.zero_grad()
        concat = torch.cat([z, protein_t]).unsqueeze(0)

        # Average ensemble predictions with gradients enabled through fc layers
        preds = []
        for m in ensemble_models:
            x = m.relu(m.fc1(concat))
            preds.append(m.fc2(x).squeeze())
        pred_mean = torch.stack(preds).mean()

        loss = (pred_mean - target_t) ** 2
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())

        if verbose and step % 100 == 0:
            print(f"  step {step:4d}  pred={pred_mean.item():.3f}  loss={loss.item():.4f}")

    return torch.cat([z.detach(), protein_t]).cpu().numpy(), loss_curve


# ---------------------------------------------------------------------------
# Strategy 2 — Genetic algorithm over SMILES
# ---------------------------------------------------------------------------

# A minimal set of polymer-relevant fragment mutations
_FRAGMENT_MUTATIONS = [
    ("C", "CC"), ("CC", "CCC"), ("O", "OC"), ("N", "NC"),
    ("c1ccccc1", "c1ccncc1"), ("C(=O)", "C(=O)O"),
    ("OC", "OCC"), ("CC", "C(C)C"), ("c1ccccc1", "c1ccc(O)cc1"),
    ("N", "NC(=O)"), ("S", "SC"), ("F", "Cl"),
]

def _mutate_smiles(smi: str, n_attempts: int = 10) -> str:
    """Apply a random fragment substitution; return original if all attempts fail."""
    for _ in range(n_attempts):
        old_frag, new_frag = random.choice(_FRAGMENT_MUTATIONS)
        if old_frag in smi:
            candidate = smi.replace(old_frag, new_frag, 1)
            if Chem.MolFromSmiles(candidate) is not None:
                return candidate
    return smi


def _crossover_smiles(smi_a: str, smi_b: str) -> str:
    """
    Single-point crossover at the token level.
    Falls back to one of the parents when the offspring is invalid.
    """
    tokens_a = list(smi_a)
    tokens_b = list(smi_b)
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return random.choice([smi_a, smi_b])
    cut_a = random.randint(1, len(tokens_a) - 1)
    cut_b = random.randint(1, len(tokens_b) - 1)
    offspring = "".join(tokens_a[:cut_a] + tokens_b[cut_b:])
    return offspring if Chem.MolFromSmiles(offspring) is not None else random.choice([smi_a, smi_b])


@dataclass
class GAResult:
    generation: int
    smiles: str
    predicted_adsorption: float
    loss: float


def genetic_algorithm(
    seed_smiles: List[str],
    protein_condition: ProteinCondition,
    tokenizer,
    ensemble_models: List[ChemBERTRegressor],
    target: float,
    population_size: int = 50,
    generations: int = 100,
    elite_fraction: float = 0.2,
    mutation_rate: float = 0.6,
    verbose: bool = True,
) -> List[GAResult]:
    """
    Run a steady-state GA to evolve SMILES toward *target* adsorption.

    Returns a list of GAResult (one per generation's best individual).
    """
    # Initialise population
    population = list(seed_smiles)
    while len(population) < population_size:
        population.append(random.choice(seed_smiles))
    population = population[:population_size]

    history: List[GAResult] = []
    n_elite = max(1, int(elite_fraction * population_size))

    for gen in range(generations):
        scores = score_smiles_batch(population, protein_condition, tokenizer, ensemble_models)
        losses = np.array([objective(s, target) if not np.isinf(s) else 1e9 for s in scores])

        # Sort by ascending loss
        ranked_idx = np.argsort(losses)
        population = [population[i] for i in ranked_idx]
        losses = losses[ranked_idx]
        scores = scores[ranked_idx]

        best = GAResult(
            generation=gen,
            smiles=population[0],
            predicted_adsorption=float(scores[0]),
            loss=float(losses[0]),
        )
        history.append(best)

        if verbose and gen % 10 == 0:
            print(f"  gen {gen:4d}  best_pred={best.predicted_adsorption:.3f} "
                  f"  loss={best.loss:.4f}  SMILES={best.smiles[:60]}")

        if best.loss < 1e-4:
            print("  Convergence criterion met.")
            break

        # Elitism: keep top fraction unchanged
        elites = population[:n_elite]
        new_population = elites[:]

        # Fill remainder with crossover + mutation
        while len(new_population) < population_size:
            p1, p2 = random.choices(elites, k=2)
            child = _crossover_smiles(p1, p2)
            if random.random() < mutation_rate:
                child = _mutate_smiles(child)
            if Chem.MolFromSmiles(child) is not None:
                new_population.append(child)

        population = new_population

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inverse polymer design via BB-EIT ensemble")
    p.add_argument("--strategy", choices=["latent_gradient", "genetic"], default="genetic")
    p.add_argument("--target", type=float, required=True,
                   help="Target protein adsorption in ng cm⁻²")
    p.add_argument("--pi",        type=float, required=True, help="Protein pI")
    p.add_argument("--thickness", type=float, required=True, help="Coating thickness (nm)")
    p.add_argument("--ca",        type=float, required=True, help="Contact angle (deg)")
    p.add_argument("--zeta",      type=float, required=True, help="Zeta potential (mV)")
    p.add_argument("--mw",        type=float, required=True, help="Protein MW (kDa)")
    p.add_argument("--seed_smiles", type=str, default=None,
                   help="Path to newline-separated seed SMILES file")
    # GA-specific
    p.add_argument("--population",   type=int, default=50)
    p.add_argument("--generations",  type=int, default=100)
    p.add_argument("--mutation_rate", type=float, default=0.6)
    # Gradient-specific
    p.add_argument("--lr",    type=float, default=1e-3)
    p.add_argument("--steps", type=int,   default=500)
    p.add_argument("--output", type=str, default="reverse_design_results.csv")
    return p.parse_args()


_DEFAULT_SEEDS = [
    "CC(=O)OCC", "CCOC(=O)C", "c1ccccc1OCC",
    "OCC(O)CO",  "CCNCC",      "CC(C)OC(=O)C",
]

def main():
    args = parse_args()

    condition = ProteinCondition(
        pI=args.pi, thickness=args.thickness,
        ca=args.ca, zeta=args.zeta, mw=args.mw,
    )

    # Load seeds
    if args.seed_smiles:
        with open(args.seed_smiles) as fh:
            seeds = [ln.strip() for ln in fh if ln.strip()]
    else:
        seeds = _DEFAULT_SEEDS

    print(f"Target adsorption : {args.target} ng cm⁻²")
    print(f"Protein condition : pI={condition.pI}, t={condition.thickness} nm, "
          f"CA={condition.ca}°, ζ={condition.zeta} mV, MW={condition.mw} kDa")
    print(f"Seeds loaded      : {len(seeds)}")
    print(f"Strategy          : {args.strategy}\n")

    tokenizer, _, ensemble_models = load_ensemble_models()

    if args.strategy == "genetic":
        history = genetic_algorithm(
            seed_smiles=seeds,
            protein_condition=condition,
            tokenizer=tokenizer,
            ensemble_models=ensemble_models,
            target=args.target,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
        )

        with open(args.output, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["generation", "smiles", "predicted_adsorption_ng_cm2", "loss"])
            for r in history:
                writer.writerow([r.generation, r.smiles, r.predicted_adsorption, r.loss])

        best = min(history, key=lambda r: r.loss)
        print(f"\nBest result  →  SMILES: {best.smiles}")
        print(f"               Predicted: {best.predicted_adsorption:.3f} ng cm⁻²  "
              f"(target {args.target})")
        print(f"Results saved to: {args.output}")

    elif args.strategy == "latent_gradient":
        seed = seeds[0]
        print(f"Seed SMILES: {seed}")
        opt_vec, loss_curve = latent_gradient_optimise(
            seed_smiles=seed,
            protein_condition=condition,
            tokenizer=tokenizer,
            ensemble_models=ensemble_models,
            target=args.target,
            lr=args.lr,
            steps=args.steps,
        )
        np.save(args.output.replace(".csv", "_latent_vector.npy"), opt_vec)
        print(f"\nFinal loss: {loss_curve[-1]:.6f}")
        print(f"Optimised latent vector saved → "
              f"{args.output.replace('.csv', '_latent_vector.npy')}")
        print("Decode via nearest-neighbour retrieval over a SMILES library "
              "or a trained VAE decoder.")


if __name__ == "__main__":
    main()