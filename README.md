# BB-EIT: A Generalized Prediction Model for Protein Adsorption on Polymer Brushes Using Augmented Chemical Embeddings

**Accepted by:** *ACS Applied Materials & Interfaces*, ASAP. DOI: [10.1021/acsami.5c25223][aim]

[aim]: https://pubs.acs.org/doi/10.1021/acsami.5c25223


**Abstract:**

Precise control of protein adsorption on polymer surfaces is essential in materials science and biomaterial design, with applications in antifouling materials, biosensors, and drug delivery systems. However, the complex interactions between polymers and proteins and the limited availability of high-quality interaction data remain major challenges in polymer informatics. Current approaches often lack the generalizability needed to model diverse polymer–protein systems within a single unified framework, and there is a paucity of comprehensive predictive models capable of handling diverse polymer-protein interactions. To address these challenges, we introduce BB-EIT (Bio-interface BERT Encoder for Interaction Translation), a novel generalized model designed to accurately predict the amount of diverse protein adsorption on polymer brushes. BB-EIT leverages the pretrained ChemBERTa large language model (LLM) architecture using SMILES strings for robust chemical representation and convenient data augmentation through SMILES enumeration. By adapting the pretrained model with an extended layer integrating a comprehensive set of physicochemical and biochemical features, including polymer thickness, water contact angle, and surface charge as well as protein isoelectric point (pI) and size, BB-EIT showed state-of-the-art performance and strong generalizability. The model accurately predicted adsorption behavior in previously unseen polymer and protein systems. This work represents an important step toward data-driven design of biomaterials with tailored protein adsorption properties.


More information can be found in the following articles:

"Machine Learning for Quantitative Prediction of Protein Adsorption on Well-Defined Polymer Brush Surfaces with Diverse Chemical Properties"
*Langmuir*, 2025, 41, 11, 7534–7545.
DOI: [10.1021/acs.langmuir.4c05151][langmuir]

[langmuir]: https://doi.org/10.1021/acs.langmuir.4c05151

"Explainable Prediction of Hydrophilic/Hydrophobic Property of Polymer Brush Surfaces by Chemical Modeling and Machine Learning"
*The Journal of Physical Chemistry B*, 2024, 128, 27, 6589–6597.
DOI: [10.1021/acs.jpcb.3c08422][jpcb]

[jpcb]: https://pubs.acs.org/doi/10.1021/acs.jpcb.3c08422


## Repository Structure

| Path | Description |
| --- | --- |
| `BB-EIT.ipynb` | Main notebook: training and evaluation of the BB-EIT model. |
| `BB-EIT_Feature_Importance.ipynb` | SHAP-based feature-importance analysis. |
| `BB-EIT_UMAP_FC1.ipynb` | UMAP visualization of the learned FC1 embedding space. |
| `inverse_design.py` | Inverse design of polymer SMILES for a target protein adsorption amount. |
| `src/` | Model definition (`model.py`), dataset, and data utilities. |
| `data/` | Training, external validation, and noise-dependency datasets. |
| `models/` | Pretrained 5-fold ensemble weights (`R2D2_All_All_fold_{0..4}.pth`). |
| `requirements.txt` | Python dependencies. |

## Installation

```bash
pip install -r requirements.txt
```

BB-EIT builds on the pretrained [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
model, which is downloaded automatically from the Hugging Face Hub on first use.

## Inverse Design (Working)

`inverse_design.py` searches for polymer SMILES that achieve a target protein
adsorption amount under a fixed set of protein/surface descriptors (pI, coating
thickness, contact angle, zeta potential, and protein MW), scoring candidates
with the pretrained BB-EIT ensemble. Two strategies are available:

- **`genetic`** — a genetic algorithm over discrete SMILES strings. Slower, but
  yields directly valid, evaluatable molecules.
- **`latent_gradient`** — Adam-based optimization in the concatenated feature
  space (ChemBERTa CLS embedding + protein features). Fast; produces a latent
  vector that must be decoded via nearest-neighbour retrieval or a trained VAE
  decoder.

Example:

```bash
python inverse_design.py \
    --strategy genetic \
    --target 50.0 \
    --pi 5.5 --thickness 20 --ca 40 --zeta -10 --mw 66 \
    --population 100 --generations 200 \
    --seed_smiles seeds.txt \
    --output results.csv
```

