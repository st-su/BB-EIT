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

