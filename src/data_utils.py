import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

CHEMBERT_MODEL = "seyonec/ChemBERTa-zinc-base-v1" 


class PolymerDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.smiles = dataframe["SMILES"].values
        self.features = dataframe[["Protein Charge (pI)", "Thickness (nm)", 
                                  "Predicted CA (deg)", "Predicted Zeta (mV)", 
                                  "Protein MW / kDa"]].values
        self.amount = dataframe["Protein adsorption / ng cm-2"].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.smiles[idx], return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=128)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        protein_features = torch.tensor(self.features[idx], dtype=torch.float32)
        amount = torch.tensor(self.amount[idx], dtype=torch.float32)
        
        return input_ids, attention_mask, protein_features, amount


def load_chemberta_tokenizer():
    return AutoTokenizer.from_pretrained(CHEMBERT_MODEL)

if __name__ == '__main__':
    print("data_utils.py: PolymerDataset class and tokenizer utility defined.")