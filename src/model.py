import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

CHEMBERT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

class ChemBERTRegressor(nn.Module):
    def __init__(self, chembert_model, num_features=5, hidden_dim=256):
        super(ChemBERTRegressor, self).__init__()
        self.chembert = chembert_model
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(768 + num_features, self.hidden_dim) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids, attention_mask, protein_features, extract_layer='output'):
        with torch.no_grad():
            outputs = self.chembert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  
        holistic_feature_vector = torch.cat((pooled_output, protein_features), dim=1)
        
        if extract_layer == 'concatenate':
            return holistic_feature_vector.detach() 

        fc1_output = self.relu(self.fc1(holistic_feature_vector))
        
        if extract_layer == 'fc1':
            return fc1_output.detach()

        return self.fc2(fc1_output).squeeze()

def load_ensemble_models(num_folds=5, device=device):
    print("Loading ChemBERTa backbone from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERT_MODEL)
    chembert = AutoModel.from_pretrained(CHEMBERT_MODEL)
    chembert.to(device)
    for param in chembert.parameters():
        param.requires_grad = False
    
    loaded_models = []
    print("Loading BB-EIT ensemble weights...")
    for i in range(num_folds):
        model = ChemBERTRegressor(chembert)
        model.load_state_dict(torch.load(f"models/R2D2_All_All_fold_{i}.pth", map_location=device))
        model.to(device)
        model.eval()
        loaded_models.append(model)
        
    return tokenizer, chembert, loaded_models

def ensemble_predictor_for_features(input_tensor, ensemble_models):
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
    
    all_predictions = []
    
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            x = model.relu(model.fc1(input_tensor))
            output = model.fc2(x).squeeze()
            
        all_predictions.append(output.cpu().numpy())
    return np.mean(all_predictions, axis=0)

if __name__ == '__main__':
    print("model.py: Model classes and utilities defined successfully.")
