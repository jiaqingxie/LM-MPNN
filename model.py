import torch.nn as nn
import torch
class ChemBERTaForPropertyPrediction(nn.Module):
    def __init__(self, chemberta_model):
        super().__init__()
        self.chemberta = chemberta_model
        self.regressor = nn.Linear(chemberta_model.config.hidden_size, 19)  # Adjust the output size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.chemberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # h = torch.Tensor([t[0] for t in outputs.hidden_states])
        # return self.regressor(h)
