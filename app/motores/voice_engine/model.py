import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class SLSModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ssl_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)  # 22847 = floor(1024/3)*floor(201/3), para audio de 64600 muestras
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def get_atten_f(self, layer_results):
        poollayer, fullf = [], []
        for layer in layer_results:  # cada layer: (batch, seq, hidden)
            pooled = layer.transpose(1, 2)
            pooled = F.adaptive_avg_pool1d(pooled, 1)
            pooled = pooled.transpose(1, 2)
            poollayer.append(pooled)
            fullf.append(layer.unsqueeze(1))
        return torch.cat(poollayer, dim=1), torch.cat(fullf, dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        outputs = self.ssl_model(x, output_hidden_states=True)
        layer_results = outputs.hidden_states[1:]  # 24 capas, sin la de embeddings
        y0, fullfeature = self.get_atten_f(layer_results)
        y0 = self.sig(self.fc0(y0))
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = torch.sum(fullfeature * y0, 1).unsqueeze(1)
        out = self.selu(self.first_bn(fullfeature))
        out = F.max_pool2d(out, (3, 3))
        out = torch.flatten(out, 1)
        out = self.selu(self.fc1(out))
        out = self.selu(self.fc3(out))
        return self.logsoftmax(out)