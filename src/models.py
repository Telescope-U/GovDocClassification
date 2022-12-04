from torch import nn
import torch
from torch.nn import functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        x = self.embedding(src)
        #         x = x.view(-1, 500*512)
        x = F.relu(self.linear(x))
        prediction = torch.sigmoid(self.out(x))
        return prediction
