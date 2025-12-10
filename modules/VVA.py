import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from clip.model import CLIP

class VisualAdapter(nn.Module):
    # Visual Vocabulary Adapter (VVA)
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, video_features):
        residual = video_features
        pseudo_tokens = self.layers(video_features)
        pseudo_tokens = F.relu(pseudo_tokens + residual)
        return pseudo_tokens

if __name__ == "__main__":
    input_dim = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # 'ViT-B/32', 'ViT-L/14'
    model: CLIP = model.eval().requires_grad_(False)
    input_dim = model.visual.output_dim
    embedding_dim = model.token_embedding.embedding_dim
    adapter = VisualAdapter(input_dim, 4 * input_dim, embedding_dim, 0.5)
    video_features = torch.randn([5, 512]).to(device='cpu')
    pseudo_tokens = adapter(video_features)
    print("Pseudo Tokens Size: ", pseudo_tokens.size())



