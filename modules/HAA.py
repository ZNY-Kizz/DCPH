import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int):
        super(TemporalAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        x = x.permute(1, 0, 2)
        att_output, _ = self.self_attention(x, x, x)
        att_output = self.layer_norm(att_output + x)
        att_output = att_output.permute(1, 0, 2)
        return att_output

class VideoSequenceModel(nn.Module):
    # Hierarchical Attention Aggregation (HAA)
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, max_sequence_length: int):
        super(VideoSequenceModel, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
        )
        self.positional_encoding = nn.Embedding(max_sequence_length, input_dim)
        self.temporal_attention = nn.ModuleList([TemporalAttention(hidden_dim) for _ in range(4)])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
        )
        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, frames_features):
        batch_size, seq_len, feature_dim = frames_features.size()
        # Adding positional encoding
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(frames_features.device)
        positional_encoding = self.positional_encoding(positions)
        frames_features = frames_features + positional_encoding
        frames_embeddings = self.embedding(frames_features)
        for i,l in enumerate(self.temporal_attention):
            frames_embeddings = self.temporal_attention[i](frames_embeddings)
        video_features = self.pooling(frames_embeddings.permute(0, 2, 1)).squeeze(-1)
        video_features = self.fc(video_features)
        return video_features
    
if __name__ == "__main__":
    input_dim = 512
    model = VideoSequenceModel(input_dim, input_dim, 0.5, 16)
    frames_features = torch.randn([5, 16, 512]).to(device='cpu')
    video_features = model(frames_features)
    print("Video Features Size: ", video_features.size())


