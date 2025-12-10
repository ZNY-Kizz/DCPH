import torch
import clip
from clip.model import CLIP

class TextEncoder(object):
    # Prompt-Augmented Sequence Constructor (PASC)
    def __init__(self, device):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model: CLIP = self.model.eval().requires_grad_(False)

    def encode_with_pseudo_tokens(self, pseudo_tokens: torch.Tensor, num_tokens=1) -> torch.Tensor:
        text = clip.tokenize(["a video of $" for _ in range(pseudo_tokens.size(0))], truncate=True).to(self.device)
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, n_ctx, d_model]
        _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
        cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
        first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        if pseudo_tokens.shape[0] == x.shape[0]:
            if len(pseudo_tokens.shape) == 2:
                pseudo_tokens = pseudo_tokens.unsqueeze(1)
            x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
                x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
        else:
            first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
            rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
            x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
                x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x.to(dtype=torch.float32)


