import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        B, L, E = query.shape
        S = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)

        q = q.view(B, self.num_heads, L, self.head_dim)
        k = k.view(B, self.num_heads, S, self.head_dim)
        v = v.view(B, self.num_heads, S, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)
        
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.img2graph = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim//64
        )
      
        self.post_norm=nn.LayerNorm(embed_dim)

    def forward(self, img, graph):
        B, L, C = img.shape
        graph = graph.unsqueeze(0).repeat(B, 1, 1).contiguous()

        
        img_features, _ = self.img2graph(
            query=img,
            key=graph,
            value=graph
        )
        return img, self.post_norm(img_features.mean(1,keepdim=True))
        
        
    
def build_cross_module(config):
    return CrossAttention(config.mm_hidden_size)