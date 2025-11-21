import torch
from torch import nn
import math
from utils import pad

class MILNet(nn.Module):
    def __init__(self, encoder, pooling_method='attn', num_heads=1, num_features=512, num_out=1, drop_rate=0, attn_hidden_size=64, debug=False):
        super(MILNet, self).__init__()
        self.encoder = encoder
        self.num_heads = num_heads
        self.num_features = num_features
        self.num_out = num_out
        self.drop_rate = drop_rate
        self.subspace_size = num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.attn_hidden_size = attn_hidden_size
        self.pool = pooling_method
        if self.pool == 'attn':
            self.pool_func = self.attn_pool
            self.attn_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        elif self.pool == 'tanh_attn':
            self.pool_func = self.tanh_attention_pool
            self.V = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size, self.subspace_size))
            self.w = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size))        
        elif self.pool == 'transformer':
            self.pool_func = self.transformer_pool
            self.proj_features = 64
            self.projection = nn.Linear(self.num_features, self.proj_features)
            # Add transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_features,
                nhead=num_heads,
                dim_feedforward=128,
                dropout=drop_rate,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_features))
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'attn', 'max', or 'avg'")
        
        if self.pool == 'transformer':
            self.fc = nn.Sequential(
                nn.Dropout(self.drop_rate),
                nn.Linear(self.proj_features, self.num_out)
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(self.drop_rate),
                nn.Linear(self.num_features, self.num_out)
            )
        self.debug = debug
        
    def forward(self, x, num_imgs):
        # hn_img: the hidden representations of each img
        print("x shape:", x.shape) if self.debug else None
        hn_img = self.encoder(x)
        print("hn_img shape:", hn_img.shape) if self.debug else None
        # expect [L*N, h_dim (num_features)]
        
        # find the hidden representation of entire study and corresponding attn score
        h_study, attn = self.pool_func(hn_img, num_imgs)
        # expect [N, num_heads, subspace_size], [L, N, num_heads]
        
        # feed to fully-connected layer for decoding
        output = self.fc(h_study)
        print("output shape:", output.shape) if self.debug else None
        # expect [N, output size]
        
        return output, attn
    
    def attn_pool(self, hn_img, num_imgs):
        # Calculate attention logits
        hn_img = hn_img.view(-1, self.num_heads, self.subspace_size)
        print("hn shape:", hn_img.shape) if self.debug else None
        # expect [L*N, num_heads, subspace size]
        print("query vector shape:", self.attn_vecs.shape) if self.debug else None
        # expect [num_heads, subspace size]
        
        alpha = (hn_img * self.attn_vecs).sum(axis=-1) / self._scale
        print("alpha shape:", alpha.shape) if self.debug else None
        # expect [L*N, num_heads]
        
        # normalized attention
        alpha = pad(alpha, num_imgs)
        print("alpha_pad shape:", alpha.shape) if self.debug else None
        # expect [L, N]?
        for ix, n in enumerate(num_imgs):
            alpha[n:, ix]=-50
        attn = torch.softmax(alpha, axis=0)
        print("attn shape:", attn.shape) if self.debug else None
        # expect [L, N, num_heads]
        
        # pool within subspaces
        h_query_pad = pad(hn_img, num_imgs) # reshapes the study with the number of images
        print("h_query_pad shape:", h_query_pad.shape) if self.debug else None
        # expect [L, N, num_heads, subspace size]
        h_study = torch.sum(h_query_pad * attn[...,None], axis=0)
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, num_heads, subspace size]
        
        h_study_wide = h_study.view(-1, self.num_features)
        print("h_study_wide shape:", h_study_wide.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study_wide, attn
            
    def tanh_attention_pool(self, h, num_imgs):
        # attention logits
        h_query = h.view(-1, self.num_heads, self.subspace_size)
        print("h_query shape:", h_query.shape) if self.debug else None
        # expect [L*N, num_heads, subspace_size]
        print("query vector shape:", self.attn_query_vecs.shape) if self.debug else None
        # expect [num_heads, subspace_size]
        alpha = torch.einsum('ijk,jlk->ijl', h_query, self.V).tanh()
        print("alpha shape:", alpha.shape) if self.debug else None
        # expect [L*N, num_heads, attn_hidden_size]
        lamb = torch.einsum('ijl,jl->ij', alpha, self.w)
        print("lambda shape:", lamb.shape) if self.debug else None
        # expect [L*N, num_heads]
        
        # normalized attention
        lamb = pad(lamb, num_imgs)
        for ix, n in enumerate(num_imgs):
            lamb[n:, ix]=-50
        attn = torch.softmax(lamb, axis=0)
        print("attn shape:", attn.shape) if self.debug else None
        # expect [L, N, num_heads]
        
        # pool within subspaces
        h_query_pad = pad(h_query, num_imgs)
        print("h_query_pad shape:", h_query_pad.shape) if self.debug else None
        # expect [L, N, num heads, subspace_size]
        h_study = torch.sum(h_query_pad * attn[...,None] / self._scale, axis=0)
        print("h_vid shape:", h_study.shape) if self.debug else None
        # expect [N, num heads, subspace_size]
        
        h_study_wide = h_study.view(-1, self.num_features)
        print("h_study_wide shape:", h_study_wide.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study_wide, attn
        
    def transformer_pool(self, h, num_imgs):
        # Project first
        h = self.projection(h)  # [L*N, proj_features]
        print("h-proj shape:", h.shape) if self.debug else None
        h_pad = pad(h, num_imgs)  # [L, N, proj_features] 
        print("h_pad shape:", h_pad.shape) if self.debug else None
        h_pad = h_pad.transpose(0, 1)  # [N, L, proj_features]
 
        # Rest remains same
        max_len = h_pad.size(1)
        batch_size = h_pad.size(0)
        attention_mask = torch.ones(batch_size, max_len + 1, dtype=torch.bool, device=h.device)
 
        for i, n in enumerate(num_imgs):
            attention_mask[i, n+1:] = False
 
        cls_tokens = self.cls_token.to(h.device).expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, h_pad), dim=1)
        print("x shape:", x.shape) if self.debug else None # [N, L+1, proj_features]
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        print("x-trans shape:", x.shape) if self.debug else None # [N, L+1, proj_features]
 
        h_study = x[:, 0] 
        print("h_study shape:", h_study.shape) if self.debug else None # [N, proj_features]
        
        return h_study, None

    def max_pool(self, hn_img, num_imgs):
        hn_pad = pad(hn_img, num_imgs)
        print("hn_pad shape:", hn_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take max
        h_study = hn_pad.max(0).values
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study, None
    
    def avg_pool(self, hn_img, num_imgs):
        hn_pad = pad(hn_img, num_imgs)
        print("hn_pad shape:", hn_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take avg
        h_study = hn_pad.mean(0)
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study, None

       

'''
Standard MIL Assumption
- following the standard assumption that at least one positive is in bag
meaning that the probability of a positive bag is equal to the probability 
of 1 minus the probability that none of the instances are bags
    p_bag = 1 - product((1-p_i), i)
'''
class StandardMILNet(nn.Module):
    def __init__(self, encoder, num_features=2048, num_out=1, dropout=0.0, debug=False):
        super(StandardMILNet, self).__init__()
        self.proj_features = num_features
        self.num_out = num_out
        self.drop_rate = dropout
        self.encoder = encoder
        if self.num_out == 1:
            self.fc = nn.Sequential(
                nn.Dropout(self.drop_rate),
                nn.Linear(self.proj_features, self.num_out),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(self.drop_rate),
                nn.Linear(self.proj_features, self.num_out),
                nn.Softmax(dim=0)
            )
        self.debug = debug
        
    def forward(self, x, num_imgs):
        # hn_img: the hidden representations of each img
        print("x shape:", x.shape) if self.debug else None
        print("num_imgs:", num_imgs) if self.debug else None
        
        # encoder
        features = self.encoder(x)
        print("feature shape:", features.shape) if self.debug else None
        
        # fc layer - instance probabilities
        probs = self.fc(features)
        print("probs shape:", probs.shape) if self.debug else None
        
        # padding images
        hn_pad = pad(probs, num_imgs)
        print("hn_pad shape:", hn_pad.shape) if self.debug else None
        
        # bag probability
        bag_p = 1 - torch.prod(1-hn_pad, dim=0)
        if self.num_out != 1:
            bag_p = bag_p / bag_p.sum(dim=1, keepdim=True)
        print("bag_p shape:", bag_p.shape) if self.debug else None
        
        return bag_p, None

        
        

       