import torch.nn as nn
import torch
from utils import pad
import math
class MedVidNet(nn.Module):
    def __init__(self, encoder, num_heads=16, num_out=1, pooling_method='attn', drop_rate=0.0, 
                 debug=False, attn_hidden_size=64):
        super(MedVidNet, self).__init__()
        
        # ResNet encoder
        self.encoder = encoder
        self.num_features = encoder.num_features
        
        assert self.num_features % num_heads == 0, "The number of encoder features must be divisble by the number of attention heads."
        self.num_heads = num_heads
        self.subspace_size = self.num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.num_out = num_out  # 1 for binary quality classification
        self.drop_rate = drop_rate
        self.debug = debug
        self.pool = pooling_method
        self.attn_hidden_size = attn_hidden_size
        
        self.encoder = encoder
        
        # FC for final classification; transformer pooling outputs the 16-d projected CLS/frame features
        out_dim = 16 if self.pool == 'transformer' else self.num_features
        self.fc_out = nn.Sequential(
            nn.Dropout(p=self.drop_rate),
            nn.Linear(out_dim, self.num_out)
        )
        
        if self.pool == 'transformer':
            self.pool_func = self.transformer_pool
            self.projection = nn.Linear(self.num_features, 16)
            # Add transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=16,
                nhead=num_heads,
                dim_feedforward=128,
                dropout=drop_rate,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=1  
            )
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, 16))
            # Positional encoding
            # self.pos_embedding = nn.Parameter(torch.randn(1, 512, self.num_features))  # 512 is max sequence length

        elif self.pool == 'attn':
            self.pool_func = self.attention_pool
            self.attn_query_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        elif self.pool == 'tanh_attn':
            self.pool_func = self.tanh_attention_pool
            self.V = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size, self.subspace_size))
            self.w = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size))
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'transformer', 'attn', 'max', or 'avg'")
    
    def forward(self, x, num_frames, study_indices):
        h = self.encoder(x)
        h_vid, attn = self.pool_func(h, num_frames, study_indices)
        output = self.fc_out(h_vid)
        return output, attn
    
    def attention_pool(self, h, num_frames, study_indices):
        h_query = h.view(-1, self.num_heads, self.subspace_size)  # [-1, 16, 128]
        alpha = (h_query * self.attn_query_vecs).sum(axis=-1) / self._scale  # [-1, 16]

        alpha = pad(alpha, num_frames)  # [max_frames, batch]
        for ix, n in enumerate(num_frames):
            alpha[n:, ix] = -50
        attn = torch.softmax(alpha, dim=0)

        h_query_pad = pad(h_query, num_frames)  # [max_frames, batch, 16, 128]
        h_study = torch.sum(h_query_pad * attn[..., None], dim=0)  # [batch, 16, 128]

        unique_studies = torch.unique(study_indices)
        study_features = []

        for study_idx in unique_studies:
            study_mask = (study_indices == study_idx)
            study_frames = h_study[study_mask]  # [num_study_frames, 16, 128]
            study_mean = study_frames.mean(dim=0)  # [16, 128]
            study_features.append(study_mean)

        h_vid = torch.stack(study_features)  # [num_studies, 16, 128]
        h_vid = h_vid.reshape(h_vid.size(0), -1)  # [num_studies, 16*128]
        return h_vid, attn
        
    def tanh_attention_pool(self, h, num_frames, study_indices):
        h_query = h.view(-1, self.num_heads, self.subspace_size)  # [-1, 16, 128]
        alpha = torch.einsum('ijk,jlk->ijl', h_query, self.V).tanh()  
        lamb = torch.einsum('ijl,jl->ij', alpha, self.w)

        lamb = pad(lamb, num_frames)
        for ix, n in enumerate(num_frames):
            lamb[n:, ix] = -50
        attn = torch.softmax(lamb, dim=0)

        h_query_pad = pad(h_query, num_frames)  # [max_frames, batch, 16, 128]
        h_study = torch.sum(h_query_pad * attn[..., None] / self._scale, dim=0)  # [batch, 16, 128]

        unique_studies = torch.unique(study_indices)
        study_features = []

        for study_idx in unique_studies:
            study_mask = (study_indices == study_idx)
            study_frames = h_study[study_mask]  # [num_study_frames, 16, 128]
            study_mean = study_frames.mean(dim=0)  # [16, 128]
            study_features.append(study_mean)

        h_vid = torch.stack(study_features)  # [num_studies, 16, 128]
        h_vid = h_vid.reshape(h_vid.size(0), -1)  # [num_studies, 16*128]
        return h_vid, attn

    def max_pool(self, h, num_frames, study_indices):
        h = h.view(-1, self.num_heads, self.subspace_size)  # [-1, 16, 128]
        h_pad = pad(h, num_frames)  # [max_frames, batch, 16, 128]

        unique_studies = torch.unique(study_indices)
        study_features = []

        for study_idx in unique_studies:
            study_mask = (study_indices == study_idx)
            study_frames = h_pad[:, study_mask, :]  # [max_frames, num_study_frames, 16, 128]
            study_max = study_frames.max(0).values.mean(0)  # [16, 128]
            study_features.append(study_max)

        h_vid = torch.stack(study_features)  # [num_studies, 16, 128]
        h_vid = h_vid.reshape(h_vid.size(0), -1)  # [num_studies, 16*128]
        return h_vid, None

    def avg_pool(self, h, num_frames, study_indices):
        h = h.view(-1, self.num_heads, self.subspace_size)  # [-1, 16, 128]
        h_pad = pad(h, num_frames)  # [max_frames, batch, 16, 128]

        unique_studies = torch.unique(study_indices)
        study_features = []

        for study_idx in unique_studies:
            study_mask = (study_indices == study_idx)
            study_frames = h_pad[:, study_mask, :]  # [max_frames, num_study_frames, 16, 128]
            study_mean = study_frames.mean(dim=0).mean(dim=0)  # [16, 128]
            study_features.append(study_mean)

        h_vid = torch.stack(study_features)  # [num_studies, 16, 128]
        h_vid = h_vid.reshape(h_vid.size(0), -1)  # [num_studies, 16*128]
        return h_vid, None
    
    def transformer_pool(self, h, num_frames, study_indices):
        # Project first  
        h = self.projection(h)  # [L*N = Total number of frames, 16]
        h_pad = pad(h, num_frames)  # [L, N, 16]
        h_pad = h_pad.transpose(0, 1)  # [N, L, 16]

        max_len = h_pad.size(1)
        batch_size = h_pad.size(0)
        attention_mask = torch.ones(batch_size, max_len + 1, dtype=torch.bool, device=h.device)
        for i, n in enumerate(num_frames):
            attention_mask[i, n+1:] = False

        cls_tokens = self.cls_token.to(h.device).expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, h_pad), dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)

        # Remove CLS token and get only frame features
        frame_features = x[:, 1:]  # [N, L, 16]

        real_frames_mask = attention_mask[:, 1:].unsqueeze(-1)  # [N, L, 1]

        # Zero out padded frames
        masked_features = frame_features * real_frames_mask

        # Process each study separately
        unique_studies = torch.unique(study_indices)
        study_features = []

        for study_idx in unique_studies:
            # Get mask for this study's videos
            study_mask = (study_indices == study_idx)
            # Get features for all frames in this study
            study_frames = masked_features[study_mask]  # Get all frames from study's videos
            study_mask = real_frames_mask[study_mask]  

            # Get only valid frames from this study
            valid_study_frames = study_frames[study_mask.squeeze(-1)]
            # Mean across all valid frames in this study
            study_mean = valid_study_frames.mean(dim=0)  # [16]
            study_features.append(study_mean)

        # Stack all study representations
        h_vid = torch.stack(study_features)  # [B, 16] 

        return h_vid, None