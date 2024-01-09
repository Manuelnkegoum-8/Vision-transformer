import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import random,math
import torch.nn.functional as F

class shift_patch(nn.Module):
    def __init__(self,patch_size=16):
        super().__init__()        
        self.patch_size = patch_size // 2
    def forward(self, x):
        bs,_, _, chanel = x.size()
        shifts = ((-self.patch_size, self.patch_size, -self.patch_size, self.patch_size), 
                    (self.patch_size, -self.patch_size, -self.patch_size, self.patch_size),
                    (-self.patch_size, self.patch_size, self.patch_size, -self.patch_size), 
                    (self.patch_size, -self.patch_size, self.patch_size, -self.patch_size))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        return shifted_x
    
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class patch_embedding(nn.Module):
    def __init__(self,height=224,width=224,n_channels=3,patch_size=16,batch_size=2,dim=512):
        super().__init__()
        
        assert height%patch_size==0 and width%patch_size==0 ,"Height and Width should be multiples of patch size wich is {0}".format(patch_size)
        #self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = posemb_sincos_2d(
            h = height // patch_size,
            w = width // patch_size,
            dim = dim,
        ) 
        self.patch_size = patch_size
        self.n_patchs = height*width//(patch_size**2)
        #self.embedding = torch.nn.Parameter(torch.randn(1,self.n_patchs+1, dim))
        self.projection = nn.Sequential(nn.LayerNorm(patch_size*patch_size*n_channels*5),
                                        nn.Linear(patch_size*patch_size*n_channels*5,dim),
                                        nn.LayerNorm(dim)
                                       )
        self.shift = shift_patch(patch_size=patch_size)
        self.patch = nn.Sequential(
            Rearrange("b c (h p1) (w p2)  -> b (h w)  (p1 p2 c)", p1 = patch_size, p2 = patch_size),
        )
    def forward(self, x):
        #x bs,h,w,c
        embedding = self.pos_embedding.to(x.device)
        left_up,right_up,left_down,right_down = self.shift(x)
        x = torch.cat([x,left_up,right_up,left_down,right_down],dim=1)
        #first we resize the inputs Bs,Num_patchs,*
        x = self.patch(x)
        #projection on the dim of model
        x = self.projection(x)
        outputs = x + embedding
        return outputs
        
class LocallySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LocallySelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.tau = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim)))
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim,bias = False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def _masked_softmax(self, attention_scores, attention_mask):
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))
        return F.softmax(attention_scores, dim=-1)

    def forward(self,x, need_weights=True, attn_mask=None):
        x = self.norm(x)
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.tau
        attention_scores = self._masked_softmax(attention_scores, attn_mask)
        attention_output = torch.matmul(attention_scores, v)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        return self.o_proj(attention_output)

class feedforward(nn.Module):
    def __init__(self,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, x):
        outputs = self.mlp(x)
        return outputs
        
class Transformer(nn.Module):
    def __init__(self, embed_dim=512, depth=4, heads=2, ff_hidden_dim=1024, dropout_rate = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocallySelfAttention(embed_dim,heads),
                feedforward(embed_dim,ff_hidden_dim, dropout_rate)
            ]))
    def forward(self, x,mask=None):
        for attn, ff in self.layers:
            x = attn(x,attn_mask = mask) + x
            x = ff(x) + x
        return x
        
class vit(nn.Module):
    def __init__(self,height=224,width=224,n_channels=3,patch_size=16,batch_size=2,dim=512,n_head=2,feed_forward=1024,num_blocks=4,num_classes=1):
        super().__init__()
        self.embedding = patch_embedding(height,width,n_channels,patch_size,batch_size,dim)
        self.n_patchs = height*width//(patch_size**2)
        # Create a diagonal attention mask
        self.diag_attn_mask = torch.eye(self.n_patchs, dtype=torch.bool)
        self.transformer_encoder = Transformer(dim,num_blocks,n_head,feed_forward)
        self.mlp_block =  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
    def forward(self, inputs):
        #x bs,h,w,c
        device_ = inputs.device
        x = self.embedding(inputs)
        x = self.transformer_encoder(x,mask=self.diag_attn_mask.to(device_))
        outputs = self.mlp_block(x.mean(dim=1))
        return outputs
        
        
