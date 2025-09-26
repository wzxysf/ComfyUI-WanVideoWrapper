import torch
import torch.nn as nn
from typing import Optional, List
from ..wanvideo.modules.attention import attention

def vector_to_list(tensor, lens, dim):
    return list(torch.split(tensor, lens, dim=dim))

def list_to_vector(tensor_list, dim):
    lens = [tensor.shape[dim] for tensor in tensor_list]
    tensor = torch.cat(tensor_list, dim)
    return tensor, lens

def merge_token_lists(list1, list2, dim):
    assert(len(list1) == len(list2))
    return [torch.cat((t1, t2), dim) for t1, t2 in zip(list1, list2)]


class WanLynxIPCrossAttention(nn.Module):
    def __init__(self, cross_attention_dim=5120, dim=5120, n_registers=16, bias=True):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        if n_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, n_registers, cross_attention_dim) / dim**0.5)
        else:
            self.registers = None
    
    def forward(self, block, q, x, ip_x):
        b, n, d = x.size(0), block.num_heads, block.head_dim

        if self.registers is not None:
            print("self.registers.shape", self.registers.shape) #torch.Size([1, 16, 5120])
            print("ip_x.shape", ip_x.shape) #torch.Size([1, 16, 5120])
           
            ip_lens = [ip_x.shape[1]]
            ip_x_list = vector_to_list(ip_x, ip_lens, 1)
            ip_x_list = merge_token_lists(ip_x_list, [self.registers] * len(ip_x_list), 1)
            ip_x, ip_lens = list_to_vector(ip_x_list, 1)

        ip_key = self.to_k_ip(ip_x)
        ip_key = ip_key * torch.rsqrt(ip_key.pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(ip_key.dtype)

        ip_value = self.to_v_ip(ip_x)

        ip_key = ip_key.view(b, -1, n, d)
        ip_value = ip_value.view(b, -1, n, d)

        ip_x = attention(q, ip_key, ip_value).reshape(b, -1, n * d)

        return ip_x


class WanLynxRefAttention(nn.Module):
    def __init__(self, dim=5120, bias=True):
        super().__init__()
        self.to_k_ref = nn.Linear(dim, dim, bias=bias)
        self.to_v_ref = nn.Linear(dim, dim, bias=bias)
    
    def forward(self, q, ref_feature: Optional[tuple] = None):

        ref_x, ref_lens = ref_feature
        ref_query = q

        ref_query = self.self_attn.norm_q(ref_query)
        ref_key = self.self_attn.norm_k(ref_key)

        ref_query = ref_query.unflatten(2, (self.self_attn.heads, -1)).transpose(1, 2)
        ref_key = ref_key.unflatten(2, (self.self_attn.heads, -1)).transpose(1, 2)
        ref_value = ref_value.unflatten(2, (self.self_attn.heads, -1)).transpose(1, 2)
        ref_x = attention(ref_query, ref_key, ref_value)

        return self.self_attn.o(ref_x.flatten(2))