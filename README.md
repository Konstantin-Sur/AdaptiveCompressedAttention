# AdaptiveCompressedAttention

```python
class AdaptiveCompressedAttention(nn.Module):
    def __init__(self, channels, heads=4, compress_ratio=4, top_k_frac=0.25):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.dim = channels // heads
        self.scale = self.dim ** -0.5
        self.top_k_frac = top_k_frac
        self.r = compress_ratio
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.compress_k = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)
        self.compress_v = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)
        self.out = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        k_s = self.compress_k(k)
        v_s = self.compress_v(v)
        n = k_s.shape[2] * k_s.shape[3]

        def reshape(t, seq):
            return t.reshape(B, self.heads, self.dim, seq).reshape(B*self.heads, self.dim, seq)

        q = reshape(q.reshape(B, C, N), N)
        k = reshape(k_s.reshape(B, C, n), n)
        v = reshape(v_s.reshape(B, C, n), n)

        attn = torch.einsum("b d i, b d j -> b i j", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        token_score = attn.sum(dim=1)
        top_k = max(1, int(n * self.top_k_frac))
        _, idx = token_score.topk(top_k, dim=-1)
        idx_exp = idx.unsqueeze(1).expand(-1, self.dim, -1)
        k_top = k.gather(2, idx_exp)
        v_top = v.gather(2, idx_exp)

        attn_full = torch.einsum("b d i, b d j -> b i j", q, k_top) * self.scale
        attn_full = attn_full.softmax(dim=-1)

        out = torch.einsum("b i j, b d j -> b d i", attn_full, v_top)
        out = out.reshape(B, self.heads, self.dim, N).reshape(B, C, H, W)
        return self.out(out)
```

***

## Benchmark results for 'measure_memory_and_time.py'

**Parameters:**
|channels|compress_ratio|heads|top_k_frac|
|-|-|-|-|
|512|32|4|0.25|

**Results (on google T4 used in google colab)**
| Size         | Peak Memory (MB) | Time (ms) |
|--------------|------------------|-----------|
| 64x64        | 64.46 MB         | 8.73 ms   |
| 128x128      | 213.21 MB        | 12.29 ms  |
| 256x256      | 864.44 MB        | 40.47 ms  |
| 512x512      | 4369.38 MB       | 206.89 ms |
| 1024x1024    | OOM              | N/A       |

*For example, baseline attention with 512 dim, 4 heads and 512x512 seq size will require 1024GB or 1TB of memory*

***

## Results using ACA together with Resnet on Oxford Pets task.
**Epoch 03 | val_acc=92.210% (best)**

*You can reproduce results using google colab(T4 gpu) and oxford_pets.py code*





## IN TEST! MAY BE NOT WORKING
```python
class AdaptiveCompressedAttention(nn.Module):
    def __init__(self, channels, heads=4, compress_ratio=4, top_k_frac=0.25):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.dim = channels // heads
        self.scale = self.dim ** -0.5
        self.top_k_frac = top_k_frac
        self.r = compress_ratio

        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        self.compress_k = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)
        self.compress_v = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)

        # новые проекции для broadcast-шага
        self.to_q2 = nn.Conv2d(channels, channels, 1, bias=False)  # запросы от всех токенов
        self.to_k2 = nn.Conv2d(channels, channels, 1, bias=False)  # ключи от победителей
        self.to_v2 = nn.Conv2d(channels, channels, 1, bias=False)  # значения от победителей

        self.out = nn.Conv2d(channels, channels, 1)

        # layer norm перед broadcast (стабилизирует)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        k_s = self.compress_k(k)
        v_s = self.compress_v(v)
        n = k_s.shape[2] * k_s.shape[3]

        def reshape(t, seq):
            return t.reshape(B, self.heads, self.dim, seq).reshape(B * self.heads, self.dim, seq)

        q_r  = reshape(q.reshape(B, C, N), N)
        k_r  = reshape(k_s.reshape(B, C, n), n)
        v_r  = reshape(v_s.reshape(B, C, n), n)

        attn = torch.einsum("b d i, b d j -> b i j", q_r, k_r) * self.scale
        attn = attn.softmax(dim=-1)

        token_score = attn.sum(dim=1)                              # [B*heads, n]
        top_k = max(1, int(n * self.top_k_frac))
        _, idx = token_score.topk(top_k, dim=-1)                   # [B*heads, k]
        idx_exp = idx.unsqueeze(1).expand(-1, self.dim, -1)        # [B*heads, dim, k]

        k_top = k_r.gather(2, idx_exp)                             # [B*heads, dim, k]
        v_top = v_r.gather(2, idx_exp)                             # [B*heads, dim, k]

        attn_full = torch.einsum("b d i, b d j -> b i j", q_r, k_top) * self.scale
        attn_full = attn_full.softmax(dim=-1)
        out1 = torch.einsum("b i j, b d j -> b d i", attn_full, v_top)

        x1 = out1.reshape(B, self.heads, self.dim, N).reshape(B, C, H, W)


        
        with torch.no_grad():
            score_per_head = token_score.reshape(B, self.heads, n)
            score_avg = score_per_head.mean(dim=1)                 # [B, n]
            _, idx_shared = score_avg.topk(top_k, dim=-1)         # [B, k]


        Hs, Ws = H // self.r, W // self.r


        x1_norm = self.norm(x1.reshape(B, C, N).permute(0, 2, 1)).permute(0, 2, 1)
        # x1_norm: [B, C, N]


        q2 = self.to_q2(x1).reshape(B, C, N)                      # [B, C, N]


        idx_shared_exp = idx_shared.unsqueeze(1).expand(-1, C, -1) # [B, C, k]


        x_compressed = nn.functional.avg_pool2d(x, self.r)        # [B, C, Hs, Ws]
        x_comp_flat = x_compressed.reshape(B, C, -1)              # [B, C, n]

        winners_feat = x_comp_flat.gather(2, idx_shared_exp)      # [B, C, k]


        wf = winners_feat.reshape(B, C, top_k, 1)
        k2_w = self.to_k2(wf).reshape(B, C, top_k)                # [B, C, k]
        v2_w = self.to_v2(wf).reshape(B, C, top_k)                # [B, C, k]


        def reshape2(t, seq):
            return t.reshape(B, self.heads, self.dim, seq).reshape(B * self.heads, self.dim, seq)

        q2_r  = reshape2(q2, N)
        k2_r  = reshape2(k2_w, top_k)
        v2_r  = reshape2(v2_w, top_k)

        attn2 = torch.einsum("b d i, b d j -> b i j", q2_r, k2_r) * self.scale
        attn2 = attn2.softmax(dim=-1)                              
        out2  = torch.einsum("b i j, b d j -> b d i", attn2, v2_r)


        out2 = out2.reshape(B, self.heads, self.dim, N).reshape(B, C, H, W)

        out = x1 + out2

        return self.out(out)
```
