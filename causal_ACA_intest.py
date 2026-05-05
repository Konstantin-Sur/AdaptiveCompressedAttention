class AdaptiveCompressedAttention1DCausal(nn.Module):

    def __init__(self, channels, heads=4, compress_ratio=4, top_k_frac=0.25):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.dim = channels // heads
        self.scale = self.dim ** -0.5
        self.top_k_frac = top_k_frac
        self.r = compress_ratio

        self.to_q = nn.Conv1d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv1d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv1d(channels, channels, 1, bias=False)

        self.compress_k = nn.Conv1d(
            channels, channels, self.r,
            stride=self.r, groups=channels,
            bias=False, padding=0,
        )
        self.compress_v = nn.Conv1d(
            channels, channels, self.r,
            stride=self.r, groups=channels,
            bias=False, padding=0,
        )
        self.out = nn.Conv1d(channels, channels, 1)

    # ------------------------------------------------------------------
    def _pad_to_multiple(self, x):
        L = x.shape[2]
        rem = L % self.r
        if rem != 0:
            x = F.pad(x, (0, self.r - rem))
        return x

    def _causal_pad(self, x):

        return F.pad(x, (self.r - 1, 0))

    # ------------------------------------------------------------------
    def forward(self, x):
        B, C, L = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)


        k_s = self.compress_k(self._causal_pad(self._pad_to_multiple(k)))
        v_s = self.compress_v(self._causal_pad(self._pad_to_multiple(v)))
        n = k_s.shape[2]  # ceil(L / r)

        def reshape(t, seq):
            return (t.reshape(B, self.heads, self.dim, seq)
                     .reshape(B * self.heads, self.dim, seq))

        BH = B * self.heads
        q_r = reshape(q,   L)   # [BH, dim, L]
        k_r = reshape(k_s, n)   # [BH, dim, n]
        v_r = reshape(v_s, n)


        attn_c = torch.einsum("b d i, b d j -> b i j", q_r, k_r) * self.scale
        # [BH, L, n]

        i_idx = torch.arange(L, device=x.device).view(1, L, 1)   # [1, L, 1]
        j_idx = torch.arange(n, device=x.device).view(1, 1, n)   # [1, 1, n]
        right_edge = j_idx * self.r + (self.r - 1)                # [1, 1, n]
        mask_c = right_edge > i_idx                               # [1, L, n] — True = маскируем

        attn_c = attn_c.masked_fill(mask_c, float("-inf"))
        attn_c_soft = torch.nan_to_num(attn_c.softmax(dim=-1))    # [BH, L, n]


        attn_c_for_topk = attn_c.clone()
        attn_c_for_topk = attn_c_for_topk.masked_fill(mask_c, -1.0)

        top_k = max(1, int(n * self.top_k_frac))

        _, idx = attn_c_for_topk.topk(top_k, dim=-1, largest=True, sorted=False)


        idx_k = idx.unsqueeze(1).expand(-1, self.dim, -1, -1)    # [BH, dim, L, top_k]
        # k_r: [BH, dim, n] → [BH, dim, 1, n] → expand
        k_exp = k_r.unsqueeze(2).expand(-1, -1, L, -1)           # [BH, dim, L, n]
        v_exp = v_r.unsqueeze(2).expand(-1, -1, L, -1)

        k_top = k_exp.gather(3, idx_k)   # [BH, dim, L, top_k]
        v_top = v_exp.gather(3, idx_k)

        # q_r: [BH, dim, L] → [BH, dim, L, 1]
        q_exp = q_r.unsqueeze(3)                                  # [BH, dim, L, 1]
        attn_f = (q_exp * k_top).sum(dim=1) * self.scale          # [BH, L, top_k]

        right_edge_f = idx * self.r + (self.r - 1)               # [BH, L, top_k]
        mask_f = right_edge_f > i_idx.squeeze(0)                  # [BH, L, top_k]
        attn_f = attn_f.masked_fill(mask_f, float("-inf"))
        attn_f = torch.nan_to_num(attn_f.softmax(dim=-1))         # [BH, L, top_k]

        out_c = torch.einsum("b i j, b d j -> b d i", attn_c_soft, v_r)

        out_f = (attn_f.unsqueeze(1) * v_top).sum(dim=3)          # [BH, dim, L]

        out = (out_c + out_f) * 0.5
        out = out.reshape(B, self.heads, self.dim, L).reshape(B, C, L)
        return self.out(out)
