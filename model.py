import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, w, eps=1e-6):
    if x.dtype == torch.float32:
        return torch.rms_norm(x, (x.shape[-1],), w, eps)
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.f1 = nn.Linear(d, dff, bias=False)
        self.f2 = nn.Linear(d, dff, bias=False)
        self.f3 = nn.Linear(dff, d, bias=False)

    def forward(self, x, fused=False):
        if fused and not self.training:
            ver = (self.f1.weight._version, self.f2.weight._version)
            if getattr(self, "_w12", None) is None or getattr(self, "_ver", None) != ver:
                self._w12 = torch.cat([self.f1.weight, self.f2.weight], 0)
                self._ver = ver
            a, b = F.linear(x, self._w12).chunk(2, -1)
            return self.f3(F.silu(a) * b)
        return self.f3(F.silu(self.f1(x)) * self.f2(x))


def rope_freqs(head_dim, ctx, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(ctx, device=device).float()
    ang = torch.outer(pos, inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, d, dff, k=4):
        super().__init__()
        self.k = k
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=0, groups=d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)
        self.mlp = MLP(d, dff)

    def forward(self, x, st=None, pos=0):
        h = self.n1(x)
        if st is None:
            c = self.dw(F.pad(h.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        else:
            hv = torch.cat([st["buf"], h.transpose(1, 2)], dim=2)
            if h.size(1) == 1:
                c = (hv * self.dw.weight.squeeze(1)).sum(-1, keepdim=True).transpose(1, 2)
            else:
                c = self.dw(hv).transpose(1, 2)
            st["buf"].copy_(hv[:, :, h.size(1):])
        x = x + self.mix(F.silu(self.gate(h)) * c)
        return x + self.mlp(self.n2(x), True)


class AttnBlock(nn.Module):
    def __init__(self, d, dff, heads, kv_heads, ctx=1024, base=10000.0):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.hd = d // heads
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.qkv = nn.Linear(d, d + 2 * kv_heads * self.hd, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp = MLP(d, dff)
        cos, sin = rope_freqs(self.hd, ctx, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, st=None, pos=0):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).split([d, self.kv_heads * self.hd, self.kv_heads * self.hd], -1)
        q = q.view(b, t, self.heads, self.hd).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.hd).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.hd).transpose(1, 2)
        if st is None:
            q = apply_rope(q, self.cos[:t].to(q.dtype), self.sin[:t].to(q.dtype))
            k = apply_rope(k, self.cos[:t].to(k.dtype), self.sin[:t].to(k.dtype))
            kk, vv = k, v
            if self.kv_heads < self.heads:
                g = self.heads // self.kv_heads
                kk = k.repeat_interleave(g, 1)
                vv = v.repeat_interleave(g, 1)
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        else:
            if t == 1:
                pi = st["n"] + torch.arange(t, device=x.device)
                q = apply_rope(q, self.cos.index_select(0, pi).to(q.dtype),
                               self.sin.index_select(0, pi).to(q.dtype))
                k = apply_rope(k, self.cos.index_select(0, pi).to(k.dtype),
                               self.sin.index_select(0, pi).to(k.dtype))
                if self.kv_heads < self.heads:
                    g = self.heads // self.kv_heads
                    k = k.repeat_interleave(g, 1)
                    v = v.repeat_interleave(g, 1)
                st["k"][:, :, pi] = k
                st["v"][:, :, pi] = v
                m = torch.arange(self.cos.shape[0], device=x.device)[None, :] <= pi[:, None]
                o = F.scaled_dot_product_attention(q, st["k"], st["v"], attn_mask=m)
                st["n"].add_(t)
            else:
                q = apply_rope(q, self.cos[pos:pos + t].to(q.dtype),
                               self.sin[pos:pos + t].to(q.dtype))
                k = apply_rope(k, self.cos[pos:pos + t].to(k.dtype),
                               self.sin[pos:pos + t].to(k.dtype))
                if self.kv_heads < self.heads:
                    g = self.heads // self.kv_heads
                    k = k.repeat_interleave(g, 1)
                    v = v.repeat_interleave(g, 1)
                st["k"][:, :, pos:pos + t] = k
                st["v"][:, :, pos:pos + t] = v
                kk = st["k"][:, :, :pos + t]
                vv = st["v"][:, :, :pos + t]
                if pos == 0:
                    o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
                else:
                    m = (torch.arange(pos + t, device=x.device)[None, :] <=
                         pos + torch.arange(t, device=x.device)[:, None])
                    o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=m)
                st["n"].fill_(pos + t)
        x = x + self.out(o.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.n2(x), True)


class CoreFlowLM(nn.Module):
    def __init__(self, vocab=4096, d=384, dff=512, nconv=8, natt=4, heads=6,
                 kv_heads=2, ctx=1024, softcap=30.0, stage_dropout=0.0,
                 dhead=0, dhead_depth=3, dhead_ff=64, dhead_rot=0, qat=0):
        super().__init__()
        assert d % heads == 0 and heads % kv_heads == 0
        self.vocab = vocab
        self.d = d
        self.ctx = ctx
        self.softcap = softcap
        self.stage_dropout = stage_dropout
        n = nconv + natt
        step = max(1, n // max(1, natt))
        attn_idx = {(k + 1) * step - 1 for k in range(natt)}
        blocks = []
        ai = 0
        for i in range(n):
            if i in attn_idx and ai < natt:
                blocks.append(AttnBlock(d, dff, heads, kv_heads, ctx))
                ai += 1
            else:
                blocks.append(ConvBlock(d, dff))
        while ai < natt:
            blocks.append(AttnBlock(d, dff, heads, kv_heads, ctx))
            ai += 1
        self.wte = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(blocks)
        self.nf = RMSNorm(d)
        self.drop_p = stage_dropout
        self.dhead = int(dhead)
        self.dhead_depth = dhead_depth
        self.dhead_rot = int(dhead_rot)
        self.qat = int(qat)
        if self.dhead:
            self.dn1 = RMSNorm(d)
            self.dm = MLP(d, dhead_ff)
            self.dn2 = RMSNorm(d)
        self.apply(self._init)
        self._scale_residual()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2))

    def _scale_residual(self):
        n = len(self.blocks)
        s = 0.02 / math.sqrt(2 * n)
        for blk in self.blocks:
            for m in ((blk.mix, blk.mlp.f3) if isinstance(blk, ConvBlock)
                      else (blk.out, blk.mlp.f3)):
                nn.init.normal_(m.weight, mean=0.0, std=s)
        if self.dhead:
            nn.init.normal_(self.dm.f3.weight, mean=0.0, std=s)

    def init_cache(self, b, device=None, dtype=None):
        dev = device if device is not None else self.wte.weight.device
        dt = dtype if dtype is not None else self.wte.weight.dtype
        cs = []
        for blk in self.blocks:
            if isinstance(blk, AttnBlock):
                cs.append({
                    "k": torch.zeros(b, blk.heads, self.ctx, blk.hd, device=dev, dtype=dt),
                    "v": torch.zeros(b, blk.heads, self.ctx, blk.hd, device=dev, dtype=dt),
                    "n": torch.zeros((), dtype=torch.long, device=dev),
                })
            else:
                cs.append({"buf": torch.zeros(b, self.d, blk.k - 1, device=dev, dtype=dt)})
        return cs

    def _fake_quant(self):
        with torch.no_grad():
            for m in self.modules():
                if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)):
                    continue
                w = m.weight
                rows = w.view(w.shape[0], -1).float()
                amax = rows.abs().amax(dim=1)
                s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
                q = torch.clamp((rows / s[:, None]).round(), -127, 127)
                w.copy_((q * s[:, None]).view_as(w))

    def forward(self, idx, cache=None, pos=0, draft=False):
        if self.qat:
            self._fake_quant()
        x = self.wte(idx)
        want = bool(self.dhead) and (self.training or draft)
        depth = 0
        if want:
            depth = (int(torch.randint(1, len(self.blocks), ()))
                     if (self.dhead_rot and self.training)
                     else self.dhead_depth)
            depth = max(1, min(depth, len(self.blocks) - 1))
        h_d = None
        if self.training and self.drop_p > 0:
            for i, blk in enumerate(self.blocks):
                if torch.rand(()) >= self.drop_p * (i + 1) / len(self.blocks):
                    x = blk(x)
                if i + 1 == depth:
                    h_d = x
        elif cache is None:
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i + 1 == depth:
                    h_d = x
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x, cache[i], pos)
                if i + 1 == depth:
                    h_d = x
        x = self.nf(x)
        logits = F.linear(x, self.wte.weight)
        if self.softcap and self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if h_d is None:
            return logits
        z = self.dn2(h_d + self.dm(self.dn1(h_d)))
        dl = F.linear(z, self.wte.weight)
        if self.softcap and self.softcap > 0:
            dl = self.softcap * torch.tanh(dl / self.softcap)
        return logits, dl


def warm_mlp(m):
    for x in m.modules():
        if isinstance(x, MLP):
            x._w12 = torch.cat([x.f1.weight, x.f2.weight], 0)
            x._ver = (x.f1.weight._version, x.f2.weight._version)
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--dff", type=int, default=512)
    p.add_argument("--nconv", type=int, default=8)
    p.add_argument("--natt", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv", type=int, default=2)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warm", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    m = CoreFlowLM(a.vocab, a.d, a.dff, a.nconv, a.natt, a.heads, a.kv, a.ctx)
    n = count_params(m)
    print("params {:.3f}M".format(n / 1e6), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    ids = torch.randint(0, a.vocab, (a.batch, a.seq))
    tgt = torch.randint(0, a.vocab, (a.batch, a.seq))
    times = []
    try:
        for step in range(a.warm + a.steps):
            t0 = time.perf_counter()
            logits = m(ids)
            loss = F.cross_entropy(logits.view(-1, a.vocab), tgt.view(-1))
            if not a.eval_only:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            el = time.perf_counter() - t0
            if step >= a.warm:
                times.append(el)
            del logits
    except MemoryError:
        print("OOM at seq={} batch={}".format(a.seq, a.batch))
        return
    avg = sum(times) / len(times)
    tok = a.batch * a.seq / avg
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1048576
    except Exception:
        rss = 0
    print("seq={} batch={} threads={} | {:.3f}M | {:.0f} tok/s | {:.0f} ms/step "
          "| rss {:.0f} MB | loss {:.3f}".format(
              a.seq, a.batch, a.threads, n / 1e6, tok, avg * 1000, rss, float(loss)))


if __name__ == "__main__":
    bench()
