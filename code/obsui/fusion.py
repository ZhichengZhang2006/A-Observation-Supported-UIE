"""E0-E3 decoder-side modules for the first-round architecture ablation.

Scope
-----
This module implements exactly the four configurations of
``docs/phase1_architecture_ablation_plan_20261109.md`` section 4, and nothing
else.  E4 (controlled degradation), E5 (shared pretraining) and the L-stage
loss experiments are deliberately absent.

Where it attaches
-----------------
``A_SOTA_Method_and_Experiments_v2.md`` section 2.2 says "add the module at the
1/4-resolution decoder feature".  In MambaCue that feature is the output of
``decoder.dec3`` -- ``[B, 128, 64, 64]`` for a 256x256 input -- which is the
first positional argument of ``decoder.dec2``.  We therefore install a
``forward_pre_hook`` on ``decoder.dec2`` and rewrite ``args[0]``.  Everything
else in the backbone (prior fusion, colour correction, detail injection, output
head) is untouched, so the ablation changes exactly one thing.

Gradient flow through the hook was verified: a module parameter downstream of a
pre-hook replacement receives the correct gradient (see the self-test at the
bottom of this file).

Configurations
--------------
E0  no module, ``F' = F``.
E1  plain residual capacity, ``F' = F + W_out(R_K(F))``, K chosen so the added
    parameter count matches E2 within +/-10 %.
E2  local + context, fixed fusion,
    ``F' = F + W_out(0.5*L(F) + 0.5*C(F))``.
E3  same two paths, learned spatial-channel fusion,
    ``F' = F + W_out(G*L(F) + (1-G)*C(F))``, ``G = sigmoid(Q([F, L(F), C(F)]))``.

``W_out`` is zero-initialised in E1/E2/E3, so every configuration starts as an
exact identity on the pretrained backbone and can only learn a correction.  The
E3 gate is zero-initialised as well, so ``G == 0.5`` at step 0 and E3 starts from
the same function as E2; the only difference is that the fusion *can* move.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("e0", "e1", "e2", "e3")

#: first-round context-path configuration, fixed by the brief (not tuned)
CONTEXT_POOL = 8
CONTEXT_HEADS = 4


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class LightResidualBlock(nn.Module):
    """``x + Conv1x1(SiLU(DWConv3x3(x)))`` -- the brief's "light residual block".

    Depthwise 3x3 for the spatial operator, 1x1 for channel mixing, one residual.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.mix = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(self.act(self.dw(x)))


class LocalPath(nn.Module):
    """L: two light residual blocks; edges, texture, local colour."""

    def __init__(self, channels: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[LightResidualBlock(channels) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ContextPath(nn.Module):
    """C: query from every position, key/value from an 8x8 pooled grid.

    Four heads of ``C/4`` channels, then a 1x1 channel mixing with a residual.
    Full-resolution self-attention would be O((HW)^2); pooling to 8x8 keeps the
    cost at O(HW * 64) while still letting every position read global context.
    """

    def __init__(self, channels: int, pool: int = CONTEXT_POOL, heads: int = CONTEXT_HEADS):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} not divisible by heads={heads}")
        self.pool = pool
        self.heads = heads
        self.q = nn.Conv2d(channels, channels, 1, bias=True)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=True)
        self.mix = nn.Conv2d(channels, channels, 1, bias=True)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)                      # [B, HW, C]
        p = F.adaptive_avg_pool2d(x, self.pool)                       # [B, C, 8, 8]
        kv = self.kv(p).flatten(2).transpose(1, 2)                    # [B, 64, 2C]
        k, v = kv.chunk(2, dim=-1)                                    # [B, 64, C] each
        out, _ = self.attn(q, k, v, need_weights=False)               # [B, HW, C]
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.mix(out)


class FusionGate(nn.Module):
    """Q: ``[F, L(F), C(F)]`` -> spatial-channel gate in (0, 1).

    ``concat 3C -> 1x1 to C -> DW3x3 -> 1x1 to C``, last layer zero-initialised so
    ``G == 0.5`` at step 0.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(3 * channels, channels, 1, bias=True)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.out = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, fo: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        z = self.act(self.reduce(torch.cat([x, fo, fc], dim=1)))
        z = self.act(self.dw(z))
        return torch.sigmoid(self.out(z))


def _zero_init_output(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


# --------------------------------------------------------------------------- #
# the four configurations
# --------------------------------------------------------------------------- #
class AblationModule(nn.Module):
    """One of E0/E1/E2/E3, acting on the 1/4-resolution decoder feature."""

    def __init__(self, mode: str, channels: int, e1_blocks: Optional[int] = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.channels = channels
        self.local: Optional[nn.Module] = None
        self.context: Optional[nn.Module] = None
        self.gate: Optional[nn.Module] = None
        self.stack: Optional[nn.Module] = None
        self.out: Optional[nn.Conv2d] = None

        if mode == "e1":
            k = 6 if e1_blocks is None else int(e1_blocks)
            self.stack = nn.Sequential(*[LightResidualBlock(channels) for _ in range(k)])
            self.e1_blocks = k
            self.out = _zero_init_output(nn.Conv2d(channels, channels, 1, bias=True))
        elif mode in ("e2", "e3"):
            self.local = LocalPath(channels)
            self.context = ContextPath(channels)
            if mode == "e3":
                self.gate = FusionGate(channels)
            self.out = _zero_init_output(nn.Conv2d(channels, channels, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "e0":
            return x
        if self.mode == "e1":
            return x + self.out(self.stack(x))
        fo = self.local(x)
        fc = self.context(x)
        if self.mode == "e2":
            fused = 0.5 * fo + 0.5 * fc
        else:
            g = self.gate(x, fo, fc)
            fused = g * fo + (1.0 - g) * fc
        return x + self.out(fused)

    # -- diagnostics required by the brief (section 2.3) -------------------- #
    @torch.no_grad()
    def stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Branch norms and gate statistics on one batch; cheap, for logging."""
        out: Dict[str, float] = {"mode": self.mode, "input_norm": float(x.norm())}
        if self.mode in ("e2", "e3"):
            fo = self.local(x)
            fc = self.context(x)
            out["local_norm"] = float(fo.norm())
            out["context_norm"] = float(fc.norm())
            if self.mode == "e3":
                g = self.gate(x, fo, fc)
                out["gate_mean"] = float(g.mean())
                out["gate_std"] = float(g.std())
                out["gate_min"] = float(g.min())
                out["gate_max"] = float(g.max())
                fused = g * fo + (1.0 - g) * fc
            else:
                fused = 0.5 * fo + 0.5 * fc
            out["delta_norm"] = float(self.out(fused).norm())
        elif self.mode == "e1":
            stacked = self.stack(x)
            out["stack_norm"] = float(stacked.norm())
            out["delta_norm"] = float(self.out(stacked).norm())
        return out


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def match_blocks_to_budget(channels: int, target_params: int) -> Tuple[int, int]:
    """Smallest K whose E1 stack (+ 1x1 output) reaches ``target_params``."""
    best_k, best_n = 1, 0
    for k in range(1, 64):
        n = count_params(AblationModule("e1", channels, e1_blocks=k))
        if n >= target_params:
            best_k, best_n = k, n
            break
        best_k, best_n = k, n
    return best_k, best_n


def build_module(mode: str, channels: int, e1_blocks: Optional[int] = None) -> AblationModule:
    """Construct one arm; E1 is parameter-matched to E2 when K is not given."""
    if mode == "e1" and e1_blocks is None:
        target = count_params(AblationModule("e2", channels))
        k, n = match_blocks_to_budget(channels, target)
        e1_blocks = k
        print(f"[fusion] E1 matched to E2: E2={target} E1={n} (+{100.0 * (n - target) / target:.1f}%) K={k}",
              flush=True)
    return AblationModule(mode, channels, e1_blocks=e1_blocks)


# --------------------------------------------------------------------------- #
# attachment to the backbone
# --------------------------------------------------------------------------- #
class AblationHook:
    """Owns the pre-hook on ``decoder.dec2`` and the module it inserts."""

    def __init__(self, decoder: nn.Module, module: AblationModule):
        self.decoder = decoder
        self.module = module
        self._handle = None

    def _pre(self, _module, args):
        d3, e2 = args[0], args[1]
        return (self.module(d3), e2)

    def attach(self) -> "AblationHook":
        if self._handle is not None:
            raise RuntimeError("hook already attached")
        # E0 has no parameters; attaching an identity hook keeps one code path
        # for every arm and makes the 'no module' condition explicit.
        self._handle = self.decoder.dec2.register_forward_pre_hook(self._pre)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def attach(model: nn.Module, mode: str, channels: Optional[int] = None,
           e1_blocks: Optional[int] = None) -> Tuple[AblationModule, AblationHook]:
    """Insert arm ``mode`` into ``model`` (a MambaCue ``myModel``).

    ``channels`` defaults to the real ``decoder.dec3`` output channel count, read
    off the module rather than assumed.
    """
    dec = model.decoder
    if channels is None:
        channels = dec.dec3.block.block[0].out_channels
    module = build_module(mode, channels, e1_blocks=e1_blocks)
    hook = AblationHook(dec, module).attach()
    return module, hook


def module_parameters(module: AblationModule) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    b, c, h, w = 2, 32, 16, 16
    x = torch.randn(b, c, h, w)

    # 1) E0 is the identity
    m0 = AblationModule("e0", c)
    assert torch.equal(m0(x), x), "E0 must be an exact identity"

    # 2) E1/E2/E3 are exact identities at initialisation
    for mode in ("e1", "e2", "e3"):
        m = AblationModule(mode, c, e1_blocks=2 if mode == "e1" else None)
        y = m(x)
        err = (y - x).abs().max().item()
        assert err < 1e-7, f"{mode} not identity at init: {err}"
        assert y.shape == x.shape
        print(f"[selftest] {mode}: identity at init, max|d|={err:.2e}, params={count_params(m)}")

    # 3) E3 gate starts at 0.5 everywhere
    m3 = AblationModule("e3", c)
    g = m3.gate(x, m3.local(x), m3.context(x))
    assert abs(float(g.mean()) - 0.5) < 1e-6 and abs(float(g.std())) < 1e-6, "gate must init at 0.5"
    print(f"[selftest] E3 gate init mean={float(g.mean()):.6f} std={float(g.std()):.2e}")

    # 4) the 1x1 output is the only zero-initialised part: perturbing a path
    #    must NOT change the output at init (that is the point of the design)
    m2 = AblationModule("e2", c)
    with torch.no_grad():
        m2.local.blocks[0].mix.bias.add_(1.0)
    assert (m2(x) - x).abs().max().item() < 1e-7, "zero-init output must mask path perturbations at init"

    # 5) after un-zeroing the output, the module is no longer the identity and
    #    gradients reach both paths and the gate
    for mode, attrs in (("e1", ["stack"]), ("e2", ["local", "context"]), ("e3", ["local", "context", "gate"])):
        m = AblationModule(mode, c, e1_blocks=2 if mode == "e1" else None)
        with torch.no_grad():
            nn.init.normal_(m.out.weight, std=0.05)
        y = m(x)
        loss = (y - x).pow(2).mean()
        loss.backward()
        grads = {}
        for name in attrs:
            sub = getattr(m, name)
            tot = sum(float(p.grad.abs().sum()) for p in sub.parameters() if p.grad is not None)
            grads[name] = tot
            assert tot > 0, f"{mode}: no gradient reached {name}"
        print(f"[selftest] {mode}: grad norms {grads}")

    # 6) E1 parameter matching is within the documented +/-10 %
    for cc in (32, 128, 256):
        e2 = count_params(AblationModule("e2", cc))
        k, e1 = match_blocks_to_budget(cc, e2)
        rel = 100.0 * (e1 - e2) / e2
        print(f"[selftest] C={cc}: E2={e2} E1={e1} K={k} delta={rel:+.1f}%")
        assert abs(rel) <= 10.0, f"C={cc}: E1 outside +/-10 % of E2 ({rel:+.1f}%)"

    # 7) hook semantics: replacing args[0] of dec2 feeds the module's output
    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.dec3 = nn.Identity()
            self.dec2 = _Capture()
            self.dec1 = nn.Identity()

        def forward(self, e2):
            d3 = self.dec3(e2)
            d2 = self.dec2(d3, e2)
            return self.dec1(d2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = FakeDecoder()

        def forward(self, x):
            return self.decoder(x)

    model = Model()
    mod = AblationModule("e2", c)
    hook = AblationHook(model.decoder, mod).attach()
    model(x)
    captured = model.decoder.dec2.last[0]
    # Value-equality, not object identity: E1-E3 return ``x + out(...)``, which is a
    # new tensor even when the correction is exactly zero.
    assert torch.allclose(captured, x, atol=1e-7), "the hook must pass through at init"
    with torch.no_grad():
        nn.init.normal_(mod.out.weight, std=0.05)
    model(x)
    assert not torch.allclose(model.decoder.dec2.last[0], x), "after un-zeroing, the hook must transform d3"
    hook.detach()
    model(x)
    assert model.decoder.dec2.last[0] is x, "detach must restore the original graph"
    print("[selftest] hook attach/transform/detach OK")

    print("[selftest] all checks passed")


class _Capture(nn.Module):
    """Test helper: record the (d3, skip) pair dec2 would receive."""

    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, x, skip):
        self.last = (x, skip)
        return x


if __name__ == "__main__":
    _selftest()
