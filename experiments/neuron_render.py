"""An agent drawn as a neuron, and it fires when it speaks.

The orb is a good abstraction but it says nothing about what the thing does.
A neuron does: it sits quiet, receives, and when it has something to say a
pulse runs from the dendrites through the soma and out along the axon. That is
the shape of an agent answering, so the metaphor carries real information
rather than being decoration.

Structure is grown, not drawn:

* **Soma** — a dense irregular body of particles at the centre.
* **Dendrites** — branches grown recursively out of the soma, each splitting
  into thinner children at a wider angle, tapering as they go. Recursion is
  what makes it read as biological; hand-placed arms look like a starfish.
* **Axon** — one much longer, straighter, thinner process leaving the far side,
  ending in a spray of terminals.

Every particle remembers its **path distance from the soma**, which is what
makes the firing possible: an action potential is just a bright band travelling
outward in that coordinate. Dendrites light up on the way in, the soma flares,
and the axon carries it away — driven by the voice, so the neuron fires on the
syllable rather than on a timer.
"""
from __future__ import annotations

import math
import os

import numpy as np

SIZE = int(os.environ.get("TTS_NEURON_SIZE", 760))
N_DENDRITES = 7           # primary branches leaving the soma
DEPTH = 4                 # levels of splitting
SOMA_PTS = 5200
BRANCH_PTS = 210          # samples per segment
SPREAD = 0.62             # how far children deviate from the parent direction

C_DEEP = np.array([255, 120, 60], np.float32) / 255.0     # BGR deep blue
C_CYAN = np.array([255, 220, 130], np.float32) / 255.0    # BGR cyan
C_MAG = np.array([215, 60, 255], np.float32) / 255.0      # BGR magenta

PULSE_SPEED = 2.35        # path-units per second an action potential travels
PULSE_WIDTH = 0.30


def _grow(rng) -> tuple:
    """Grow the arbor. Returns (points, path distance, thickness, is_axon)."""
    pts, dist, thick, axon = [], [], [], []

    def branch(origin, direction, length, radius, travelled, depth, is_axon):
        # Curve each segment slightly: a straight branch reads as a wire.
        bend = rng.normal(0, 0.20, 3)
        n = BRANCH_PTS if depth > 1 else BRANCH_PTS // 2
        u = np.linspace(0, 1, n)
        curve = (direction[None, :] * u[:, None]
                 + bend[None, :] * (u ** 2)[:, None] * 0.45)
        p = origin[None, :] + curve * length
        # Jitter across the branch so it has body rather than being a line.
        p = p + rng.normal(0, radius * 0.42, p.shape)
        pts.append(p.astype(np.float32))
        dist.append((travelled + u * length).astype(np.float32))
        thick.append(np.full(n, radius, np.float32))
        axon.append(np.full(n, 1.0 if is_axon else 0.0, np.float32))

        end = origin + (direction + bend * 0.45) * length
        end_travel = travelled + length
        if depth <= 0:
            return
        kids = 2 if is_axon and depth > 1 else rng.integers(2, 4)
        for _ in range(int(kids)):
            d = direction + rng.normal(0, SPREAD, 3)
            d /= np.linalg.norm(d) + 1e-6
            branch(end, d, length * rng.uniform(0.55, 0.78),
                   radius * 0.62, end_travel, depth - 1, is_axon)

    # Soma: a lumpy ball, not a sphere.
    i = np.arange(SOMA_PTS)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / SOMA_PTS)
    th = math.pi * (1.0 + 5.0 ** 0.5) * i
    s = np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)
    lump = 1.0 + 0.16 * np.sin(3.1 * s[:, 0] + 1.0) + 0.13 * np.sin(2.7 * s[:, 1])
    s = s * (0.30 * lump * (0.80 + 0.35 * rng.random(SOMA_PTS)))[:, None]
    pts.append(s.astype(np.float32))
    dist.append(np.linalg.norm(s, axis=1).astype(np.float32))
    thick.append(np.full(SOMA_PTS, 0.30, np.float32))
    axon.append(np.zeros(SOMA_PTS, np.float32))

    # Dendrites, spread over the hemisphere away from the axon.
    for k in range(N_DENDRITES):
        a = 2 * math.pi * k / N_DENDRITES + rng.uniform(-0.25, 0.25)
        el = rng.uniform(-0.65, 0.85)
        d = np.array([math.cos(a) * math.cos(el), math.sin(el),
                      math.sin(a) * math.cos(el)], np.float64)
        d /= np.linalg.norm(d)
        branch(d * 0.28, d, rng.uniform(0.52, 0.78), 0.055, 0.28, DEPTH, False)

    # One axon: longer, straighter, thinner, leaving the opposite side.
    d = np.array([0.42, -0.86, 0.28]); d /= np.linalg.norm(d)
    branch(d * 0.28, d, 1.55, 0.040, 0.28, DEPTH - 1, True)

    return (np.concatenate(pts), np.concatenate(dist),
            np.concatenate(thick), np.concatenate(axon))


class Neuron:
    def __init__(self, size: int = SIZE, seed: int = 5,
                 device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.size = size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        rng = np.random.default_rng(seed)
        P, D, T, A = _grow(rng)

        self.p0 = torch.tensor(P, device=self.device)
        self.dist = torch.tensor(D, device=self.device)
        self.thick = torch.tensor(T, device=self.device)
        self.is_axon = torch.tensor(A, device=self.device)
        self.n = P.shape[0]
        self.max_dist = float(D.max())
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.jitter = torch.tensor(rng.normal(0, 1, (self.n, 3)).astype(np.float32),
                                   device=self.device)
        self.bright_var = (0.55 + 0.7 * torch.rand(self.n, generator=g)).to(self.device)

        self.trail = torch.zeros(3, size, size, device=self.device)
        self.zoom = size * 0.235
        self._fires: list = []        # times an action potential was launched
        self._was_quiet = True
        self._t_prev = 0.0
        self._scrim = None

    def set_backdrop(self, strength: float) -> None:
        return

    def set_background(self, bgr) -> None:
        return

    def render(self, t: float, level=0.0, angle: float = 0.0) -> np.ndarray:
        torch = self.torch
        b = np.clip(np.broadcast_to(np.asarray(level, np.float32), (4,)), 0, 1) \
            if np.asarray(level).size >= 4 else np.append(
                np.clip(np.broadcast_to(np.asarray(level, np.float32), (3,)), 0, 1), 0.4)
        lo, mid, hi = float(b[0]), float(b[1]), float(b[2])
        s = self.size
        energy = max(lo, mid)

        # Fire on each onset. A neuron does not glow continuously with volume;
        # it spikes, and the spike then travels. Firing on the leading edge is
        # what makes it read as a signal rather than as a level meter.
        if energy > 0.30 and self._was_quiet:
            self._fires.append(t)
        self._was_quiet = energy < 0.15
        self._fires = [f for f in self._fires
                       if t - f < (self.max_dist + 1.0) / PULSE_SPEED]

        # Gentle breathing so it is alive between spikes.
        wob = 1.0 + 0.012 * math.sin(1.3 * t) + 0.03 * energy
        p = self.p0 * wob + self.jitter * (0.004 + 0.012 * hi)

        ca, sa = math.cos(angle), math.sin(angle)
        X = p[:, 0] * ca + p[:, 2] * sa
        Z = -p[:, 0] * sa + p[:, 2] * ca
        Y = p[:, 1]
        px = X * self.zoom + s * 0.5
        py = -Y * self.zoom + s * 0.5
        depth = ((Z + 1.4) / 2.8).clamp(0, 1)

        # The action potential: a band of light at a given path distance,
        # travelling outward. Several can be in flight at once.
        glow = torch.zeros_like(self.dist)
        for f in self._fires:
            front = (t - f) * PULSE_SPEED
            glow = glow + torch.exp(-((self.dist - front) / PULSE_WIDTH) ** 2)
        glow = glow.clamp(0, 2.2)

        # Thicker parts of the arbor carry more light, as they hold more of it.
        base = (0.10 + 0.55 * self.thick / 0.30) * self.bright_var
        bright = base * (0.35 + 0.5 * depth) * (0.55 + 0.9 * mid + 0.4 * lo)
        bright = bright + glow * (0.9 + 1.7 * mid) * self.bright_var

        radius = (0.55 + 5.5 * self.thick).clamp(0.5, 2.6)
        bright = bright / (radius ** 1.15)

        # Axon runs cooler, dendrites warmer, and the travelling pulse burns
        # toward cyan so the signal is legible against the resting arbor.
        C_d = torch.tensor(C_DEEP, device=self.device)
        C_m = torch.tensor(C_MAG, device=self.device)
        C_c = torch.tensor(C_CYAN, device=self.device)
        mixv = (0.5 + 0.5 * (p[:, 1] / 1.4)).clamp(0, 1).unsqueeze(1)
        col = C_d * (1 - mixv) + C_m * mixv
        col = col + (C_c - col) * (0.25 * depth + 0.75 * glow.clamp(0, 1)).unsqueeze(1)
        col = col * (bright * 4.2).unsqueeze(1)
        return self._raster(px, py, col, radius)

    def _raster(self, px, py, col, radius) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F
        s = self.size
        pad = 4
        ok = (px >= pad) & (px < s - pad) & (py >= pad) & (py < s - pad)
        px, py, col, radius = px[ok], py[ok], col[ok], radius[ok]
        ix, iy = px.long(), py.long()
        buf = torch.zeros(s * s, 3, device=self.device)
        offs = torch.arange(-pad, pad + 1, device=self.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        dx = ox + ix.float().unsqueeze(1) - px.unsqueeze(1)
        dy = oy + iy.float().unsqueeze(1) - py.unsqueeze(1)
        r = radius.unsqueeze(1)
        cov = torch.exp(-(dx * dx + dy * dy) / (0.5 * r * r + 1e-4)).clamp(0, 1)
        idx = ((iy.unsqueeze(1) + oy.long()) * s + (ix.unsqueeze(1) + ox.long()))
        buf.index_add_(0, idx.reshape(-1), (col.unsqueeze(1) * cov.unsqueeze(2)).reshape(-1, 3))

        img = buf.view(s, s, 3).permute(2, 0, 1) * 255.0
        self.trail = self.trail * 0.55 + img
        img = self.trail.unsqueeze(0)
        glow = F.interpolate(F.avg_pool2d(img, 4), size=(s, s), mode="bilinear", align_corners=False)
        wide = F.interpolate(F.avg_pool2d(img, 16), size=(s, s), mode="bilinear", align_corners=False)
        out = img + glow * 0.30 + wide * 0.18
        lum = out.amax(1, keepdim=True)
        out = out * (255.0 / (lum + 210.0))
        out = 255.0 * (out / 255.0).clamp(0, 1) ** 1.22
        grey = out.mean(1, keepdim=True)
        out = grey + (out - grey) * 1.5
        a = out.amax(1, keepdim=True).clamp(0, 255)
        rgba = torch.cat([out.clamp(0, 255), a], 1)
        return rgba[0].permute(1, 2, 0).to(torch.uint8).contiguous().cpu().numpy()
