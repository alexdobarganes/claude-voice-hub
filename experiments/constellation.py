"""Many orbs at once: one per live agent, moving in space.

Each speaking session gets an orb. Position is not decoration — it is the
state: the one talking swims to the centre and grows, the ones queued orbit
close behind it, the ones that have finished drift outward and fade. So who
matters right now is legible without reading a word.

Motion on top of that is a slow constant drift plus soft repulsion, for the
same reason the single orb breathes in silence: something perfectly still
reads as switched off. Depth is real — orbs sit at different z and the
depth-of-field already in the renderer blurs the ones further back, which is
what gives the field its sense of space.

Built on OrbGPU: every orb is the same simulation, so they all bristle with
their own voice. Sharing one particle buffer across orbs keeps it to a single
GPU dispatch per frame instead of one per agent.
"""
from __future__ import annotations

import math
import os

import numpy as np

SIZE = int(os.environ.get("TTS_FIELD_SIZE", 1400))
PER_ORB = int(os.environ.get("TTS_FIELD_PARTICLES", 14_000))
MAX_ORBS = 7

# Where an orb wants to be, by state. Radius is in field units from the centre.
# Spread wide and deep. The orbs are meant to read as bodies at different
# distances, which needs real separation in z as well as across the frame —
# packed close together they just look like circles of different sizes.
SLOT = {
    "speaking": (0.00, 1.00),   # centre, full size, nearest the camera
    "queued":   (3.10, 0.58),
    "recent":   (4.60, 0.42),
}
# Perspective. Orthographic projection (what this did first) has no vanishing
# point, so distance could only be implied by blur; with a camera at CAM_Z the
# far orbs genuinely shrink and pull toward the centre the way they do in
# space, and the near one spreads.
CAM_Z = 5.2
FOCAL = 3.4
SPRING = 3.2       # how eagerly an orb moves to its slot
DAMP = 1.9
DRIFT = 0.055      # constant wander, so nothing is ever frozen
REPEL = 0.30       # soft push so two orbs never sit on top of each other


class Agent:
    """One orb: who it is, what it is doing, and where it currently floats."""

    def __init__(self, key: str, label: str, hue: float = 0.0) -> None:
        self.key, self.label, self.hue = key, label, hue
        self.state = "queued"
        self.pos = np.array([math.cos(hue * 6.283) * 1.2,
                             math.sin(hue * 6.283) * 0.7, 0.0], np.float32)
        self.vel = np.zeros(3, np.float32)
        self.scale = 0.30
        self.energy = np.zeros(4, np.float32)
        self.alpha = 0.0
        self.dying = False

    def target(self) -> tuple:
        r, s = SLOT.get(self.state, SLOT["recent"])
        a = self.hue * 6.283
        # Queued and recent orbs are spread around the speaker rather than
        # stacked: the angle comes from the agent's own hue so its place in the
        # field is stable between utterances instead of jumping about.
        # Depth comes from the agent's own hue, so each keeps a stable distance
        # rather than all sitting on one plane.
        # Depth kept shallow on purpose. Pushed to a realistic distance the far
        # orbs shrink and dim into specks — measured at z=-6 they were a third
        # the size and barely visible. Enough separation to read as depth, not
        # enough to lose them.
        z = -0.3 - 0.62 * r * (0.35 + 0.65 * ((self.hue * 2.7) % 1.0))
        return np.array([math.cos(a) * r * 1.15,
                         math.sin(a) * r * 0.66,
                         z], np.float32), s


class Field:
    """The whole constellation, rendered as one image."""

    def __init__(self, size: int = SIZE, per_orb: int = PER_ORB,
                 device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.size = size
        self.per_orb = per_orb
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agents: dict = {}
        self._t_prev = 0.0
        self.zoom = size * 0.150
        g = torch.Generator(device="cpu").manual_seed(11)

        # ONE shell of unit-sphere directions, reused by every orb. Each orb
        # scales and translates it, so N agents cost one dispatch rather than N.
        n = per_orb * MAX_ORBS
        i = torch.arange(n, dtype=torch.float32)
        phi = torch.acos(1.0 - 2.0 * ((i % per_orb) + 0.5) / per_orb)
        theta = math.pi * (1.0 + 5.0 ** 0.5) * i
        self.dir = torch.stack([torch.sin(phi) * torch.cos(theta),
                                torch.sin(phi) * torch.sin(theta),
                                torch.cos(phi)], 1).to(self.device)
        self.slot = (i // per_orb).long().to(self.device)      # which orb owns it
        self.react = (0.35 + 1.3 * torch.rand(n, generator=g) ** 2).to(self.device)
        self.bright_var = (0.55 + 0.75 * torch.rand(n, generator=g)).to(self.device)
        self.size_var = (0.75 + 0.5 * torch.rand(n, generator=g)).to(self.device)
        self.trail = torch.zeros(3, size, size, device=self.device)

        li = torch.arange(11, dtype=torch.float32)
        lphi = torch.acos(1.0 - 2.0 * (li + 0.5) / 11)
        lth = math.pi * (1.0 + 5.0 ** 0.5) * li
        self.lobe = torch.stack([torch.sin(lphi) * torch.cos(lth),
                                 torch.sin(lphi) * torch.sin(lth),
                                 torch.cos(lphi)], 1).to(self.device)

    # ---------------------------------------------------------------- roster

    def upsert(self, key: str, label: str, state: str) -> None:
        a = self.agents.get(key)
        if a is None:
            if len(self.agents) >= MAX_ORBS:
                # Drop the faintest rather than refuse the newcomer: the point
                # is to show what is live now.
                weakest = min(self.agents.values(), key=lambda x: x.alpha)
                self.agents.pop(weakest.key, None)
            a = Agent(key, label, hue=(len(self.agents) * 0.37) % 1.0)
            self.agents[key] = a
        a.label, a.state, a.dying = label, state, False

    def retire(self, key: str) -> None:
        a = self.agents.get(key)
        if a:
            a.state, a.dying = "recent", True

    def feed(self, key: str, bands) -> None:
        a = self.agents.get(key)
        if a is not None:
            a.energy = np.clip(np.asarray(bands, np.float32), 0, 1)

    # ----------------------------------------------------------------- step

    def step(self, t: float) -> None:
        dt = float(np.clip(t - self._t_prev, 1e-3, 0.05))
        self._t_prev = t
        live = list(self.agents.values())
        for a in live:
            tgt, s = a.target()
            # Constant wander, unique per agent, so the field is never static.
            tgt = tgt + DRIFT * np.array([
                math.sin(0.31 * t + a.hue * 9.0),
                math.sin(0.23 * t + a.hue * 5.0),
                math.sin(0.17 * t + a.hue * 7.0)], np.float32)
            # Soft repulsion: two agents must never occupy the same spot.
            for b in live:
                if b is a:
                    continue
                d = a.pos - b.pos
                dist = float(np.linalg.norm(d)) + 1e-3
                if dist < 1.1:
                    tgt = tgt + (d / dist) * (REPEL * (1.1 - dist))
            a.vel += (tgt - a.pos) * (SPRING * dt)
            a.vel *= math.exp(-DAMP * dt)
            a.pos += a.vel * dt
            a.scale += (s - a.scale) * min(1.0, 3.0 * dt)
            want = 0.0 if (a.dying and a.alpha < 0.02) else (0.30 if a.dying else 1.0)
            a.alpha += (want - a.alpha) * min(1.0, 1.6 * dt)
            if a.dying:
                a.alpha *= math.exp(-0.55 * dt)
        for a in [x for x in live if x.dying and x.alpha < 0.01]:
            self.agents.pop(a.key, None)

    # --------------------------------------------------------------- render

    def render(self, t: float) -> np.ndarray:
        torch = self.torch
        self.step(t)
        s = self.size
        live = list(self.agents.values())[:MAX_ORBS]
        if not live:
            return np.zeros((s, s, 4), np.uint8)

        # Per-orb state gathered into tensors, then indexed per particle by the
        # slot it belongs to. One dispatch for the whole field.
        cen = torch.zeros(MAX_ORBS, 3, device=self.device)
        scl = torch.zeros(MAX_ORBS, device=self.device)
        alp = torch.zeros(MAX_ORBS, device=self.device)
        lo_ = torch.zeros(MAX_ORBS, device=self.device)
        mid_ = torch.zeros(MAX_ORBS, device=self.device)
        hi_ = torch.zeros(MAX_ORBS, device=self.device)
        hue = torch.zeros(MAX_ORBS, device=self.device)
        for k, a in enumerate(live):
            cen[k] = torch.tensor(a.pos, device=self.device)
            scl[k] = a.scale
            alp[k] = a.alpha
            lo_[k], mid_[k], hi_[k] = float(a.energy[0]), float(a.energy[1]), float(a.energy[2])
            hue[k] = a.hue
        used = self.slot < len(live)

        d = self.dir[used]
        sl = self.slot[used]
        lo_p, mid_p, hi_p = lo_[sl], mid_[sl], hi_[sl]

        # The same surging shell as the single orb, per agent.
        # Normalised, and it matters: an un-normalised direction makes the dot
        # product exceed 1, and raised to the 11th that turns a gentle surge
        # into particles fired across the screen.
        drift = self.lobe / self.lobe.norm(dim=1, keepdim=True).clamp(min=1e-4)
        aff = (d @ drift.T).clamp(min=0) ** 11
        pulse = (0.55 + 0.45 * torch.sin(5.2 * t + torch.arange(
            self.lobe.shape[0], device=self.device))).view(1, -1)
        surge = (aff * pulse).amax(1) * self.react[used]
        r = 1.0 + 0.30 * surge * (0.14 + 0.22 * lo_p + 0.6 * mid_p + 0.35 * hi_p)
        r = r * (1.0 + 0.03 * math.sin(1.7 * t))

        p = d * r.unsqueeze(1) * scl[sl].unsqueeze(1) + cen[sl]
        # Perspective divide: everything scales by how far it is from the eye.
        proj = FOCAL / (CAM_Z - p[:, 2]).clamp(min=0.6)
        px = p[:, 0] * proj * self.zoom + s * 0.5
        py = -p[:, 1] * proj * self.zoom + s * 0.5
        depth = ((p[:, 2] + 3.0) / 4.0).clamp(0, 1)

        # Focus sits on the speaker; everything behind it softens with distance.
        defocus = (p[:, 2] - 0.0).abs().clamp(0, 4)
        radius = ((0.45 + 0.9 * defocus) * proj / FOCAL).clamp(0.4, 3.4)             * self.size_var[used] * scl[sl].clamp(min=0.22)
        # Falls off with distance, as light does, so the far orbs sit back
        # instead of being small but equally bright.
        fall = (proj / FOCAL * 1.55) ** 0.9
        bright = (0.30 + 0.85 * depth) * self.bright_var[used] * alp[sl] * fall *             (0.5 + 0.9 * mid_p + 0.35 * lo_p) / (radius ** 1.2)

        h = hue[sl].unsqueeze(1)
        C_A = torch.tensor([255., 120., 60.], device=self.device) / 255.
        C_B = torch.tensor([215., 60., 255.], device=self.device) / 255.
        C_C = torch.tensor([255., 215., 110.], device=self.device) / 255.
        base = C_A * (1 - h) + C_B * h
        col = base + (C_C - base) * (0.25 * depth + 0.4 * hi_p).clamp(0, 1).unsqueeze(1)
        col = col * (bright * 5.5).unsqueeze(1)
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
        self.trail = self.trail * 0.40 + img
        img = self.trail.unsqueeze(0)
        glow = F.interpolate(F.avg_pool2d(img, 4), size=(s, s), mode="bilinear", align_corners=False)
        wide = F.interpolate(F.avg_pool2d(img, 16), size=(s, s), mode="bilinear", align_corners=False)
        out = img + glow * 0.22 + wide * 0.13
        lum = out.amax(1, keepdim=True)
        out = out * (255.0 / (lum + 230.0))
        out = 255.0 * (out / 255.0).clamp(0, 1) ** 1.28
        grey = out.mean(1, keepdim=True)
        out = grey + (out - grey) * 1.5
        a = out.amax(1, keepdim=True).clamp(0, 255)
        rgba = torch.cat([out.clamp(0, 255), a], 1)
        return rgba[0].permute(1, 2, 0).to(torch.uint8).contiguous().cpu().numpy()
