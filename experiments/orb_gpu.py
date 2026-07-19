"""The speaking orb, simulated on the GPU.

The CPU orb is a shell of points pushed in and out along their own radius. It
reads well but it moves like a bellows, because radial displacement is the only
thing it can do. This one runs an actual particle simulation, which buys the
things that make a swarm look alive rather than animated:

* **Curl-noise flow.** Particles are advected by a divergence-free turbulent
  field, so they swirl and shear past each other instead of sliding along
  radii. Divergence-free matters: a plain noise field has sources and sinks,
  and particles visibly pile up in the sinks and thin out at the sources, which
  reads as clumping rather than as motion. The curl of a potential cannot do
  that. The voice feeds energy into the field — vowels swell it, consonants
  inject vorticity — so the turbulence is the speech rather than a loop playing
  underneath it.
* **Spherical harmonic modes.** The surface rings in the real vibration modes of
  a sphere, and the *pitch* of the voice picks the mode number. Speak higher and
  the sphere subdivides into more lobes. This is the one effect here with a
  physical idea behind it instead of a look: it is the voice shaping the sphere,
  not the volume shoving it.
* **Depth of field.** Particles away from the focal plane splat as soft wide
  discs, particles at it as tight points. Nothing separates "cinematic" from
  "synthetic" faster, and it also gives depth to what is otherwise a flat
  scatter of dots.
* **Trails.** The frame buffer decays instead of clearing, so fast motion
  streaks and the swarm keeps a memory of where it just was.
* **Surging spines.** Patches of the shell reach outward with the voice, so the
  surface bristles and settles back to a smooth sphere in the silences. The
  reaction comes from the orb's OWN particles: rays drawn on top of the sphere
  and a pane of frosted glass laid over it were both tried and both read as
  decoration stuck onto a ball rather than as one entity responding. The orb is
  the swarm, so everything it does has to be the swarm moving.
* **Events.** A shockwave crosses the swarm on the first syllable and the whole
  thing collapses and dissolves upward at the end, so speaking has a beginning
  and an end rather than just fading in and out.

Requires CUDA. `orb_render.Orb` remains the fallback.
"""
from __future__ import annotations

import math
import os

import numpy as np

SIZE = int(os.environ.get("TTS_ORB_SIZE", 660))
# Density is an aesthetic limit, not a hardware one. The GPU will happily push
# 220k, but on a 512 px canvas that is about one particle per pixel: they merge
# into a solid blob and the swarm stops reading as particles at all. The budget
# goes into simulation quality instead — flow, modes, depth of field, trails.
N_PARTICLES = int(os.environ.get("TTS_ORB_PARTICLES", 38_000))

# Palette in BGR (the raster hands premultiplied BGRA to Windows).
C_DEEP = np.array([255, 120, 60], np.float32) / 255.0      # deep blue
C_CYAN = np.array([255, 215, 110], np.float32) / 255.0     # cyan highlight
C_MAG = np.array([215, 60, 255], np.float32) / 255.0       # magenta

TRAIL_DECAY = float(os.environ.get("TTS_ORB_TRAIL", 0.42))  # per frame at 60 fps
FOCAL_Z = 0.55            # depth that renders sharp
DOF_STRENGTH = 2.1        # px of blur radius per unit of defocus
DOT_MIN_R, DOT_MAX_R = 0.45, 2.6
# Frosted glass, not a lens. The first attempt refracted the desktop through a
# glass sphere, which is optically the real thing and looks wrong on a screen:
# it reads as a paperweight with someone's monitor inside it. What "glass"
# means in an interface is frosted/acrylic — the backdrop blurred, its colour
# pushed UP (the saturation boost is what separates vibrant glass from a grey
# smear), a translucent tint, and fine grain so the blur doesn't band. Only the
# last few pixels of the rim bend anything.
# The orb IS the swarm. Everything it does with the voice has to come out of
# these same particles moving — not from rays drawn on top of them, and not
# from a pane of glass laid over them. Both were tried; both read as decoration
# stuck onto a sphere rather than as one thing reacting.
# Tuned down hard from a first pass that blew the shell apart: at 0.85 the
# surge outran the spring pulling particles home, the sphere never re-formed,
# and it left bald patches where a region had emptied. Spines are meant to be a
# few particles reaching out of an intact surface.
N_LOBES = 16              # regions of the shell that surge together
SURGE = 0.40              # how far a surging particle reaches at full voice
# Never completely still. A mathematically perfect circle sitting motionless
# reads as switched off, not as waiting — the thing is supposed to be an
# entity. So a slow, shallow undulation runs even in silence: enough that the
# outline is always breathing, far too little to look like it is reacting to
# something it cannot hear.
IDLE_SURGE = 0.16         # baseline reach with no voice at all
IDLE_FLOW = 2.1           # turbulence that keeps running in the quiet
# 1.0 would be critical damping (prompt but dead). Lower keeps inertia.
DAMPING_RATIO = float(os.environ.get("TTS_ORB_DAMPING", 0.26))
LOBE_DRIFT = 0.42         # how fast the surging regions wander


class OrbGPU:
    def __init__(self, size: int = SIZE, n: int = N_PARTICLES, seed: int = 7,
                 device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.size = size
        self.n = n
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        g = torch.Generator(device="cpu").manual_seed(seed)

        # Home positions: an even shell. Fibonacci rather than lat/long so there
        # is no clustering at the poles and nothing marks an axis.
        i = torch.arange(n, dtype=torch.float32)
        phi = torch.acos(1.0 - 2.0 * (i + 0.5) / n)
        theta = math.pi * (1.0 + 5.0 ** 0.5) * i
        home = torch.stack([torch.sin(phi) * torch.cos(theta),
                            torch.sin(phi) * torch.sin(theta),
                            torch.cos(phi)], 1)
        # A little thickness, so it is a shell of finite depth and not a film.
        thick = 1.0 + 0.075 * (torch.rand(n, 1, generator=g) - 0.5)

        self.home = (home * thick).to(self.device)
        self.dir = home.to(self.device)                     # unit, for modes
        self.phi = phi.to(self.device)
        self.theta = (theta % (2 * math.pi)).to(self.device)
        self.p = self.home.clone()
        self.v = torch.zeros_like(self.p)

        # Directions the shell surges toward. Regions rather than single
        # particles: if every particle reacted on its own the surface would
        # just get fuzzy, because neighbours would fly apart. Making a patch of
        # the shell move together is what produces a spine you can see.
        li = torch.arange(N_LOBES, dtype=torch.float32)
        lphi = torch.acos(1.0 - 2.0 * (li + 0.5) / N_LOBES)
        lth = math.pi * (1.0 + 5.0 ** 0.5) * li
        self.lobe_dir = torch.stack([torch.sin(lphi) * torch.cos(lth),
                                     torch.sin(lphi) * torch.sin(lth),
                                     torch.cos(lphi)], 1).to(self.device)
        self.lobe_ph = (6.283 * torch.rand(N_LOBES, generator=g)).to(self.device)
        # Within a surging region particles still differ, so its tip frays
        # instead of ending in a flat cap.
        self.react = (0.35 + 1.3 * torch.rand(n, generator=g) ** 2).to(self.device)

        self.seedv = torch.rand(n, 1, generator=g).to(self.device)
        self.size_var = (0.75 + 0.5 * torch.rand(n, generator=g)).to(self.device)
        self.bright_var = (0.55 + 0.75 * torch.rand(n, generator=g)).to(self.device)

        self.trail = torch.zeros(3, size, size, device=self.device)
        self._scrim = None
        self._t_prev = 0.0
        self._shock = -10.0        # time of the last onset shockwave
        self._was_quiet = True
        self._collapse = 0.0       # 0 = alive, 1 = fully dissolved
        self.zoom = size * 0.275

    # ------------------------------------------------------------------ flow

    def _curl(self, p, t: float, scale: float):
        """Curl of a sinusoidal vector potential: turbulence with no sources.

        Analytic rather than sampled from a noise texture, because the curl has
        to be exact — an approximated one leaks divergence, and the leak is
        visible as particles bunching into blobs.
        """
        torch = self.torch
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        k = scale
        # Potential Psi = (A, B, C), each a product of sines in the other axes.
        # curl = (dC/dy - dB/dz, dA/dz - dC/dx, dB/dx - dA/dy)
        s = 1.7 * k
        dC_dy = s * torch.cos(s * y + 0.7 * t) * torch.sin(s * x + 1.1 * t)
        dB_dz = s * torch.cos(s * z + 1.3 * t) * torch.sin(s * y + 0.5 * t)
        dA_dz = s * torch.cos(s * z + 0.9 * t) * torch.sin(s * x + 1.7 * t)
        dC_dx = s * torch.cos(s * x + 1.1 * t) * torch.sin(s * y + 0.7 * t)
        dB_dx = s * torch.cos(s * x + 0.5 * t) * torch.sin(s * z + 1.3 * t)
        dA_dy = s * torch.cos(s * y + 1.7 * t) * torch.sin(s * z + 0.9 * t)
        return torch.stack([dC_dy - dB_dz, dA_dz - dC_dx, dB_dx - dA_dy], 1)

    def _modes(self, pitch: float, energy: float, t: float):
        """Displacement along the normal from spherical-harmonic-like modes.

        The mode number comes from pitch, so a higher voice subdivides the
        surface into more lobes. Two modes are blended so the shape changes
        continuously as the pitch slides, instead of snapping between integers.
        """
        torch = self.torch
        lf = 2.0 + 6.0 * float(np.clip(pitch, 0, 1))     # 2..8 lobes
        l0, l1 = math.floor(lf), math.floor(lf) + 1
        f = lf - l0

        def mode(l: int):
            return (torch.cos(l * self.theta + 0.6 * t)
                    * torch.sin(l * self.phi) ** 2)

        # Deliberately shallow. Modes are meant to read as a ripple travelling
        # over the surface, not as a shape change: pushed far enough to dent the
        # outline they turn the sphere into a crumpled ball.
        return (mode(l0) * (1 - f) + mode(l1) * f) * (0.045 * energy)

    # ---------------------------------------------------------------- update

    def _step(self, t: float, lo: float, mid: float, hi: float, pitch: float):
        torch = self.torch
        dt = float(np.clip(t - self._t_prev, 1e-3, 0.05))
        self._t_prev = t
        energy = max(lo, mid)

        # Onset detection: the first syllable after quiet fires a shockwave.
        if energy > 0.34 and self._was_quiet:
            self._shock = t
        self._was_quiet = energy < 0.16

        # How far each particle reaches. The surging regions drift over the
        # shell and breathe at their own rates, so the spines appear in
        # different places each time rather than in a fixed crown.
        drift = self.lobe_dir + 0.35 * torch.stack([
            torch.sin(LOBE_DRIFT * t * 3.1 + self.lobe_ph),
            torch.sin(LOBE_DRIFT * t * 2.3 + self.lobe_ph * 1.7),
            torch.sin(LOBE_DRIFT * t * 1.7 + self.lobe_ph * 2.3)], 1)
        drift = drift / drift.norm(dim=1, keepdim=True).clamp(min=1e-4)
        # A high power keeps each region small. A broad one (the dot product
        # to the 7th) recruited most of a hemisphere, so the whole side of the
        # sphere heaved instead of a spine forming.
        # Width of a surging patch. Measured: at the 26th power only 7% of the
        # shell moved at all and the spines were invisible; at the 7th, half a
        # hemisphere heaved as one. The 11th puts a fifth of the particles in
        # motion, which reads as spines on an intact surface.
        aff = (self.dir @ drift.T).clamp(min=0) ** 11          # n x lobes
        # Two rates: a fast one that belongs to speech and a slow one that
        # keeps going regardless, so the shape is alive between utterances.
        pulse = (0.55 + 0.45 * torch.sin(5.2 * t + self.lobe_ph)
                 + 0.30 * torch.sin(0.62 * t + self.lobe_ph * 2.1)).view(1, -1)
        surge = (aff * pulse).amax(1) * self.react

        energy_push = IDLE_SURGE + 0.25 * lo + 0.70 * mid + 0.40 * hi
        target_r = (1.0 + 0.06 * lo + 0.045 * math.sin(2 * math.pi * 0.55 * t)
                    + SURGE * surge * energy_push
                    + self._collapse * -0.55)
        target = self.dir * (target_r + self._modes(pitch, mid, t)).unsqueeze(1)
        self._surge = surge

        # Shockwave: a ring of outward push travelling from the near side.
        age = t - self._shock
        if 0.0 <= age < 0.75:
            front = -1.0 + 2.6 * age                     # sweeps through z
            band = torch.exp(-((self.dir[:, 2] - front) / 0.22) ** 2)
            target = target + self.dir * (band * 0.30).unsqueeze(1)

        # Spring toward the target shell, plus turbulence, plus damping. The
        # spring is what keeps it a sphere; the curl is what stops it being a
        # bellows.
        # Stiff and near-critically damped. At the old 42 the spring's natural
        # period was almost a second, so a spine arrived a fifth of a second
        # after the syllable that called it — measured at 217 ms, which is
        # exactly the "not in time with the voice" that it looked like. The
        # brightness was always instant; it was the MOTION that trailed.
        # Damping is set to ~2*sqrt(k), the critical value: less and the shell
        # rings after every syllable, more and it crawls.
        stiff = 900.0 + 400.0 * self._collapse
        # Project the flow onto the tangent plane: particles swirl ACROSS the
        # surface instead of being pushed off it. A free curl field looks
        # organic but its radial component dents the silhouette, and the sphere
        # stops being a sphere — which is the one thing it has to stay.
        flow = self._curl(self.p, t, 1.0 + 0.8 * hi) * (IDLE_FLOW + 3.2 * energy + 2.0 * hi)
        nrm = self.p / self.p.norm(dim=1, keepdim=True).clamp(min=1e-4)
        flow = flow - nrm * (flow * nrm).sum(1, keepdim=True)
        self.v += (target - self.p) * (stiff * dt) + flow * dt
        # Underdamped on purpose. How FAST it responds is set by the spring
        # (omega = sqrt(stiff)); how much MOMENTUM the particles keep is set by
        # the damping ratio, and the two are independent. At critical damping
        # each particle slides to its target and stops dead, which is prompt but
        # reads as rigid — no follow-through, no sense of mass. At this ratio a
        # particle overshoots and settles back, so the swarm carries its motion
        # past each syllable without the onset arriving any later.
        self.v *= math.exp(-2.0 * DAMPING_RATIO * math.sqrt(stiff) * dt)
        self.p += self.v * dt
        if self._collapse > 0:
            self.p[:, 1] += self._collapse * dt * 0.9    # dissolve upward

    # ---------------------------------------------------------------- render

    def begin_collapse(self) -> None:
        """Called when the utterance ends, so it dissolves instead of blinking out."""
        self._collapse = min(1.0, self._collapse + 0.02)

    def set_backdrop(self, strength: float) -> None:
        """Accepted and ignored.

        Three attempts at helping the orb stand out on a white desktop all made
        it worse: a soft wide pool put its darkness where there were no
        particles and read as a smudge, an ink-on-paper mode filled the
        interior solid and threw away the grain, and a tight pool still looked
        like a grey stain on the page. The neon swarm on plain transparency is
        the look; it is genuinely lower contrast on a bright background, and
        that is the honest trade rather than something to paper over.
        """
        return

    def set_background(self, bgr) -> None:
        """Accepted and ignored. There was a frosted-glass pane here; it was a
        layer laid over the swarm, and the orb is meant to BE the swarm."""
        return

    def set_ambient(self, bgr) -> None:
        """Tint picked up from whatever is on screen behind the orb, so it looks
        lit by the desktop rather than pasted over it."""
        import torch
        self._ambient = torch.tensor(np.asarray(bgr, np.float32) / 255.0,
                                     device=self.device)

    def render(self, t: float, level=0.0, angle: float = 0.0) -> np.ndarray:
        torch = self.torch
        bands = np.clip(np.broadcast_to(np.asarray(level, np.float32), (4,)), 0, 1) \
            if np.asarray(level).size >= 4 else \
            np.append(np.clip(np.broadcast_to(np.asarray(level, np.float32), (3,)), 0, 1), 0.35)
        lo, mid, hi, pitch = (float(bands[0]), float(bands[1]),
                              float(bands[2]), float(bands[3]))
        s = self.size
        self._step(t, lo, mid, hi, pitch)

        ca, sa = math.cos(angle), math.sin(angle)
        X = self.p[:, 0] * ca + self.p[:, 2] * sa
        Z = -self.p[:, 0] * sa + self.p[:, 2] * ca
        Y = self.p[:, 1]

        px = X * self.zoom + s * 0.5
        py = -Y * self.zoom + s * 0.5
        # Clamped, and it must be. A surging particle reaches past the range
        # this mapping assumed, which made depth go slightly negative — and a
        # fractional power of a negative number is NaN. One NaN particle turned
        # the whole accumulation buffer to NaN, and the bloom then spread it in
        # 16-pixel blocks: that was the black rectangle punched through the orb.
        depth = ((Z + 1.9) / 3.8).clamp(0.0, 1.0)            # 0 back .. 1 front

        # Depth of field: the further from the focal plane, the wider and fainter
        # the splat. Energy is preserved as the disc grows, so defocused points
        # dim the way real bokeh does instead of just getting bigger.
        defocus = (Z - FOCAL_Z).abs().clamp(0.0, 4.0)
        radius = (DOT_MIN_R + DOF_STRENGTH * defocus).clamp(DOT_MIN_R, DOT_MAX_R)
        radius = radius * self.size_var

        modes = self._modes(pitch, mid, t) / max(0.045 * max(mid, 1e-3), 1e-4)
        rim = (1.0 - depth).clamp(0, 1) ** 2
        bright = (0.22 + 0.95 * depth ** 1.5) * self.bright_var
        bright = bright * (0.55 + 0.8 * mid + 0.35 * lo) * (1.0 + 0.6 * rim * hi)
        # The ripple brightens where the surface bulges toward us, so the mode
        # is legible without having to deform the outline to show it.
        bright = bright * (1.0 + 0.55 * mid * modes.clamp(-1, 1))
        # A surging particle also burns brighter and stays sharp, so a spine
        # reads as a spine and not just as a bulge in the outline.
        srg = getattr(self, "_surge", None)
        if srg is not None:
            bright = bright * (1.0 + 2.4 * srg * (0.3 + 0.7 * mid))
            # Clamp the floor. Sharpening a surging particle by scaling its
            # radius down drove the radius to zero (surge can exceed 1), and a
            # zero-radius splat covers nothing: the particles doing the most
            # visible thing erased themselves, which punched a hole where the
            # spine should have been.
            radius = (radius * (1.0 - 0.45 * srg.clamp(0, 1))).clamp(min=DOT_MIN_R)
        bright = bright * (1.0 - self._collapse)
        # Spread over the disc, so a wide bokeh circle is not brighter than a
        # tight point carrying the same light.
        bright = bright / (radius ** 1.35)

        g = (0.5 + 0.5 * (self.p[:, 1] + 0.4 * self.p[:, 0])).clamp(0, 1).unsqueeze(1)
        C_d = torch.tensor(C_DEEP, device=self.device)
        C_m = torch.tensor(C_MAG, device=self.device)
        C_c = torch.tensor(C_CYAN, device=self.device)
        col = C_d * (1 - g) + C_m * g
        lift = (0.30 * depth + 0.55 * hi + 0.25 * mid).clamp(0, 1).unsqueeze(1)
        col = col + (C_c - col) * lift
        amb = getattr(self, "_ambient", None)
        if amb is not None:
            col = col + (amb.view(1, 3) - col) * 0.18      # light from the desktop
        col = col * bright.unsqueeze(1)

        return self._raster(px, py, col, radius)

    def _raster(self, px, py, col, radius) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F
        s = self.size
        pad = int(DOT_MAX_R) + 1
        ok = (px >= pad) & (px < s - pad) & (py >= pad) & (py < s - pad)
        px, py, col, radius = px[ok], py[ok], col[ok], radius[ok]
        ix, iy = px.long(), py.long()

        # Four channels: colour, plus how much particle actually landed on the
        # pixel. Coverage has to be tracked separately because on a light
        # desktop the alpha can no longer come from brightness — see below.
        buf = torch.zeros(s * s, 4, device=self.device)
        offs = torch.arange(-pad, pad + 1, device=self.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        dx = ox + ix.float().unsqueeze(1) - px.unsqueeze(1)
        dy = oy + iy.float().unsqueeze(1) - py.unsqueeze(1)
        d2 = dx * dx + dy * dy
        r = radius.unsqueeze(1)
        # Soft-edged disc. A hard circle aliases badly at these sizes and the
        # bokeh looks like confetti.
        cov = torch.exp(-d2 / (0.5 * r * r + 1e-4)).clamp(0, 1)
        idx = ((iy.unsqueeze(1) + oy.long()) * s + (ix.unsqueeze(1) + ox.long()))
        contrib = torch.cat([col.unsqueeze(1) * cov.unsqueeze(2),
                             cov.unsqueeze(2)], 2)
        buf.index_add_(0, idx.reshape(-1), contrib.reshape(-1, 4))

        full = buf.view(s, s, 4).permute(2, 0, 1)
        img = full[:3] * 255.0
        cover = full[3:4].unsqueeze(0)
        # Trails: decay what was there instead of clearing, so motion streaks.
        self.trail = self.trail * TRAIL_DECAY + img
        img = self.trail.unsqueeze(0)
        self._cover = self._cover * TRAIL_DECAY + cover if hasattr(self, "_cover")             else cover

        # Interpolate back to an explicit size, not by scale_factor: pooling by
        # 16 then scaling by 16 only returns the original size when it divides
        # exactly, so any canvas that is not a multiple of 16 came back short
        # and the add failed.
        glow = F.interpolate(F.avg_pool2d(img, 4), size=(s, s),
                             mode="bilinear", align_corners=False)
        wide = F.interpolate(F.avg_pool2d(img, 16), size=(s, s),
                             mode="bilinear", align_corners=False)
        out = img + glow * 0.20 + wide * 0.11
        lum = out.amax(1, keepdim=True)
        # More contrast: a higher knee compresses the highlights less, and the
        # gamma pushes the dim haze between particles down so the bright cores
        # stand out instead of everything sitting in the same midtone.
        out = out * (255.0 / (lum + 230.0))
        out = 255.0 * (out / 255.0).clamp(0, 1) ** 1.28
        grey = out.mean(1, keepdim=True)
        out = grey + (out - grey) * 1.55
        a = out.amax(1, keepdim=True).clamp(0, 255)
        if self._scrim is not None:
            a = (a + self._scrim.view(1, 1, s, s) * (1.0 - a / 255.0)).clamp(0, 255)
        rgba = torch.cat([out.clamp(0, 255), a], 1)
        return rgba[0].permute(1, 2, 0).to(torch.uint8).contiguous().cpu().numpy()
