"""Live renderer for the speaking orb: a 3D particle sphere that reacts to audio.

Renders one frame at a time from `(t, level)`, so the overlay can drive it with
the real amplitude envelope of whatever is being spoken instead of playing a
canned loop.

`level` is the per-band audio energy (low, mid, high), each 0..1. Giving each
band its own visual dimension is what makes the motion read as locked to the
voice rather than merely reacting to volume — a single amplitude number can't
distinguish a vowel from an "s", so everything ends up looking like the same
generic pulse:
  * **low** (60-350 Hz, the body of the voice) → particles push outward, the
    whole sphere breathes bigger. This is the "weight" of the sound.
  * **mid** (350-2200 Hz, vowels/formants) → brightness and bloom. This is what
    tracks the syllables you actually hear.
  * **high** (2200-9000 Hz, consonants/sibilance) → per-particle shimmer and a
    cyan lift on the rim. Sharp, short, and visible exactly on the "s"/"t"
    sounds that carry no low-frequency energy at all.

Performance: the loop must hold 60 fps on one core WHILE audio plays, so every
frame stays under ~16 ms. What that costs us in code (all of it measured, not
guessed — a naive version of this same image ran at 20 ms/frame):
  * splatting via `np.bincount` on flat indices, not `np.add.at` (which is slow)
  * colors carried in **BGR** order end to end, so handing the frame to Windows
    is one contiguous copy instead of three per-channel ones
  * bloom blurred at 1/2 and 1/4 resolution and scaled back up
  * tonemap and saturation algebraically folded into a single pair of fused
    multiply-adds over the image (see `_finish`), instead of four passes
  * every working buffer preallocated once and reused
"""
from __future__ import annotations

import time

import numpy as np

SIZE = 384          # 4K screens: 256 read as a thumbnail, and the post-process
                    # rewrite bought enough budget to render this at 60 fps
# Particles only. There is no wireframe of meridians and parallels: those lines
# read as a drawn grid laid over a ball rather than as a cloud, and they give
# the sphere an axis, which is exactly what it should not have.
# Density is nearly free here: dropping from 11600 points to 9300 moved the
# frame rate by 0.3 fps (56.3 -> 56.0), because the cost is pixel-bound (bloom,
# tonemap, and handing the frame to Windows) rather than point-bound. So use the
# count that looks right, not the one that seems cheap.
import os

N_SHELL = int(os.environ.get("TTS_ORB_N_SHELL", 9000))    # particles forming the sphere's surface
N_DUST = int(os.environ.get("TTS_ORB_N_DUST", 2600))       # looser speckle floating just outside it
GAIN = float(os.environ.get("TTS_ORB_GAIN", 1.85))         # mesh spreads the same light over many more
                                                            # points, so the per-point gain has to come
                                                            # up to keep the old punch
# How hard the surface pushes outward on the low band, and how much shimmer
# rides the high band. Raised from the original 0.16/0.055 for a punchier,
# more liquid-feeling reaction to the voice.
REACT_LO = float(os.environ.get("TTS_ORB_REACT_LO", 0.38))
REACT_HI = float(os.environ.get("TTS_ORB_REACT_HI", 0.14))

# Palette in BGR order (see module docstring): blue shell -> magenta shell,
# cyan lift on the lit/front side. TTS_ORB_C_* overrides ("b,g,r" 0-255) let a
# caller randomize the look for exploration without touching this file.
def _bgr(name: str, default: list[float]) -> np.ndarray:
    v = os.environ.get(name)
    if not v:
        return np.array(default, dtype=np.float32) / 255.0
    b, g, r = (float(x) for x in v.split(","))
    return np.array([b, g, r], dtype=np.float32) / 255.0

C_BLUE = _bgr("TTS_ORB_C_BLUE", [255, 130, 70])
C_CYAN = _bgr("TTS_ORB_C_CYAN", [255, 200, 80])
C_MAGENTA = _bgr("TTS_ORB_C_MAGENTA", [210, 60, 255])

BLOOM_MARGIN = 0.09  # fraction of the canvas kept clear so the glow isn't cut
_KNEE = 150.0        # tonemap soft-knee point
_SAT = 1.45          # saturation boost (blurs always wash color out)
_BLOOM_THRESHOLD = 0.38   # only cores above this glow; see Orb._bloom

# Tonemap lookup table. The curve is 255*(1-exp(-lum/knee))/lum, and evaluating
# np.exp per pixel cost more than everything else in the frame combined. The
# curve is smooth and one-dimensional, so a table sampled finely enough is
# indistinguishable and turns the whole tonemap into one gather.
_LUT_N = 4096
_LUT_MAX = 4000.0
_lut_x = np.linspace(0.0, _LUT_MAX, _LUT_N, dtype=np.float32)
_TONE_LUT = np.where(
    _lut_x > 1e-6,
    (255.0 / np.maximum(_lut_x, 1e-6)) * (1.0 - np.exp(-_lut_x / _KNEE)),
    255.0 / _KNEE,
).astype(np.float32)
_LUT_SCALE = np.float32((_LUT_N - 1) / _LUT_MAX)


def _chan_max(a: np.ndarray) -> np.ndarray:
    """Per-pixel max across 3 channels.

    `a.max(axis=2)` measured 3.9 ms at 384 px and runs twice per frame: numpy
    reduces over a length-3 innermost axis one strided element at a time. Two
    np.maximum calls over whole contiguous planes do it ~8x faster.
    """
    return np.maximum(np.maximum(a[:, :, 0], a[:, :, 1]), a[:, :, 2])


def _chan_mean(a: np.ndarray) -> np.ndarray:
    """Per-pixel channel mean, plane-wise for the same reason as _chan_max."""
    return (a[:, :, 0] + a[:, :, 1] + a[:, :, 2]) * np.float32(1.0 / 3.0)


class Orb:
    def __init__(self, size: int = SIZE,
                 seed: int = int(os.environ.get("TTS_ORB_SEED", 7)),
                 n_shell: int = N_SHELL, n_dust: int = N_DUST) -> None:
        self.size = size
        rng = np.random.default_rng(seed)

        # Two concentric shells of free particles. Both are Fibonacci
        # spirals, which spread points evenly over a sphere without the
        # clustering at the poles that a latitude/longitude layout produces —
        # and without any lines, so nothing marks an axis.
        def shell(count: int, jitter: float) -> tuple:
            i = np.arange(count)
            phi = np.arccos(1.0 - 2.0 * (i + 0.5) / max(count, 1))
            theta = np.pi * (1.0 + 5.0 ** 0.5) * i
            r = 1.0 + jitter * rng.random(count)
            return ((np.sin(phi) * np.cos(theta) * r).astype(np.float32),
                    (np.sin(phi) * np.sin(theta) * r).astype(np.float32),
                    (np.cos(phi) * r).astype(np.float32))

        sx, sy, sz = shell(n_shell, 0.045)      # the body of the sphere
        dx, dy, dz = shell(n_dust, 0.34)        # the halo around it
        dx, dy, dz = dx * 1.10, dy * 1.10, dz * 1.10

        self.x = np.concatenate([sx, dx])
        self.y = np.concatenate([sy, dy])
        self.z = np.concatenate([sz, dz])
        n = self.x.size
        self.n_mesh = n_shell                   # boundary between shell and halo
        self.base_r = np.ones(n, dtype=np.float32)
        self.weight = np.empty(n, dtype=np.float32)
        self.weight[:n_shell] = 0.85 + 0.30 * rng.random(n_shell)
        self.weight[n_shell:] = 0.35 + 0.55 * rng.random(n_dust)

        self.k = rng.integers(1, 4, n).astype(np.float32)
        self.off = rng.random(n).astype(np.float32)
        self.amp = (0.08 + 0.18 * rng.random(n)).astype(np.float32)
        # Per-point reactivity: some particles fly out harder than others, so
        # loud moments look like a burst rather than a uniform zoom. Kept small
        # on the mesh, which has to stay a coherent surface.
        self.react = (0.5 + 1.1 * rng.random(n)).astype(np.float32)
        self.react[:n_shell] *= 0.55
        # Per-particle shimmer frequency, for the high-band sibilance jitter.
        self.shimmer = (2.0 + 6.0 * rng.random(n)).astype(np.float32)

        self._flat = np.empty(4 * n, dtype=np.int64)      # 4 bilinear taps
        self._vals = np.empty(4 * n, dtype=np.float32)
        self._out = np.empty((size, size, 4), dtype=np.uint8)

        # Mass-spring-damper state driving the elastic stretch in `_radii`:
        # a plain multiply-by-loudness (an earlier version of this method)
        # tracks the input instantly and has no momentum, which reads as
        # rigid, not elastic -- real soft bodies overshoot past where they're
        # pushed and settle back, they don't just scale in lockstep with the
        # force. Lightly underdamped on purpose, for a small natural jiggle
        # on the settle rather than a dead stop.
        self._spring_pos = 0.0
        self._spring_vel = 0.0
        self._spring_wall: float | None = None
        self._SPRING_STIFFNESS = 55.0
        self._SPRING_DAMPING = 6.5

        # Fit the sphere to the canvas instead of hoping a hand-picked zoom
        # clears it. Dust sits well outside the mesh (base_r up to ~1.42) and
        # everything swells further on loud passages, so a fixed zoom that
        # looked right in silence clipped the halo the moment the voice came in.
        #
        # The worst-case radius is *sampled* rather than derived by adding every
        # term's maximum: those maxima belong to different particles at
        # different instants, so summing them describes a frame that never
        # happens and shrinks the orb to a dot in a mostly empty canvas.
        # A high percentile rather than the outright max: the dust shell has a
        # few stragglers far outside the body, and letting one of them set the
        # scale shrinks the orb to a small disc in a mostly empty canvas. The
        # bloom margin absorbs those few.
        # spring=1.25 rather than the implicit lo=1.0 fallback: the
        # mass-spring-damper in _step_spring is underdamped on purpose (a
        # natural jiggle on the settle) and overshoots a sustained target=1.0
        # by roughly 20%. Fit against that overshoot, not just the target, or
        # a loud passage clips the halo the instant the spring rebounds past 1.
        worst = max(float(np.percentile(
                        np.abs(self._radii(t, 1.0, 1.0, spring=1.25)), 99.0))
                    for t in np.linspace(0.0, 4.0, 24))
        self.zoom = (0.5 - BLOOM_MARGIN) / (worst * (1.0 + 0.06))

    def _life_pulse(self, t: float) -> float:
        """A slow breath plus a faster, subtler double-beat riding on top of
        it -- present at all times, including silence, so the orb reads as a
        living thing at rest rather than something that only moves when
        spoken to. One clean fast sinusoid (the old 2 Hz pulse) felt
        mechanical; a slow breath with an irregular heartbeat-like texture on
        top does not, because nothing about it repeats on a short, obvious
        period."""
        breath = np.sin(2 * np.pi * 0.32 * t)
        heartbeat = (np.sin(2 * np.pi * 1.15 * t)
                    + 0.5 * np.sin(2 * np.pi * 2.31 * t + 0.6))
        return 0.06 * breath + 0.02 * heartbeat

    def _blob(self, t: float) -> np.ndarray:
        """Asymmetric, continuously-writhing lobes -- what makes the silhouette
        read as a creature rather than a ball. A single fixed standing wave (an
        earlier version of this file used one) makes a static-looking egg and
        nothing more; the fix isn't to drop the asymmetry but to keep it always
        moving: four low-order harmonics of position, each with its own slow,
        mutually-incommensurate frequency, so lobes grow, recede and migrate
        around the surface independently and the shape never resettles into a
        fixed egg or visibly repeats within any short window."""
        x, y, z = self.x, self.y, self.z
        h1 = x * y
        h2 = y * z
        h3 = z * x
        h4 = x * x - y * y
        a1 = np.sin(2 * np.pi * 0.070 * t + 0.0)
        a2 = np.sin(2 * np.pi * 0.113 * t + 1.7)
        a3 = np.sin(2 * np.pi * 0.051 * t + 3.4)
        a4 = np.sin(2 * np.pi * 0.137 * t + 5.1)
        return 0.16 * (a1 * h1 + a2 * h2 + a3 * h3 + 0.7 * a4 * h4)

    def _step_spring(self, target: float) -> float:
        """Advance the mass-spring-damper one real time-step toward `target`,
        return the new position. Uses wall-clock dt (not the caller's `t`,
        which the overlay slows to 0.35x) so the physics has a consistent
        real-world feel regardless of that time-dilation."""
        now = time.monotonic()
        if self._spring_wall is None:
            dt = 1.0 / 60.0
        else:
            dt = min(0.05, max(0.0, now - self._spring_wall))
        self._spring_wall = now
        accel = (self._SPRING_STIFFNESS * (target - self._spring_pos)
                 - self._SPRING_DAMPING * self._spring_vel)
        self._spring_vel += accel * dt
        self._spring_pos += self._spring_vel * dt
        return self._spring_pos

    def _radii(self, t: float, lo: float, hi: float,
              spring: float | None = None) -> np.ndarray:
        """Per-point radius at time `t` for the given band levels. Shared by
        render() and the canvas fit so the two can never disagree.

        `spring` is the elastic-stretch driver; render() advances the real
        mass-spring-damper and passes its position here, so the stretch can
        overshoot past `lo` and settle back like a soft body actually would.
        The canvas fit calls this without `spring`, which falls back to `lo`
        directly (a spring driven by a synthetic, non-monotonic `t` sweep
        during __init__ wouldn't mean anything -- there's no real time to
        advance it against), padded up so the fit still clears a real
        overshoot at runtime.
        """
        r = self.base_r.copy()
        # Elastic coupling: voice doesn't just puff the surface outward on top
        # of the blob's existing lobes, it STRETCHES those lobes -- the same
        # asymmetric bulge that's already there gets pulled further out where
        # it's already out, and pushed further in where it's already in.
        # That's what reads as squeezing a soft body rather than inflating a
        # balloon. `self.react` still gives per-particle variation, so the
        # stretch isn't perfectly uniform across the surface either.
        drive = lo if spring is None else spring
        blob = self._blob(t)
        r += blob * (1.0 + 2.4 * drive * self.react)
        # Per-particle drift, small and only on the halo, so the surface itself
        # keeps its radius.
        r[self.n_mesh:] += (self.amp
                            * np.sin(2 * np.pi * (self.k * t + self.off)))[self.n_mesh:]
        r *= 1.0 + self._life_pulse(t)
        # Smaller, faster traveling ripples on top of the blob: `_blob` sets
        # the large-scale asymmetric shape, this is surface-level liquid
        # texture -- a phase that sweeps across the surface with `t` instead
        # of sitting still. Two mismatched frequencies so it never repeats.
        r += (0.09 * np.sin(3.1 * self.x + 2.3 * self.y + 2.6 * t)) * (1.0 + lo)
        r += (0.065 * np.sin(4.7 * self.z - 1.9 * self.x + 1.9 * t + 1.3)) * (1.0 + lo)
        # A small residual direct push keeps quiet/breathy speech visible even
        # where the blob's own lobes happen to be near zero at that instant.
        r += 0.5 * REACT_LO * lo * self.react
        if hi > 0.02:
            r += REACT_HI * hi * np.sin(self.shimmer * (12.0 * t) + self.off * 6.283)
        return r

    def render(self, t: float, level=0.0, angle: float = 0.0) -> np.ndarray:
        """One frame -> uint8 [size, size, 4] premultiplied BGRA.

        `level` is (low, mid, high) band energies; a bare float is accepted and
        treated as all three bands at once. Callers may pass extra trailing
        channels (e.g. pitch) after the three bands -- only the first three
        are used here, the rest is ignored rather than broadcast-failing.

        The returned array is reused between calls; copy it if you need to keep
        a frame around.
        """
        arr = np.asarray(level, np.float32)
        if arr.ndim > 0 and arr.shape[-1] >= 3:
            bands = np.clip(arr[:3], 0.0, 1.0)
        else:
            bands = np.clip(np.broadcast_to(arr, (3,)), 0.0, 1.0)
        lo, mid, hi = float(bands[0]), float(bands[1]), float(bands[2])
        lv = float(bands.max())
        s = self.size

        # Displacement must be a SMOOTH function of position, not per-point
        # noise: neighbouring samples on a wireframe line have to move together
        # or the grid dissolves into speckle. See _radii for the shape of it.
        spring = self._step_spring(lo)
        r = self._radii(t, lo, hi, spring=spring)
        X, Y, Z = self.x * r, self.y * r, self.z * r

        ca, sa = np.cos(angle), np.sin(angle)
        Xr = X * ca + Z * sa
        Zr = -X * sa + Z * ca
        tilt = 0.38
        ct, st = np.cos(tilt), np.sin(tilt)
        Yt = Y * ct - Zr * st
        Zt = Y * st + Zr * ct

        # self.zoom is fitted in __init__ so the worst-case frame still clears
        # the canvas; a clipped sphere reads as a broken square, not as loudness.
        zoom = self.zoom * (1.0 + 0.06 * lo)        # whole sphere breathes
        px = (Xr * zoom + 0.5) * s
        py = (Yt * zoom + 0.5) * s
        # Clipped to >=0: `bright` below raises (0.30 + 0.95*depth) to a
        # fractional power, which is only defined for a non-negative base.
        # The elastic stretch can push a particle's radius (and so Zt) well
        # outside the range depth was implicitly assumed to stay in before
        # that stretch existed -- an unclamped negative depth silently
        # produced NaN pixels here with no warning at the call site, which
        # then spread through the blur/bloom passes into much larger patches
        # of corrupted output.
        depth = np.clip((Zt + 1.5) / 3.0, 0.0, None)
        # Glow rides the same breath/heartbeat phase as the geometry, so the
        # orb visibly brightens as it "inhales" instead of only pulsing size:
        # coordinated motion+light is what makes a pulse read as one living
        # thing rather than two unrelated animations layered by coincidence.
        bright = ((0.30 + 0.95 * depth) ** 1.5 * GAIN * (1.0 + 0.75 * mid)
                 * (1.0 + 0.5 * self._life_pulse(t)))
        rim = np.sqrt(np.clip(Xr * Xr + Yt * Yt, 0, None)) / 1.15
        bright *= 1.0 + (0.9 + 0.7 * hi) * np.clip(rim, 0, 1) ** 3

        g = np.clip((px / s) * 0.55 + (py / s) * 0.45, 0, 1)[:, None]
        col = C_BLUE[None, :] * (1 - g) + C_MAGENTA[None, :] * g
        col = col + (C_CYAN - col) * (0.25 * depth[:, None] + 0.22 * hi)
        col = col * (bright * self.weight)[:, None]

        rgb = self._splat(px, py, col.astype(np.float32))
        return self._finish(self._bloom(rgb, mid), lv)

    def _splat(self, px: np.ndarray, py: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Accumulate points with bilinear subpixel placement.

        The previous version rounded each point to an integer pixel and stamped
        a fixed 3x3 blob. Rounding is what made the particles crawl between
        pixels instead of gliding, and the 3x3 blob is why a point was never
        smaller than a smudge. Distributing each point across its four
        neighbours by fractional position places it exactly, so a wireframe line
        resolves as a line. It is also cheaper: four taps instead of nine.
        """
        s = self.size
        ok = (px >= 0) & (px < s - 1) & (py >= 0) & (py < s - 1)
        px, py, col = px[ok], py[ok], col[ok]
        n = px.size
        if n == 0:
            return np.zeros((s, s, 3), np.float32)

        ix = px.astype(np.int32)
        iy = py.astype(np.int32)
        fx = (px - ix).astype(np.float32)
        fy = (py - iy).astype(np.float32)
        base = iy * s + ix
        flat = self._flat[: n * 4]
        flat[0:n] = base
        flat[n:2 * n] = base + 1
        flat[2 * n:3 * n] = base + s
        flat[3 * n:4 * n] = base + s + 1
        w = (((1 - fx) * (1 - fy)), (fx * (1 - fy)),
             ((1 - fx) * fy), (fx * fy))

        vals = self._vals[: n * 4]
        acc = np.empty((s * s, 3), np.float32)
        for c in range(3):
            cc = col[:, c]
            for j in range(4):
                np.multiply(cc, w[j], out=vals[j * n:(j + 1) * n])
            acc[:, c] = np.bincount(flat, weights=vals, minlength=s * s)
        return acc.reshape(s, s, 3)

    @staticmethod
    def _halve(a: np.ndarray) -> np.ndarray:
        """Box-downsample by 2 with strided adds.

        The obvious `reshape(h//2, 2, w//2, 2, -1).mean((1, 3))` measured 5.3 ms
        at 384 px: reducing over two tiny inner axes of a non-contiguous view
        defeats the cache. Four strided slices do the same arithmetic on
        contiguous-ish data an order of magnitude faster.
        """
        h, w = a.shape[0] - a.shape[0] % 2, a.shape[1] - a.shape[1] % 2
        b = a[:h, :w]
        return (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2]) * 0.25

    @staticmethod
    def _blur3(a: np.ndarray, passes: int = 2) -> np.ndarray:
        """Separable 1-2-1 blur, repeated. Two passes approximate a gaussian
        closely enough at bloom radii, at a fraction of PIL's cost."""
        for _ in range(passes):
            a = (a + np.roll(a, 1, 0) * 0.5 + np.roll(a, -1, 0) * 0.5) / 2.0
            a = (a + np.roll(a, 1, 1) * 0.5 + np.roll(a, -1, 1) * 0.5) / 2.0
        return a

    def _bloom(self, buf: np.ndarray, lv: float, mid_band: float = 0.0) -> np.ndarray:
        """Thresholded multi-scale bloom, entirely in numpy.

        Two changes from the PIL version, both aimed at definition rather than
        speed alone (though it is ~4x faster, which is what pays for the extra
        geometry):

        * **Only bright pixels bloom.** Blooming the whole image, as before,
          smeared every dim particle into its neighbours and was the main reason
          the dots read as mush. Halos now come from the hot cores only, so a
          particle keeps a crisp centre and still glows.
        * **No uint8 round-trip.** The old path quantized to 8-bit before
          blurring, which threw away exactly the low-amplitude detail that makes
          fine grid lines legible.
        """
        # Threshold AFTER downsampling, and combine both blur scales at half
        # res so only one upsample touches the full-size image. Every full-res
        # pass costs about a millisecond at 384 px, and this removes three.
        half_src = self._halve(buf)
        bright = np.maximum(half_src - _BLOOM_THRESHOLD, 0.0)
        half = self._blur3(bright, 2)
        quarter = self._blur3(self._halve(bright), 2)

        bloom = np.float32(1.0 + 0.35 * lv)          # louder = more bloom
        half *= np.float32(0.75) * bloom
        half += np.repeat(np.repeat(quarter, 2, 0), 2, 1)[:half.shape[0], :half.shape[1]] * (np.float32(0.60) * bloom)

        out = buf * np.float32(255.0)
        up = np.repeat(np.repeat(half, 2, 0), 2, 1)[:out.shape[0], :out.shape[1]]
        out += up * np.float32(255.0)
        return out

    def _finish(self, out: np.ndarray, lv: float) -> np.ndarray:
        """Tonemap + saturation + premultiplied-BGRA packing, in two passes.

        The naive form is four passes over the image:
            gain  = 255*(1-exp(-lum/knee)) / lum      (soft-knee tonemap)
            o     = out * gain
            grey  = mean(o)
            final = grey + (o - grey) * SAT
        Since `gain` is a per-pixel scalar it factors straight out of the mean,
        so the whole thing collapses to
            final = out*(SAT*gain) - mean(out)*((SAT-1)*gain)
        which is two fused multiply-adds. Same pixels, a third of the time.
        """
        s = self.size
        lum = _chan_max(out)
        # Table lookup instead of a per-pixel exp; see _TONE_LUT.
        idx = np.clip(lum * _LUT_SCALE, 0, _LUT_N - 1).astype(np.int32)
        gain = _TONE_LUT[idx]
        mean = _chan_mean(out)
        np.multiply(mean, gain * (_SAT - 1.0), out=mean)
        np.multiply(out, (gain * _SAT)[:, :, None], out=out)
        np.subtract(out, mean[:, :, None], out=out)
        np.clip(out, 0, 255, out=out)

        res = self._out
        res[:, :, :3] = out                     # already BGR: one contiguous copy
        res[:, :, 3] = _chan_max(out)           # premultiplied alpha = brightest channel
        return res
