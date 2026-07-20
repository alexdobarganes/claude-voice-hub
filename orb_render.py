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
N_SHELL = 9000      # particles forming the sphere's surface
N_DUST = 2600       # looser speckle floating just outside it
GAIN = 1.85         # mesh spreads the same light over many more points, so the
                    # per-point gain has to come up to keep the old punch

# Palette in BGR order (see module docstring): blue shell -> magenta shell,
# cyan lift on the lit/front side.
C_BLUE = np.array([255, 130, 70], dtype=np.float32) / 255.0
C_CYAN = np.array([255, 200, 80], dtype=np.float32) / 255.0
C_MAGENTA = np.array([210, 60, 255], dtype=np.float32) / 255.0

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
    def __init__(self, size: int = SIZE, seed: int = 7,
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
        self.amp = (0.05 + 0.11 * rng.random(n)).astype(np.float32)
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
        worst = max(float(np.percentile(np.abs(self._radii(t, 1.0, 1.0)), 99.0))
                    for t in np.linspace(0.0, 4.0, 24))
        self.zoom = (0.5 - BLOOM_MARGIN) / (worst * (1.0 + 0.06))

    def _radii(self, t: float, lo: float, hi: float) -> np.ndarray:
        """Per-point radius at time `t` for the given band levels. Shared by
        render() and the canvas fit so the two can never disagree."""
        # A standing wave in x/y/z used to push the radius around, which is what
        # pulled the sphere into an egg: three sinusoids keyed to position add up
        # to fat and thin lobes that sit still in space, so the silhouette stops
        # being round. The breathing below is uniform instead — every point
        # moves by the same factor, so the shape stays a sphere at every instant
        # and only its size changes.
        r = self.base_r.copy()
        # Per-particle drift, small and only on the halo, so the surface itself
        # keeps its radius.
        r[self.n_mesh:] += (self.amp
                            * np.sin(2 * np.pi * (self.k * t + self.off)))[self.n_mesh:]
        r *= 1.0 + 0.045 * np.sin(2 * np.pi * 2.0 * t)
        r += 0.16 * lo * self.react
        if hi > 0.02:
            r += 0.055 * hi * np.sin(self.shimmer * (12.0 * t) + self.off * 6.283)
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
        r = self._radii(t, lo, hi)
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
        depth = (Zt + 1.5) / 3.0
        bright = (0.30 + 0.95 * depth) ** 1.5 * GAIN * (1.0 + 0.75 * mid)
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
