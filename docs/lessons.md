# Lessons from real-time rendering

Notes collected while building the audio-reactive orb, plus a parked experiment
in rendering a face from an image that is not part of this repository. The
findings below are the transferable part: each one cost time to learn and is
stated so it can be checked, or contradicted, by someone else's measurements.

## Self-normalising metrics will lie to you

This is the most expensive lesson here, and the most general.

The face experiment scored a render against a reference by normalising **each
side by its own percentiles** before comparing. The intent was reasonable: make
the score robust to differences in overall exposure so it would measure shape
rather than brightness.

The effect was that the score became blind to exposure entirely. The
optimisation loop converged happily, reporting **~98% agreement**, on a render
that was too dark to see. Every step that made the image dimmer was free,
because the metric renormalised the darkness away before comparing. The number
went up while the artifact got worse, and nothing in the loop could notice.

The mechanism generalises: **any normalisation applied independently to both
sides of a comparison removes that dimension from the objective.** If you divide
out scale, you can no longer optimise scale. If you divide out mean brightness,
brightness is now unconstrained and the optimiser is free to destroy it in
exchange for gains elsewhere.

How to avoid it:

- Name, explicitly, which dimensions your metric is deliberately blind to. If
  you cannot name them, you do not know what you are optimising.
- Keep at least one **absolute** term alongside any normalised one (an unscaled
  error, a mean-level check, a hard range assertion). It does not need to be the
  headline number; it needs to be able to fail.
- **Look at the output.** A converged score is a hypothesis, not a result. In
  this case a single glance at the image falsified 98% instantly.
- Be suspicious of a metric that improves monotonically and fast. Real objectives
  fight back.

A second, narrower version of the same problem showed up in the comparison
itself: both images were halftones with different dot phase, so a per-pixel
comparison was partly measuring dot misalignment, not shape. Blurring both past
the dot pitch before comparing showed that of ~2% disagreement, 1.4% was phase
and only 0.6% was actual shape error. If two signals are sampled on different
lattices, low-pass both before you compare them.

## Adding geometry is cheap; adding pixels is expensive

Measured on the orb's CPU renderer (numpy, no GPU):

- **~90% of frame time** went to post-processing over the full image (bloom at
  multiple scales, tonemap, handing the frame to the compositor).
- **~6%** went to drawing the particles themselves.

Two practical consequences, both counter-intuitive if you assume cost scales
with scene complexity:

- **Particle count is nearly free.** Dropping from 11,600 points to 9,300 moved
  the frame rate by 0.3 fps (56.3 → 56.0). So pick the count that looks right,
  not the one that seems cheap. `orb_render.py` runs 9,000 shell particles plus
  2,600 dust for that reason.
- **Resolution is the real budget.** 384 px turned out to be the ceiling for
  this renderer at 60 fps (`SIZE = 384` in `orb_render.py`). Going larger means
  dropping to 30 fps or moving the post-process to the GPU. Smaller was worse
  for a different reason: on a 4K screen, 256 px reads as a thumbnail.

The corollary for optimisation work: profile before you thin the scene. The
obvious lever (fewer things) was the wrong one; the win came from making the
post-process cheaper (thresholded bloom so only bright pixels glow, blurring at
1/2 and 1/4 resolution, and cutting redundant full-image reductions), which then
paid for the higher resolution.

## Structure comes from data, not from procedural bumps

The face experiment's first approach sculpted a face procedurally out of
gaussian lobes. It can be tuned indefinitely and still reads as a mask, because
what makes a face recognisable is a great deal of small, irregular, mutually
constrained structure that no small set of smooth primitives reproduces.

Deriving the shape from an actual image instead solved in one step what tuning
had failed at over many. The general form: when a target has irregular
structure that you cannot enumerate, sample it rather than model it.

## Tone in a halftone is carried by dot area, not dot brightness

Radius ∝ √luminance, at constant dot brightness. The alternative (fixed dot
size, modulated brightness) produces grey mush; modulating size produces crisp
discs with black between them, which is what reads as halftone.

Related: the lattice must be **regular**. Rows of dots curve as they cross the
relief, and that curvature is what the eye reads as a surface. The same points
scattered at random read as noise, no matter how correct their density is.

Neither of these was visible from summary statistics. They became obvious only
by zooming into the reference and comparing patches side by side.

## Two smaller ones

- **Isolate a subject by where marks exist, not by how bright they are.** Using a
  local-maximum test rather than a brightness threshold matters because a
  brightness threshold eats the shadowed side and cuts the subject in half.
- **Calibrate multiplicatively when the pipeline has gain and a tonemap.** An
  additive correction converged at roughly 1% per pass, because the error it was
  correcting was proportional to the signal, not offset from it.
