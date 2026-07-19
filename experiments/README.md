# experiments/

Unintegrated visual explorations. None of this is part of the supported product: no
tests cover it, nothing in `say.py` or `hub.py` imports it, and it may break without
notice or be deleted outright.

- **`constellation.py`** — one orb per live agent, arranged as a constellation.
- **`neuron_render.py`** — an agent drawn as a neuron that fires when it speaks.
- **`orb_gpu.py`** — the particle orb on the GPU. Requires `torch` and a working CUDA
  setup (`pip install -e ".[gpu]"`); it will not run on CPU-only machines.

Run them directly if you want to see where the visuals might go. Do not build on them.
