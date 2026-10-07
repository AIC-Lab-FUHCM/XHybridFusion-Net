# Official BiFormer source

- Repository: https://github.com/rayleizhu/BiFormer
- Branch: `public_release`
- Commit: `1697bbbeafb8680524898f1dcaac10defd0604be`
- Paper: BiFormer: Vision Transformer with Bi-Level Routing Attention, CVPR 2023
- License: MIT, included in `LICENSE`

The `models/` and `ops/` directories are copied from the official repository.
The local `__init__.py` only adds this vendored repository directory to Python's
module search path because the original source imports `models` and `ops` as
top-level modules.
