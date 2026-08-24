# Third-party software and weights

This repository does not redistribute third-party source trees or model
weights. `third_party/manifest.yaml` records the exact revisions used.

| Component | Upstream | Evaluated revision/checkpoint |
|---|---|---|
| Challenge baseline | `imed-challenge/imedpe` | local base commit `a2267625c51f52c5e14ec5f1cd0610ec57a44eaf` |
| LightGlue / ALIKED | `cvg/LightGlue` | `eb42fee2d71449efb0aa5c10549752b5d75384d8` |
| XFeat | `verlab/accelerated_features` | `e92685f57f8318b18725c5c8c0bd28c7fe188d9a` |
| EfficientLoFTR | `zju3dv/EfficientLoFTR` | `ffd4a4644064354468eb1f0c7a3e732233cb732f`, `eloftr_outdoor.ckpt` |
| VGGT | `facebookresearch/vggt` | `a288dd0f14786c93483e45524328726ab7b1b4ce`, `facebook/VGGT-1B` |

Verified hashes for locally evaluated standalone checkpoints:

```text
XFeat xfeat.pt:
0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b

EfficientLoFTR eloftr_outdoor.ckpt:
0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878
```

LightGlue/ALIKED and VGGT may retrieve weights through their official APIs on
first use. Cache them before offline or Docker execution. Review every upstream
license and model card before redistribution. A checksum verifies identity; it
does not grant redistribution rights.

