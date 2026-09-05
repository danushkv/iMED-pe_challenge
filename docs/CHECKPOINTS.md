# Model checkpoints and pretrained weights

This repository does not redistribute third-party model weights. The table
below records the exact model variants used, where they are expected locally,
and the official download source. Source revisions are frozen in
[`../third_party/manifest.yaml`](../third_party/manifest.yaml).

Run the setup helper from the repository root after reviewing the relevant
upstream model license:

```bash
bash third_party/setup_models.sh --lightglue
bash third_party/setup_models.sh --xfeat
bash third_party/setup_models.sh --loftr
bash third_party/setup_models.sh --roma
bash third_party/setup_models.sh --vggt
```

The helper clones the evaluated commits and verifies the hashes that were
recorded. Some upstream libraries download weights on first model creation;
perform that step once with network access before offline or Docker inference.

## Evaluated checkpoints

| Component | Exact model/file | Expected location | Official source | Verified SHA-256 |
|---|---|---|---|---|
| ALIKED | `aliked-n16.pth` | PyTorch hub cache, or a bundled offline cache | [ALIKED N16 checkpoint](https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth) | `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2`* |
| LightGlue for ALIKED | `aliked_lightglue.pth` (`v0.1_arxiv`) | PyTorch hub cache | [LightGlue ALIKED checkpoint](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth) | Not recorded in the research log |
| XFeat sparse | `xfeat.pt` | `third_party/accelerated_features/weights/xfeat.pt` | [official XFeat checkpoint](https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt) | `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b` |
| EfficientLoFTR | `eloftr_outdoor.ckpt`, full outdoor model | `third_party/EfficientLoFTR/weights/eloftr_outdoor.ckpt` | [official model card](https://huggingface.co/zju-community/efficientloftr) or [upstream download folder](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q) | `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878` |
| Full RoMa | `roma_outdoor.pth` | `third_party/RoMa/.torch/hub/checkpoints/roma_outdoor.pth` | [official RoMa outdoor checkpoint](https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth) | `c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba` |
| RoMa backbone | DINOv2 ViT-L/14, `dinov2_vitl14_pretrain.pth` | `third_party/RoMa/.torch/hub/checkpoints/dinov2_vitl14_pretrain.pth` | [official DINOv2 ViT-L/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| RoMa source detector | ALIKED N16, `aliked-n16.pth` | `third_party/RoMa/.torch/hub/checkpoints/aliked-n16.pth` | [ALIKED N16 checkpoint](https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth) | `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2` |
| VGGT | `facebook/VGGT-1B`, `model.pt` | Hugging Face cache or a configured offline cache | [VGGT-1B model card](https://huggingface.co/facebook/VGGT-1B) / [model.pt](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) | Not recorded in the research log |

\*This is the hash of the ALIKED N16 file verified for the RoMa-2A source
detector. Verify any separately downloaded LightGlue/ALIKED cache before use.

## Project-trained Method 6 router

Method 6B uses one small project-trained artifact, which is distributed in this
repository because it contains only logistic coefficients, feature ordering,
and robust scaling values—no images, poses, or per-sequence features.

| File | Location | SHA-256 |
|---|---|---|
| Exact scikit-learn 1.7.2 artifact | `models/method6/router.pkl` | `7b4c167d762e32f809455cd735c8ec40d31e7f4cbb0d5e7e4fb7545687298da2` |
| Version-independent parameter export | `models/method6/router.npz` | `8841893c2c5096cb8287de8ccb67b49a632889a361de55ac53199a8e7b207254` |
| Feature schema and metadata | `models/method6/router.json` | `73b0d44ce02bf7beb61ab859449769e1826300b6a5fa0334f75eda45306e723f` |

Prefer the NPZ/JSON export for deployment. Only load pickle files obtained from
a trusted checkout.

## Verify local files

The setup helper verifies XFeat, EfficientLoFTR, and RoMa assets when they are
present. The project router can be checked with:

```bash
sha256sum models/method6/router.pkl \
          models/method6/router.npz \
          models/method6/router.json
```

Checkpoint availability does not imply unrestricted redistribution. Review the
license and model card linked for each model before packaging or publishing
weights.
