# Prompt: Create the public repository methods overview and visual report

Use the following prompt with ChatGPT to generate the public-facing documentation for the `iMED-pe_challenge` repository.

---

We developed a collection of geometric and learned correspondence pipelines for the iMED Pose Estimation Challenge. I want you to write polished, technically accurate, public-facing documentation that explains all approaches we tried, why we tried them, what worked, what failed, and why the final method was selected.

The documentation is intended for a GitHub portfolio repository. It should be visually appealing, easy to understand, and useful to researchers who want to reproduce the work. This is separate from the short three-page challenge submission report: it may be more comprehensive, but it should remain readable.

Produce the complete contents of these four files:

1. `docs/METHODS_OVERVIEW.md`
2. `docs/methods_report.tex`
3. `scripts/make_report_plots.py`
4. `docs/figures/README.md`

Do not invent results, implementation details, citations, or dataset properties. If a detail is missing, mark it clearly as a placeholder or TODO. Do not imply that hidden-validation results are final challenge rankings.

## Repository and disclosure context

- Repository name: `iMED-pe_challenge`
- The repository should contain code, configurations, installation instructions, and reproducibility documentation.
- Raw challenge data, private paths, generated prediction files, evaluation JSON files, and organizer-provided material must not be committed.
- The challenge results have not yet been publicly released. Describe hidden-validation scores as internal challenge-server measurements and include an embargo/disclosure note.
- Avoid all machine-specific paths such as `/mnt/cluster/...` in generated documentation and scripts.
- Figures derived from challenge images should only be published if organizer rules permit it. Aggregate metric plots, trajectory plots without image content, and TikZ diagrams are preferred.

## Core pipeline conventions

The geometric methods estimate the relative pose between two endoscopes. When a stored transform follows

```text
X_camera = R X_reference + t,
```

the camera center is

```text
C = -R^T t.
```

Challenge evaluation uses trajectory-level Sim(3) alignment before computing ATE. Explain that this removes arbitrary global rotation, translation, and scale, but not errors in trajectory shape.

## Methods to explain

Explain each method in simple language first, followed by a compact technical description. Make clear which ideas were retained and which were negative experiments.

### Baseline

Describe the provided/basic reference method only at a high level unless more implementation information is supplied. Released TEST mean ATE: `2.0421 mm`.

### Method 1 — Temporal stereo visual odometry

Uses ALIKED + LightGlue for temporal matching in E2-L and stereo information to estimate local relative motion. It supplies smooth local motion increments but accumulates drift.

Results:

- TRAIN mean ATE: `1.1580 mm`
- released TEST mean ATE: `1.5343 mm`

### Method 1.5A — Rotation-anchored temporal method

An intermediate hybrid that introduced stronger orientation/anchor information into the temporal trajectory. Released TEST mean ATE: `1.5292 mm`. Present this as an exploratory bridge rather than a final method.

### Method 2A — E1 triangulation followed by cross-endoscope PnP

Pipeline:

```text
E1-L <-> E1-R stereo matches
        -> triangulate points in E1
E1-L <-> E2-L cross-endoscope matches
        -> intersect exact E1-L feature IDs
3D(E1) <-> 2D(E2)
        -> solvePnPRansac
        -> T_E2_from_E1
```

Uses ALIKED + LightGlue with 2048 features and a session-level E1 stereo calibration.

Results:

- TRAIN mean ATE: `1.2002 mm`
- released TEST mean ATE: `1.4000 mm`
- hidden validation: `2.474 mm`

### Method 2B — Symmetric E2 triangulation branch

Pipeline:

```text
E2-L <-> E2-R stereo matches
        -> triangulate points in E2
E2-L <-> E1-L cross-endoscope matches
        -> intersect exact E2-L feature IDs
3D(E2) <-> 2D(E1)
        -> solvePnPRansac
        -> invert pose convention
```

Results:

- TRAIN mean ATE: `1.0996 mm`
- released TEST mean ATE: `1.3373 mm`
- hidden validation: `2.484 mm`

### Method 3 — VGGT experiments

Explain that VGGT-based/global learned geometry variants were investigated but did not outperform the dedicated stereo-PnP and temporal fusion pipeline. A reported Method 3B released TEST mean ATE was `1.6986 mm`. Do not fabricate details of Method 3A/3B beyond what is supported by the repository.

### Method 4A — Stereo-VO and absolute-pose fusion

Method 4A combines:

- Method 1 local VO increments,
- Method 2A absolute position observations,
- Method 2B absolute position observations,
- a robust position optimizer,
- a frozen rotation source and frame-zero normalization.

Explain two variants if supported by code:

- Method 4A0: TRAIN `1.0416 mm`, TEST `1.2916 mm`
- Method 4A1: TRAIN `0.8801 mm`, TEST `1.1546 mm`, hidden `2.316 mm`

Method 4A1 is the central base fusion architecture reused by later experiments.

### Point 3 — Spatially balanced ALIKED + LightGlue

Hypothesis: correspondences might be numerous but concentrated in a small anatomical region. Tested 4x4 and 6x6 image grids with per-cell caps.

Observed coverage was already broad. Representative median statistics included:

```text
pair                 matches   occupied 4x4   max-cell fraction   covariance ratio
E1 stereo              1047        1.00             0.134              0.782
E2 stereo               972        1.00             0.125              0.737
cross E1L-E2L           739        0.94             0.166              0.749
temporal E2L           1100        1.00             0.121              0.750
```

Balancing slightly changed local development metrics but failed full TRAIN confirmation:

- original 2A: TRAIN overall `1.2002`, session-balanced `1.7231`, worst `3.8866`
- balanced 2A candidate: overall `1.1981`, session-balanced `1.7251`, worst `3.9162`

Conclusion: explicit pruning was not retained.

### Point 4 — XFeat

Official sparse XFeat with approximately 2048 features was substituted for ALIKED + LightGlue while preserving downstream geometry.

Cross-endoscope comparison:

```text
ALIKED + LightGlue: about 471 matches, 203 MAGSAC inliers, ratio 0.576
XFeat sparse:       about 843 matches, 171 MAGSAC inliers, ratio 0.207
```

XFeat produced more and broader matches but substantially lower geometric purity for cross-endoscope matching. Full XFeat inference hurt Method 2A/2B. XFeat session calibration was more useful than full replacement, and its resulting Method-4A hybrid was competitive.

Result:

- Method 4A + XFeat TRAIN mean: `0.8664 mm`
- released TEST mean: `1.1563 mm`
- hidden validation: approximately `2.310 mm`

Explain the key lesson: raw match count is not a proxy for pose quality.

### Point 5 — EfficientLoFTR

EfficientLoFTR produced strong pairwise cross-endoscope geometry at confidence >= 0.8:

```text
ALIKED + LightGlue: 471 matches, 203 MAGSAC inliers, ratio 0.576
EfficientLoFTR:     754 matches, 480 MAGSAC inliers, ratio 0.627
```

The first L1 hybrid associated LoFTR coordinates to sparse ALIKED stereo keypoints. This interoperability layer retained only about 16-19% of matches and left only roughly 13-18 median PnP inliers, so L1 was rejected.

A pure EfficientLoFTR multiview experiment then used LoFTR for both stereo and cross-camera pairs and associated coordinates in the shared source image. It produced healthier three-view tracks but standalone trajectory accuracy was worse than original Method 2:

- LoFTR-MV 2A TRAIN mean `1.3622`, session-balanced `1.9242`, worst `4.6149`
- LoFTR-MV 2B TRAIN mean `1.1940`, session-balanced `1.5449`, worst `2.5733`

Despite weaker standalone ATE, replacing only Method 2A observations inside Method 4A helped released TEST:

- Method 4A + LoFTR-2A TRAIN mean `0.8968 mm`
- released TEST mean `1.1402 mm`
- released TEST session-balanced `1.1715 mm`
- released TEST worst session `2.0349 mm`
- hidden validation `2.308 mm`

Replacing both 2A and 2B was worse on hidden validation (`2.332 mm`). Explain that branch diversity can matter more than standalone accuracy.

### Point 6 — RoMa-2A exact-source tracking

RoMa dense warps were sampled at the exact same ALIKED E1-L source keypoints for both E1-R and E2-L. This constructed exact three-view identities without nearest-neighbor merging:

```text
ALIKED E1-L source point i
       /                 \
RoMa warp             RoMa warp
   to E1-R               to E2-L
```

Forward-backward cycle consistency and RoMa certainty filtering were evaluated. Existing E1 session calibration and downstream PnP were preserved.

Results:

- RoMa-2A TRAIN mean `1.0057 mm`
- RoMa-2A released TEST mean `1.3089 mm`
- RoMa-2A hidden validation `2.341 mm`
- Method 4A + RoMa-2A TRAIN mean `0.8477 mm`
- released TEST mean `1.1465 mm`
- hidden validation `2.310 mm`

Explain that exact source identity solved the sparse/dense interoperability problem, but full RoMa was computationally expensive.

### Method 4C — Multi-observer geometric consensus

Aligned original 2A, original 2B, LoFTR-2A, and RoMa-2A trajectories using prediction-only robust Sim(3), then tested geometric-median consensus and consensus-aware VO optimization.

Observer aligned error correlations were moderately high, approximately `0.66-0.72`, indicating partially shared/systematic failure modes.

TRAIN results:

```text
method                 mean     session-balanced    worst     frame p95
4C geometric median    0.9960       1.2441          2.7946      2.9324
4C VO, uniform         0.8844       0.9910          1.8102      2.6784
4C VO, consensus       0.8834       1.0339          2.1598      2.6541
```

Conclusion: simple geometric consensus could not determine which observer was correct and was not retained.

### Motion-shape prior baseline

An image-free baseline built canonical circular, left-right, and zoom-in trajectories from TRAIN ground truth using strict leave-one-session-out templates. Mean, median, and medoid templates were evaluated, with a wrong-class control.

```text
template       mean ATE   session-balanced   worst
mean            2.5555        4.6539         9.5473
median          2.5448        4.5599         9.4689
medoid          2.5475        4.6742        10.0394
wrong class     2.5318        4.5564         9.9044
```

The wrong-class control performed similarly, showing that motion labels alone did not provide a useful trajectory prior.

### Method 5 / Method 5R — Session stereo correction

Optimized a small session-level correction around the existing E1 stereo calibration using RoMa three-view reprojection evidence. Rotation and translation direction were corrected while preserving baseline magnitude. The strict gate rejected mixed per-view reprojection changes; Method 5R relaxed only the validity rule while keeping the optimizer frozen.

Method 5R + RoMa-2A results:

- TRAIN mean `1.0002 mm` versus original RoMa `1.0057`
- TRAIN session-balanced `1.3099` versus `1.3025`
- TRAIN worst `3.2603` versus `3.1589`
- released TEST mean `1.3016 mm` versus `1.3089`

Frozen Method 4A + Method5R-RoMa results:

- TRAIN mean `0.8444 mm`
- TRAIN session-balanced `0.8777 mm`
- TRAIN worst `1.2356 mm`
- released TEST mean `1.1447 mm`
- released TEST session-balanced `1.1787 mm`
- released TEST worst `2.0371 mm`

The correction produced only a small gain and did not clearly improve the LoFTR branch. Present this as a narrowly positive but not transformative experiment.

### Final Method 6 — Reliability-routed mixture of pose experts

This is the selected final method.

Experts:

- original Method 2A absolute poses,
- original Method 2B absolute poses,
- LoFTR-2A absolute poses,
- Method 1 remains a relative VO constraint rather than an absolute expert.

All absolute experts are converted to camera centers and aligned in a common prediction-only coordinate system. A lightweight supervised router predicts a soft reliability distribution per frame from inference-time diagnostics only. Features include PnP inlier statistics, reprojection error, track counts, calibration diagnostics, pose jumps, cross-expert disagreements, and innovation relative to Method-1 VO. Session ID, sequence name, motion class, hidden scores, and GT-derived features are excluded.

Strict physical-session LOSO is used: the held-out session contributes neither labels nor scaler/router training data. GT is used only to create TRAIN reliability targets and evaluate held-out predictions.

Frozen router configuration:

```text
model: multinomial logistic regression
features: R3, all diagnostic/disagreement/VO features
feature count: 224
target: soft reliability target
scaling: median/MAD
smoothing: none, alpha = 1.0
entropy gate: disabled
fusion: Method 6B
lambda_vo: 5
seed: 2026
training set: 61 sequences, 6095 frames, 7 physical sessions
```

The router outputs:

```text
p_2A(t), p_2B(t), p_LoFTR(t), with p >= 0 and sum p = 1.
```

These probabilities weight the absolute-position residuals in the frozen Method-4A-style robust optimizer, while Method-1 VO preserves local trajectory shape.

Oracle diagnostic:

```text
method          mean     session-balanced    worst    frame p95
always 2A       1.2002       1.7231          3.8866    3.3049
always 2B       1.0996       1.4591          3.2645    3.0178
always LoFTR    1.3622       1.9242          4.6149    3.5014
hard oracle     1.0713       1.4755          3.0425    3.0327
soft oracle     0.7839       0.8936          1.6052    2.3842
```

Explain why the soft oracle is important: the best trajectory often lies between experts rather than at one hard expert choice.

LOSO ablations:

```text
variant                          mean    session-balanced   worst    frame p95
Logistic R1, Method6B            0.8712       0.9164        1.3895     2.6003
Logistic R2, Method6B            0.8736       0.9415        1.6042     2.5969
Logistic R3, Method6B            0.8707       0.9156        1.4191     2.5914
HistGradientBoosting R3          0.8792       0.9530        1.6313     2.6429
Method6A, weighted average       1.0127       1.2365        2.4695     2.8369
Method6C, entropy gated          0.9229       0.9511        1.4011     2.6807
```

Smoothing ablation for Method6B R3:

```text
alpha 1.0: mean 0.8707, session-balanced 0.9156, worst 1.4191
alpha 0.7: mean 0.8712, session-balanced 0.9178, worst 1.4182
alpha 0.5: mean 0.8718, session-balanced 0.9201, worst 1.4250
alpha 0.3: mean 0.8728, session-balanced 0.9243, worst 1.4461
```

No smoothing was retained. Router-only accuracy was modest (`0.4061`, balanced accuracy `0.3985`), but soft weights improved trajectory fusion; emphasize that top-1 classification accuracy is not the final objective.

Final released TEST comparison:

```text
method                 mean    session-balanced   worst       CV      frame p95
Method 4A              1.1546      1.1889         2.0488     0.4596    3.1710
4A + LoFTR-2A          1.1402      1.1715         2.0349     0.4636    3.1481
4A + RoMa-2A           1.1465      1.1806         2.0371     0.4608    3.2105
5R + 4A + RoMa-2A      1.1447      1.1787         2.0371     0.4629    3.2129
Method 6B              1.1357      1.1663         2.0296     0.4645    3.0973
```

Method 6B per-session released TEST ATE:

```text
session_002: 0.7074 mm, 5 sequences
session_004: 0.7031 mm, 6 sequences
session_005: 1.2251 mm, 3 sequences
session_007: 2.0296 mm, 5 sequences
```

Router/fusion post-processing runtime over 1895 frames:

```text
router inference:        0.0078 ms/frame
probability processing:  0.0079 ms/frame
fusion optimization:     4.0897 ms/frame
total post-processing:  12.0736 seconds
```

The hidden-validation score for Method 6 is not yet known. State it as pending rather than inventing a value.

## Hidden-validation table

Include this table with a clear note that these are challenge-server validation measurements rather than final public rankings:

```text
Method 2A                         2.474 mm
Method 2B                         2.484 mm
Method 4A                         2.316 mm
Method 4A + XFeat                ~2.310 mm
Method 4A + LoFTR-2A              2.308 mm
Method 4A + LoFTR-both            2.332 mm
RoMa-2A standalone                2.341 mm
Method 4A + RoMa-2A               2.310 mm
Method 6B                         pending
```

## Required narrative

Organize the story around these findings:

1. Stereo-PnP provides absolute anchors but is vulnerable to session-dependent calibration and correspondence failures.
2. Temporal stereo VO preserves local shape but drifts.
3. Frozen robust fusion is much stronger than either component alone.
4. More matches are not necessarily better: XFeat increased density while lowering geometric purity.
5. EfficientLoFTR greatly improved pairwise robust inliers, but consistent multiview construction and branch diversity mattered.
6. Exact-source RoMa tracks solved correspondence-identity ambiguity but were expensive and did not clearly beat LoFTR on hidden validation.
7. Hand-designed geometric consensus failed because observer failures remained correlated and consensus could not know which estimate was correct.
8. A lightweight learned reliability router offered a better way to convert diagnostics and disagreements into soft per-frame expert weights.
9. Method 6 was selected because it improved released TEST mean, session-balanced mean, worst-session ATE, and frame-tail error without adding another neural model beyond the existing LoFTR pipeline.

Be honest about limitations:

- only seven physical TRAIN sessions were available for router LOSO;
- hidden-distribution generalization remains uncertain;
- the router's top-1 classification accuracy is modest;
- several methods share calibration and geometric biases;
- RoMa runtime is high;
- released TEST is not the hidden challenge set;
- Method 6 hidden performance is pending.

## Required diagrams

Use TikZ in `docs/methods_report.tex`. The LaTeX must compile without shell escape and should use common packages only (`tikz`, `pgfplots`, `booktabs`, `subcaption`, `amsmath`, `geometry`, `xcolor`, and related standard packages).

Create at least these TikZ figures:

1. Overall method-family diagram:

```text
images
  -> temporal VO (Method 1)
  -> absolute stereo-PnP branches (2A, 2B)
  -> alternative 2A observers (XFeat, LoFTR, RoMa)
  -> robust fusion (Method 4A)
  -> reliability router (Method 6)
  -> final trajectory
```

2. Method 2A vs Method 2B symmetric geometry diagram showing stereo triangulation, shared feature identity, PnP, and inversion for 2B.

3. Method 4A diagram showing absolute anchors plus Method-1 VO edges.

4. Method 6 mixture-of-experts diagram showing diagnostic features, softmax router probabilities, weighted absolute residuals, VO prior, and final optimization.

Use consistent colors across diagrams:

- blue: image/matcher operations
- orange: triangulation/PnP geometry
- green: temporal VO
- purple: fusion/router
- gray/red: rejected or negative experiments

## Required generated plots

The generated `scripts/make_report_plots.py` must be standalone and must not depend on private output JSON files. Store the numerical results above as clearly labeled constants near the top of the script. It should create publication-quality PDF and PNG files under an output directory supplied by `--output-dir`.

Generate at least:

1. `test_method_comparison.pdf/png`
   - grouped comparison of released TEST mean, session-balanced mean, and worst-session ATE for Method 4A, 4A-LoFTR, 4A-RoMa, Method5R-4A-RoMa, and Method6B.

2. `method6_loso_ablation.pdf/png`
   - compare Logistic R1/R2/R3, HistGB R3, Method6A, and Method6C.

3. `method6_session_ate.pdf/png`
   - bar plot of Method6B ATE for sessions 002, 004, 005, and 007.

4. `matcher_geometry.pdf/png`
   - ALIKED+LightGlue, XFeat, and EfficientLoFTR cross-camera match count, robust inlier count, and inlier ratio. Use separate panels or axes so ratios are readable.

5. `method_progression.pdf/png`
   - a clear chronological plot of released TEST mean ATE from baseline through Method6B. Clearly distinguish negative experiments and final retained methods.

6. `oracle_headroom.pdf/png`
   - compare always-2A, always-2B, always-LoFTR, hard oracle, soft oracle, and Method6B LOSO metrics.

Use a colorblind-friendly palette, readable labels, sensible ordering, vector PDF output, 300-dpi PNG output, and no seaborn dependency unless explicitly listed as a project dependency.

## Optional trajectory figures

In `METHODS_OVERVIEW.md` and the LaTeX report, reserve optional figure slots for trajectory plots. Use captions that explain:

- GT and predictions are aligned using the official trajectory-level Sim(3) evaluator;
- plots show camera-center trajectories;
- axes are in aligned millimetres;
- data-derived figures should only be published if challenge rules permit.

Suggested examples:

- a difficult Session 007 circular sequence;
- a Session 005 sequence illustrating calibration/correspondence difficulty;
- columns for GT, Method 2A, Method 4A, 4A-LoFTR, and Method6B;
- alternatively overlay all trajectories in one panel plus error-over-time in a second panel.

Do not include raw endoscopic images unless their publication is explicitly allowed.

## `docs/METHODS_OVERVIEW.md` requirements

Write a polished GitHub document with:

- a short project overview;
- a compact results table near the top;
- a Mermaid overview diagram as a GitHub-native fallback;
- one clear subsection per method/experiment;
- short “Hypothesis / Outcome / Decision” blocks for negative experiments;
- an explanation of Method 6 suitable for readers who do not know pose estimation;
- links/placeholders for generated plots;
- reproducibility notes;
- limitations and disclosure status;
- a concise final takeaway.

Avoid turning it into a chronological lab notebook. Group methods into: base geometry, correspondence alternatives, fusion/refinement experiments, final routed ensemble.

## `docs/methods_report.tex` requirements

Write a self-contained 7-10 page technical report suitable for rendering to PDF. Include:

- title and abstract;
- challenge/problem formulation;
- transform convention and Sim(3) evaluation note;
- method progression;
- detailed Method 4A and Method 6 formulations;
- equations for weighted absolute and VO residual optimization;
- experiments and ablations;
- released TEST and hidden-validation tables;
- discussion of negative results;
- runtime/deployment discussion;
- limitations;
- conclusion;
- bibliography placeholders only where citations need to be supplied.

Use `\IfFileExists` or another graceful mechanism for optional generated plot files so the document can compile before plots are generated. Do not use absolute paths.

For Method 6, include an objective of the form:

```latex
\min_{\{C_t\}}
\sum_{t,m} p_m(t)\,\rho\!\left(\|C_t-\tilde C_{m,t}\|_2^2\right)
+
\lambda_{\mathrm{VO}}
\sum_t \rho\!\left(\|C_{t+1}-C_t-\Delta C_t^{\mathrm{VO}}\|_2^2\right),
```

where `\tilde C_{m,t}` are prediction-only aligned expert camera centers and `p_m(t)` are router probabilities.

Explain explicitly that the router is supervised on TRAIN GT, but inference uses only geometry diagnostics and predicted disagreement.

## `docs/figures/README.md` requirements

Explain:

- which figures are fully synthetic/aggregate and safe to regenerate;
- which trajectory figures depend on challenge-derived predictions/GT;
- how to run `scripts/make_report_plots.py`;
- where figures should be placed;
- that raw dataset images should not be committed without permission;
- recommended PDF/PNG formats.

## Output format

Return each requested file in a separate fenced code block. Precede each block with the exact relative path as a Markdown heading. Do not omit code for brevity. Ensure the Python script and LaTeX are syntactically complete.

---

