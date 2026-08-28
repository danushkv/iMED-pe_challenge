# Point 6 — RoMa-2A

This isolated experiment uses ALIKED only to select 2048 source pixels in
E1-L. A single full-RoMa dense warp maps those exact pixels to E1-R and a
second dense warp maps the same IDs to E2-L. No nearest-neighbour merge of
independently detected feature sets is permitted.

The experiment is intentionally staged:

1. verify official RoMa loading, pixel coordinates, certainty and cycles;
2. inspect exact-index E1-L/E1-R/E2-L TRAIN certainty/cycle distributions;
3. run three representative TRAIN sequences;
4. freeze certainty/cycle settings and confirm on all TRAIN sessions;
5. pass only RoMa-2A observations into frozen Method 4A;
6. run released TEST once after TRAIN selection.

The sequence runner is `python -m experiments.roma_2a.run_method2a_roma`.
It delegates all geometry after match construction to the unchanged official
Method-2A implementation in `src/imcpe/methods/cross_stereo_pnp`.

The frozen full-TRAIN configuration selected from the representative screen is
certainty >= 0.20 and forward/backward cycle error <= 2 original-image pixels.

`pose.txt` is never read by correspondence generation or prediction. Existing
Method-2A E1 stereo calibration and all downstream triangulation/PnP settings
remain unchanged.
