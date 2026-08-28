# Method 5-R — relaxed downstream validation

Method 5-R does not alter the Method-5 optimizer. It changes only the global,
GT-free deployment rule. A correction is exported when the optimizer converged,
the correction is inside the existing bounds, split-half and cross-sequence
stability pass, and the total robust objective decreased. Individual-view
reprojection p90 values remain diagnostics but are not vetoes.

Completed Method-5 fits for sessions 002, 004, and 005 are reused directly.
Only sessions 001, 003, 006, and 007 require the unchanged caching/refinement
stages before the seven-session rule can be applied.

