# Method 6 staged runbook

Run all commands from the repository root in the locked Python 3.10
environment. Method 6 consumes previously generated predictions and diagnostics;
router development does not rerun a feature matcher.

The reproducible command sequence and configurable output roots are documented
in [`../../docs/RUN_METHODS.md`](../../docs/RUN_METHODS.md#method-6b--selected-routed-ensemble).
The intended order is:

1. generate frozen Method 1, Method 1.5A, Method 2A, Method 2B, and LoFTR-2A inputs;
2. build the TRAIN feature dataset with TRAIN-only reliability targets;
3. evaluate oracle bounds;
4. run strict physical-session LOSO router selection;
5. fit the selected logistic R3 router on all TRAIN sessions;
6. build target-free features for the evaluation split;
7. apply the frozen router and routed VO-constrained optimizer;
8. evaluate only after prediction is complete.

Released TEST and hidden results must not select the router configuration. The
bundled `models/method6/` artifact may be used to reproduce final inference
without retraining.
