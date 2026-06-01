# Goal

Hybrid V2 improved ADE and drift.

However, visual inspection suggests loss of sharp turning geometry.

Goal:

evaluate and preserve local maneuver fidelity.

# Experiment A

Geometry evaluation only.

Do not retrain.

Compute:

- heading
- turn angle
- curvature

Metrics:

- heading error
- turn-angle error
- sharp-turn recall
- sharp-turn precision
- curvature error

Expected outcome:

ADE improves but sharp-turn metrics degrade.

# Experiment B

Hybrid V3 geometry-aware supervision.

Add:

- `heading_loss`
- `turn_loss`

Recommended:

- `lambda_heading = 0.05`
- `lambda_turn = 0.1`

No architecture change.

Input remains:

GAF image

Only output supervision changes.

# Experiment C

Anchor mechanism analysis.

Compute:

`offset = pred_start - true_start`

Analyze:

`corr(offset_norm, oracle_ADE - learned_ADE)`

Purpose:

verify whether learned start acts as drift compensation.
