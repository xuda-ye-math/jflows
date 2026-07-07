"""Internal machinery for jflows flows.

Organised after zuko's core layout (transforms, MLPs, ODE solver), with two
deliberate divergences from it:

    1. there is no `context` / conditional-on-c plumbing anywhere;
    2. MonotonicRQSTransform / CircularShiftTransform accept a per-coord
       `bound` array so spline knots can span [-bound_i, bound_i] without
       an affine scaling sandwich.

The public flow API lives in jflows/flow.py and re-exports
`ComposedTransform` for downstream code.
"""
