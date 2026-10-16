## Algorithm Overview

The pseudo-group reconstruction procedure follows the method described in the manuscript.

### 1. Patch-level similarity

For each CT image, patch-level representations are extracted from layer 8 of DINOv2 ViT-S/14. A \(3\times3\) average pooling operation is applied to the patch features, followed by L2 normalization.

For an image pair \((i,j)\), the same-position cosine similarities are computed over the pooled patch grid. The median patch similarity is used as the primary similarity measure:

```math
M_{ij}
=
Q_{50}
\left(
\{c_{ij}(p):p\in\Omega\}
\right).
```

To additionally characterize the spatial consistency of highly similar regions, the similarity map is thresholded over

```text
{0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.98}
```

and eight-connected components are identified at each threshold. The normalized trapezoidal area under the resulting spatial-coverage curve is used as the spatial consistency score \(A_{ij}\).

For computational efficiency, pairwise evaluation is restricted to the union of the Top-30 similarity neighborhoods.

### 2. Strict pseudo-group initialization

A pair satisfies the strict similarity criterion when

```math
M_{ij}\ge0.967,
\qquad
A_{ij}\ge0.65.
```

Images are connected as strict seed neighbors when this criterion is satisfied and the two images occur within each other's Top-3 neighborhoods.

Connected components of these mutual strict relations form the initial pseudo-groups.

### 3. Adaptive candidate recovery

Images that are not sufficiently connected by the strict criterion are further considered using adaptive similarity thresholds derived from their local neighborhoods.

Candidate relationships are restricted to the Top-10 neighborhood and must satisfy both an image-specific similarity requirement and the spatial-consistency requirement.

A separate Top-3 recovery step is used for singleton images to identify additional plausible connections.

These adaptive and singleton-recovery relationships do not directly determine the final grouping; instead, they provide additional evidence for subsequent group-level merging.

### 4. Group-level merging

Candidate relationships between different initial pseudo-groups are evaluated using cross-group similarity and spatial-consistency statistics.

The merge procedure considers:

- cross-group patch similarity;
- spatial consistency;
- cross-group connectivity coverage;
- evidence from strict and adaptive relations; and
- a maximum pseudo-group size constraint.

Candidate group pairs are processed in a deterministic order. After each accepted merge, the current group composition is updated before evaluating subsequent candidates.

The resulting connected components define the final reconstructed pseudo-groups.

### 5. Output

The reconstruction pipeline produces the final pseudo-group assignment for every image.

The corresponding assignments used in the manuscript are provided under:

```text
results/pseudo_group_assignments/
```

Detailed parameter settings are reported in the manuscript and supplementary material.
