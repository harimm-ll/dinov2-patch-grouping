# Algorithm specification

## 1. Patch similarity

For image `i`, let `x_i(p)` be its layer-8 DINOv2 token at patch position `p`.
After `3 x 3` average pooling and L2 normalization, the same-position cosine
map for images `i` and `j` is

```math
c_{ij}(p)=\left\langle \hat{x}_i(p),\hat{x}_j(p)\right\rangle .
```

The primary pair metric is the median over the pooled patch grid:

```math
M_{ij}=Q_{50}\left(\{c_{ij}(p):p\in\Omega\}\right).
```

For a threshold `t`, binarize the similarity map and find its
eight-connected components. If `C_1(t)` and `C_2(t)` are the two largest
components, define

```math
r_{ij}(t)=\frac{|C_1(t)|+|C_2(t)|}{|\Omega|}.
```

With threshold grid `T=(0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.98)`,
the spatial metric is the normalized trapezoidal AUC

```math
A_{ij}=\frac{1}{t_{\max}-t_{\min}}
\int_{t_{\min}}^{t_{\max}}r_{ij}(t)\,dt.
```

Every image keeps its top 30 neighbors ranked by `M_ij`. Pair metrics are
computed once for the union of these directed neighbor lists.

## 2. Strict seed groups and seed singletons

A pair passes the strict metric rule when

```math
M_{ij}\ge 0.967 \quad\text{and}\quad A_{ij}\ge 0.65.
```

It becomes a seed edge only if each image ranks the other within its top 3.
Connected components of these seed edges form the initial seed groups. An image
in a size-one seed component is a **seed singleton**. This set is computed
before adaptive and singleton-recovery proposals are assigned.

## 3. Adaptive thresholds

Let `S_0` be the seed-singleton set and `n_1(i)` the top-1 neighbor of image
`i`. The two global floors are

```math
F_M=Q_{10}\left(\{M_{i,n_1(i)}:i\in S_0\}\right),
\qquad
F_A=Q_{10}\left(\{A_{i,n_1(i)}:i\in S_0\}\right).
```

If `S_0` is empty, all images are used for the two quantiles. For image `i`,
the adaptive primary threshold is

```math
\tau_i=\max\left(F_M,
Q_{75}\left(\{M_{ij}:j\in\mathcal{N}_{30}(i)\}\right)\right).
```

The directed relation `i -> j` passes locally when

```math
\operatorname{rank}_i(j)\le 10,
\qquad M_{ij}\ge\tau_i,
\qquad A_{ij}\ge F_A.
```

## 4. Four pair types

Each top-30-union pair receives exactly one type in this priority order:

1. **Strict seed edge:** `M_ij >= 0.967` and `A_ij >= 0.65`.
2. **Adaptive near-miss edge:** the pair is not strict and both `i -> j` and
   `j -> i` pass locally.
3. **Singleton rescue proposal:** the pair is neither strict nor adaptive, and
   at least one endpoint is a seed singleton that ranks the other endpoint in
   its top 3 and passes the directed local rule.
4. **Top-k neighbor only:** every remaining pair in the top-30 union. These
   pairs are retained for audit and do not propose a merge.

The strict edges used to form seed groups are the subset of type 1 that also
satisfy mutual top-3 ranking. Other type-1 edges can provide bridge evidence.

## 5. Candidate bridges

For every pair of distinct seed groups connected by at least one type 1, 2, or
3 edge, compute statistics over the cross-group candidate edges:

- `cross_median_p25`: 25th percentile of `M_ij`;
- `cross_auc_p10`: 10th percentile of `A_ij`;
- `cross_bidirectional_coverage`: the smaller of the fractions of members in
  the two groups incident to at least one cross edge;
- `cross_score_max`: maximum of `0.75 M_ij + 0.25 A_ij`.

From the initial candidate-group table, compute the fixed bridge thresholds

```math
\theta_M=Q_5\left(\{\operatorname{P25}_M(G,H)\}\right),
\qquad
\theta_A=Q_5\left(\{\operatorname{P10}_A(G,H)\}\right).
```

Candidates are ordered lexicographically in descending order by:

1. maximum pair score;
2. maximum patch median;
3. maximum spatial AUC;
4. number of strict edges;
5. number of adaptive edges.

Group identifiers provide the final deterministic tie break.

## 6. Merge acceptance and update

At a candidate's turn, first find the two current components containing its
original seed groups. Recompute all cross-edge statistics using the members of
these current components. Accept the merge only when

```math
|G\cup H|\le45,
```

```math
N_{\mathrm{cross}}\ge1,
\qquad
\operatorname{P25}_M(G,H)\ge\theta_M,
\qquad
\operatorname{P10}_A(G,H)\ge\theta_A,
```

and

```math
\min\left(
\frac{|V_G^{\mathrm{cross}}|}{|G|},
\frac{|V_H^{\mathrm{cross}}|}{|H|}
\right)\ge0.20.
```

After an accepted merge, the union-find state is updated immediately. The next
candidate therefore uses the newly enlarged components and freshly recomputed
cross statistics. Pair metrics, candidate order, `F_M`, `F_A`, `theta_M`, and
`theta_A` remain fixed.

Singleton-recovery proposals are created after strict seed formation and are
evaluated in this same ordered merge pass. After every candidate has been
processed, final components are written to `final_groups.csv`. The
`remaining_singletons.csv` table is then obtained by selecting final groups of
size one.

