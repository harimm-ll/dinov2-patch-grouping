# lung-ct-pseudo-group-reconstruction

This repository groups visually related images from same-position DINOv2 patch
features. It contains the complete path from an image manifest to pairwise
similarities, seed groups, adaptive bridge proposals, and final groups.

## Method at a glance

1. Resize each image to `518 x 518`, apply ImageNet normalization, and extract
   the layer-8 patch tokens from DINOv2 ViT-S/14. This produces a `37 x 37`
   patch grid.
2. Average each `3 x 3` patch window and L2-normalize every pooled token.
3. Compare tokens at the same spatial position. Rank the top 30 neighbors of
   every image by the median patch cosine similarity.
4. Measure spatial support with the normalized AUC of the two largest
   eight-connected components over thresholds
   `{0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.98}`.
5. Build strict seed groups, compute adaptive thresholds, assign the four pair
   types, and greedily merge seed groups under group-level cohesion rules.

The exact equations, trigger conditions, bridge order, and update rules are in
[ALGORITHM.md](ALGORITHM.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first feature-extraction run downloads DINOv2 through `torch.hub`. A local
Torch Hub cache can be supplied with `--torch-home`.

## Input

Create a CSV with two columns:

```csv
image_id,image_path
image_0001,images/image_0001.png
image_0002,images/image_0002.png
```

`image_id` must be unique. Relative paths are resolved against the directory
containing the CSV unless `--image-root` is supplied.

## Run

Extract layer-8 patch features:

```bash
patch-grouping extract \
  --images examples/images.csv \
  --image-root /path/to/dataset \
  --output-dir features \
  --layer 8 \
  --image-size 518 \
  --device auto
```

Build groups with the experiment configuration:

```bash
patch-grouping cluster \
  --feature-dir features \
  --output-dir outputs/iq_oth \
  --config configs/iq_oth.json
```

The same commands can be run without installing the package:

```bash
PYTHONPATH=src python -m patch_grouping.cli --help
```

## Main outputs

| File | Contents |
|---|---|
| `anchor_thresholds.csv` | Per-image top-1 statistics and adaptive threshold |
| `topk_pairs.csv` | All unique pairs appearing in at least one top-30 list |
| `candidate_edges.csv` | Strict, adaptive, and singleton-recovery proposals |
| `seed_groups.csv` | Connected components of mutual top-3 strict edges |
| `bridge_candidates.csv` | Ranked group-pair candidates before greedy merging |
| `accepted_merges.csv` | Accepted merges with metrics recomputed at decision time |
| `rejected_merges.csv` | Rejected merges and their failed conditions |
| `final_groups.csv` | Final image-to-group assignment |
| `remaining_singletons.csv` | Rows whose final group has size one |
| `summary.json` | Parameters, thresholds, and group counts |

`remaining_singletons.csv` is generated at the end of the run from
`final_groups.csv`; it is not an input table. The singleton set used to create
recovery proposals is generated earlier from size-one strict seed components.

## Reproducibility

All thresholds and ranking parameters are stored in a JSON configuration. Ties
are resolved by stable image and group identifiers. Global floors and bridge
thresholds are computed once per run and remain fixed during greedy merging.

The reference IQ-OTH run contains 1,097 images and produces 208 strict seed
groups, yielding 120 final groups(including 8 singletons). Its computed
thresholds and counts are stored in
[`results/iq_oth_summary.json`](results/iq_oth_summary.json).

Run the tests with:

```bash
pip install -r requirements-dev.txt
pytest -q
```

