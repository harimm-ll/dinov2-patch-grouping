from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_csv, write_csv, write_json


@dataclass(frozen=True)
class GroupingConfig:
    layer: int = 8
    window_size: int = 3
    neighbor_topk: int = 30
    local_pass_topk: int = 10
    seed_mutual_topk: int = 3
    singleton_rescue_topk: int = 3
    strict_median_threshold: float = 0.967
    strict_auc_threshold: float = 0.65
    local_primary_quantile: float = 75.0
    global_floor_quantile: float = 10.0
    bridge_threshold_quantile: float = 5.0
    auc_thresholds: tuple[float, ...] = (0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.98)
    max_merged_size: int = 45
    min_cross_pair_count: int = 1
    min_bidirectional_coverage: float = 0.20
    chunk_size: int = 8

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GroupingConfig":
        values = dict(values)
        if "auc_thresholds" in values:
            values["auc_thresholds"] = tuple(values["auc_thresholds"])
        return cls(**values)


class UnionFind:
    def __init__(self, nodes: list[str] | set[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> str:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return root_a
        keep, drop = sorted((root_a, root_b))
        self.parent[drop] = keep
        return keep

    def components(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for node in self.parent:
            result[self.find(node)].append(node)
        return result


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def average_pool_grid(features: np.ndarray, window_size: int) -> np.ndarray:
    if window_size == 1:
        return features.astype(np.float32, copy=False)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    count, height, width, channels = features.shape
    out_h, out_w = height - window_size + 1, width - window_size + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("window_size is larger than the patch grid")
    pooled = np.empty((count, out_h, out_w, channels), dtype=np.float32)
    for row in range(out_h):
        for col in range(out_w):
            pooled[:, row, col] = features[:, row : row + window_size, col : col + window_size].mean(axis=(1, 2))
    return pooled


def component_sizes(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            size = 0
            while queue:
                current_row, current_col = queue.popleft()
                size += 1
                for delta_row, delta_col in offsets:
                    next_row, next_col = current_row + delta_row, current_col + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        queue.append((next_row, next_col))
            sizes.append(size)
    return sorted(sizes, reverse=True)


def spatial_auc(similarity_map: np.ndarray, thresholds: np.ndarray) -> float:
    ratios = []
    for threshold in thresholds:
        sizes = component_sizes(similarity_map >= threshold)
        ratios.append(sum(sizes[:2]) / similarity_map.size)
    return float(np.trapezoid(ratios, thresholds) / (thresholds[-1] - thresholds[0]))


def _topk_neighbors(pooled: np.ndarray, topk: int, chunk_size: int) -> list[list[tuple[int, float]]]:
    count = pooled.shape[0]
    if count < 2:
        raise ValueError("At least two images are required")
    topk = min(topk, count - 1)
    result: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        similarity = np.einsum("awd,bwd->abw", pooled[start:end], pooled, optimize=True)
        medians = np.percentile(similarity, 50, axis=2)
        for local_index, image_index in enumerate(range(start, end)):
            medians[local_index, image_index] = -np.inf
            candidates = np.argpartition(medians[local_index], -topk)[-topk:]
            ordered = candidates[np.argsort(medians[local_index, candidates])[::-1]]
            result[image_index] = [(int(other), float(medians[local_index, other])) for other in ordered]
    return result


def _pair_key(a: Any, b: Any) -> tuple[Any, Any]:
    return (a, b) if a <= b else (b, a)


def _connected_components(nodes: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    union_find = UnionFind(nodes)
    for edge in edges:
        union_find.union(edge["source"], edge["target"])
    groups = [sorted(group) for group in union_find.components().values()]
    return sorted(groups, key=lambda group: (-len(group), group[0]))


def _pair_stats(edges: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(edge["patch_median"]) for edge in edges]
    aucs = [float(edge["spatial_auc"]) for edge in edges]
    scores = [float(edge["pair_score"]) for edge in edges]
    return {
        "pair_count": len(edges),
        "score_max": max(scores) if scores else 0.0,
        "median_max": max(medians) if medians else 0.0,
        "median_p25": percentile(medians, 25),
        "auc_max": max(aucs) if aucs else 0.0,
        "auc_p10": percentile(aucs, 10),
    }


def _coverage(edges: list[dict[str, Any]], ids_a: set[str], ids_b: set[str]) -> float:
    hit_a: set[str] = set()
    hit_b: set[str] = set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source in ids_a:
            hit_a.add(source)
            hit_b.add(target)
        else:
            hit_a.add(target)
            hit_b.add(source)
    return min(len(hit_a) / len(ids_a), len(hit_b) / len(ids_b))


def _cross_edges(edges: list[dict[str, Any]], ids_a: set[str], ids_b: set[str]) -> list[dict[str, Any]]:
    return [
        edge for edge in edges
        if (edge["source"] in ids_a and edge["target"] in ids_b)
        or (edge["source"] in ids_b and edge["target"] in ids_a)
    ]


def group_features(feature_dir: Path, output_dir: Path, config: GroupingConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = read_csv(feature_dir / "image_index.csv")
    if not images or not {"image_id", "image_path"}.issubset(images[0]):
        raise ValueError("image_index.csv must contain image_id and image_path")
    image_ids = [row["image_id"] for row in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_id values must be unique")

    feature_path = feature_dir / f"patch_layer_{config.layer}.npz"
    features = np.load(feature_path, allow_pickle=False)["features"].astype(np.float32)
    if len(features) != len(images):
        raise ValueError("Feature and image-index row counts differ")
    if features.shape[1] != features.shape[2]:
        raise ValueError("A square patch grid is required")

    thresholds = np.asarray(config.auc_thresholds, dtype=np.float64)
    if len(thresholds) < 2 or not np.all(np.diff(thresholds) > 0):
        raise ValueError("auc_thresholds must be strictly increasing")
    pooled_grid = average_pool_grid(features, config.window_size)
    grid_size = pooled_grid.shape[1]
    pooled = l2_normalize(pooled_grid.reshape(len(images), -1, pooled_grid.shape[-1]))
    neighbors = _topk_neighbors(pooled, config.neighbor_topk, config.chunk_size)

    unique_pairs = {
        _pair_key(anchor, other)
        for anchor, neighbor_rows in enumerate(neighbors)
        for other, _ in neighbor_rows
    }
    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for a, b in sorted(unique_pairs):
        similarity_map = np.sum(pooled[a] * pooled[b], axis=1).reshape(grid_size, grid_size)
        median = float(np.percentile(similarity_map, 50))
        auc = spatial_auc(similarity_map, thresholds)
        pair_metrics[(a, b)] = {
            "patch_median": median,
            "spatial_auc": auc,
            "pair_score": 0.75 * median + 0.25 * auc,
        }

    directional: dict[tuple[int, int], dict[str, Any]] = {}
    for anchor, neighbor_rows in enumerate(neighbors):
        for rank, (other, median) in enumerate(neighbor_rows, start=1):
            directional[(anchor, other)] = {"rank": rank, "patch_median": median}

    strict_pairs: list[dict[str, Any]] = []
    for (a, b), metrics in pair_metrics.items():
        rank_ab = directional.get((a, b), {}).get("rank", 10**9)
        rank_ba = directional.get((b, a), {}).get("rank", 10**9)
        strict = metrics["patch_median"] >= config.strict_median_threshold and metrics["spatial_auc"] >= config.strict_auc_threshold
        if strict and rank_ab <= config.seed_mutual_topk and rank_ba <= config.seed_mutual_topk:
            strict_pairs.append({"source": image_ids[a], "target": image_ids[b]})

    seed_components = _connected_components(image_ids, strict_pairs)
    seed_singletons = {group[0] for group in seed_components if len(group) == 1}
    singleton_indices = {index for index, image_id in enumerate(image_ids) if image_id in seed_singletons}

    top1_medians = [neighbor_rows[0][1] for neighbor_rows in neighbors]
    top1_aucs = [pair_metrics[_pair_key(index, neighbor_rows[0][0])]["spatial_auc"] for index, neighbor_rows in enumerate(neighbors)]
    floor_indices = sorted(singleton_indices) or list(range(len(images)))
    primary_floor = percentile([top1_medians[index] for index in floor_indices], config.global_floor_quantile)
    secondary_floor = percentile([top1_aucs[index] for index in floor_indices], config.global_floor_quantile)

    local_thresholds: list[float] = []
    for anchor, neighbor_rows in enumerate(neighbors):
        local_threshold = max(primary_floor, percentile([value for _, value in neighbor_rows], config.local_primary_quantile))
        local_thresholds.append(local_threshold)
        for other, median in neighbor_rows:
            pair = pair_metrics[_pair_key(anchor, other)]
            directional[(anchor, other)]["local_pass"] = int(
                directional[(anchor, other)]["rank"] <= config.local_pass_topk
                and median >= local_threshold
                and pair["spatial_auc"] >= secondary_floor
            )

    all_pair_rows: list[dict[str, Any]] = []
    candidate_edges: list[dict[str, Any]] = []
    for (a, b), metrics in sorted(pair_metrics.items(), key=lambda item: (-item[1]["pair_score"], item[0])):
        direction_ab = directional.get((a, b), {})
        direction_ba = directional.get((b, a), {})
        strict = metrics["patch_median"] >= config.strict_median_threshold and metrics["spatial_auc"] >= config.strict_auc_threshold
        adaptive = bool(direction_ab.get("local_pass", 0) and direction_ba.get("local_pass", 0))
        rescue = bool(
            (a in singleton_indices and direction_ab.get("rank", 10**9) <= config.singleton_rescue_topk and direction_ab.get("local_pass", 0))
            or (b in singleton_indices and direction_ba.get("rank", 10**9) <= config.singleton_rescue_topk and direction_ba.get("local_pass", 0))
        )
        if strict:
            edge_type = "strict_seed"
        elif adaptive:
            edge_type = "adaptive_near_miss"
        elif rescue:
            edge_type = "singleton_rescue"
        else:
            edge_type = "topk_neighbor_only"
        row = {
            "source": image_ids[a],
            "target": image_ids[b],
            "source_path": images[a]["image_path"],
            "target_path": images[b]["image_path"],
            "source_rank_to_target": direction_ab.get("rank", ""),
            "target_rank_to_source": direction_ba.get("rank", ""),
            "source_local_threshold": local_thresholds[a],
            "target_local_threshold": local_thresholds[b],
            "source_seed_singleton": int(a in singleton_indices),
            "target_seed_singleton": int(b in singleton_indices),
            **metrics,
            "strict_pass": int(strict),
            "mutual_local_pass": int(adaptive),
            "singleton_rescue_pass": int(rescue),
            "edge_type": edge_type,
        }
        all_pair_rows.append(row)
        if edge_type != "topk_neighbor_only":
            candidate_edges.append(row)

    seed_edges = [
        edge for edge in candidate_edges
        if edge["edge_type"] == "strict_seed"
        and int(edge["source_rank_to_target"] or 10**9) <= config.seed_mutual_topk
        and int(edge["target_rank_to_source"] or 10**9) <= config.seed_mutual_topk
    ]
    seed_components = _connected_components(image_ids, seed_edges)
    node_to_seed: dict[str, str] = {}
    seed_to_nodes: dict[str, set[str]] = {}
    seed_group_rows: list[dict[str, Any]] = []
    seed_image_rows: list[dict[str, Any]] = []
    for index, component in enumerate(seed_components, start=1):
        group_id = f"seed_group_{index:04d}"
        seed_to_nodes[group_id] = set(component)
        seed_group_rows.append({"seed_group_id": group_id, "group_size": len(component), "image_ids": " | ".join(component)})
        for image_id in component:
            node_to_seed[image_id] = group_id
            image_index = image_ids.index(image_id)
            seed_image_rows.append({**images[image_index], "seed_group_id": group_id, "seed_group_size": len(component)})

    grouped_edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in candidate_edges:
        group_a, group_b = node_to_seed[edge["source"]], node_to_seed[edge["target"]]
        if group_a != group_b:
            grouped_edges[_pair_key(group_a, group_b)].append(edge)

    group_candidates: list[dict[str, Any]] = []
    for (group_a, group_b), edges in grouped_edges.items():
        ids_a, ids_b = seed_to_nodes[group_a], seed_to_nodes[group_b]
        stats = _pair_stats(edges)
        group_candidates.append({
            "group_a": group_a,
            "group_b": group_b,
            "size_a": len(ids_a),
            "size_b": len(ids_b),
            "merged_size": len(ids_a | ids_b),
            "strict_edge_count": sum(edge["edge_type"] == "strict_seed" for edge in edges),
            "adaptive_edge_count": sum(edge["edge_type"] == "adaptive_near_miss" for edge in edges),
            "singleton_rescue_edge_count": sum(edge["edge_type"] == "singleton_rescue" for edge in edges),
            "cross_pair_count": stats["pair_count"],
            "cross_score_max": stats["score_max"],
            "cross_median_max": stats["median_max"],
            "cross_median_p25": stats["median_p25"],
            "cross_auc_max": stats["auc_max"],
            "cross_auc_p10": stats["auc_p10"],
            "cross_bidirectional_coverage": _coverage(edges, ids_a, ids_b),
        })

    bridge_median_threshold = percentile([row["cross_median_p25"] for row in group_candidates], config.bridge_threshold_quantile)
    bridge_auc_threshold = percentile([row["cross_auc_p10"] for row in group_candidates], config.bridge_threshold_quantile)
    ordered_candidates = sorted(
        group_candidates,
        key=lambda row: (
            -row["cross_score_max"],
            -row["cross_median_max"],
            -row["cross_auc_max"],
            -row["strict_edge_count"],
            -row["adaptive_edge_count"],
            row["group_a"],
            row["group_b"],
        ),
    )

    original_groups = set(seed_to_nodes)
    union_find = UnionFind(original_groups)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate_rank, row in enumerate(ordered_candidates, start=1):
        root_a, root_b = union_find.find(row["group_a"]), union_find.find(row["group_b"])
        if root_a == root_b:
            rejected.append({**row, "candidate_rank": candidate_rank, "decision": "rejected", "reason": "already_same_component"})
            continue
        groups_a = [group for group in original_groups if union_find.find(group) == root_a]
        groups_b = [group for group in original_groups if union_find.find(group) == root_b]
        ids_a = set().union(*(seed_to_nodes[group] for group in groups_a))
        ids_b = set().union(*(seed_to_nodes[group] for group in groups_b))
        cross = _cross_edges(candidate_edges, ids_a, ids_b)
        stats = _pair_stats(cross)
        coverage = _coverage(cross, ids_a, ids_b)
        current = {
            **row,
            "candidate_rank": candidate_rank,
            "current_size_a": len(ids_a),
            "current_size_b": len(ids_b),
            "current_merged_size": len(ids_a | ids_b),
            "current_cross_pair_count": stats["pair_count"],
            "current_cross_median_p25": stats["median_p25"],
            "current_cross_auc_p10": stats["auc_p10"],
            "current_cross_bidirectional_coverage": coverage,
        }
        failures: list[str] = []
        if current["current_merged_size"] > config.max_merged_size:
            failures.append("merged_size")
        if stats["pair_count"] < config.min_cross_pair_count:
            failures.append("cross_pair_count")
        if stats["median_p25"] < bridge_median_threshold:
            failures.append("cross_median_p25")
        if stats["auc_p10"] < bridge_auc_threshold:
            failures.append("cross_auc_p10")
        if coverage < config.min_bidirectional_coverage:
            failures.append("bidirectional_coverage")
        if failures:
            rejected.append({**current, "decision": "rejected", "reason": "|".join(failures)})
        else:
            union_find.union(root_a, root_b)
            accepted.append({**current, "decision": "accepted", "reason": ""})

    final_seed_components = union_find.components()
    ordered_final = sorted(
        final_seed_components.values(),
        key=lambda groups: (-sum(len(seed_to_nodes[group]) for group in groups), min(groups)),
    )
    seed_to_final: dict[str, str] = {}
    for index, groups in enumerate(ordered_final, start=1):
        final_id = f"final_group_{index:04d}"
        for group in groups:
            seed_to_final[group] = final_id
    final_rows = [{**row, "final_group_id": seed_to_final[row["seed_group_id"]]} for row in seed_image_rows]
    final_sizes = Counter(row["final_group_id"] for row in final_rows)
    for row in final_rows:
        row["final_group_size"] = final_sizes[row["final_group_id"]]
    final_rows.sort(key=lambda row: (row["final_group_id"], row["image_id"]))
    remaining_singletons = [row for row in final_rows if row["final_group_size"] == 1]

    anchor_rows = [
        {
            **images[index],
            "seed_singleton": int(image_ids[index] in seed_singletons),
            "top1_patch_median": top1_medians[index],
            "top1_spatial_auc": top1_aucs[index],
            "local_primary_threshold": local_thresholds[index],
            "global_primary_floor": primary_floor,
            "global_secondary_floor": secondary_floor,
        }
        for index in range(len(images))
    ]
    write_csv(output_dir / "anchor_thresholds.csv", anchor_rows)
    write_csv(output_dir / "topk_pairs.csv", all_pair_rows)
    write_csv(output_dir / "candidate_edges.csv", candidate_edges)
    write_csv(output_dir / "seed_edges.csv", seed_edges)
    write_csv(output_dir / "seed_groups.csv", seed_group_rows)
    write_csv(output_dir / "seed_assignments.csv", sorted(seed_image_rows, key=lambda row: (row["seed_group_id"], row["image_id"])))
    write_csv(output_dir / "bridge_candidates.csv", ordered_candidates)
    write_csv(output_dir / "accepted_merges.csv", accepted)
    write_csv(output_dir / "rejected_merges.csv", rejected)
    write_csv(output_dir / "final_groups.csv", final_rows)
    write_csv(output_dir / "remaining_singletons.csv", remaining_singletons, fieldnames=list(final_rows[0]))

    summary = {
        "image_count": len(images),
        "config": asdict(config),
        "thresholds": {
            "global_primary_floor": primary_floor,
            "global_secondary_floor": secondary_floor,
            "bridge_cross_median_p25": bridge_median_threshold,
            "bridge_cross_auc_p10": bridge_auc_threshold,
        },
        "pair_type_counts": dict(Counter(row["edge_type"] for row in all_pair_rows)),
        "strict_seed_edge_count": len(seed_edges),
        "seed_group_count": len(seed_components),
        "seed_singleton_count": len(seed_singletons),
        "bridge_candidate_count": len(group_candidates),
        "accepted_merge_count": len(accepted),
        "final_group_count": len(final_sizes),
        "remaining_singleton_count": len(remaining_singletons),
        "largest_group_size": max(final_sizes.values()),
    }
    write_json(output_dir / "summary.json", summary)
    return summary

