from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clustering import GroupingConfig, group_features
from .features import extract_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patch-grouping")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Extract same-position DINOv2 patch features")
    extract.add_argument("--images", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--image-root", type=Path)
    extract.add_argument("--model", default="dinov2_vits14")
    extract.add_argument("--repo", default="facebookresearch/dinov2")
    extract.add_argument("--layer", type=int, default=8)
    extract.add_argument("--image-size", type=int, default=518)
    extract.add_argument("--patch-size", type=int, default=14)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    extract.add_argument("--torch-home", type=Path)

    cluster = commands.add_parser("cluster", help="Build pair edges and image groups")
    cluster.add_argument("--feature-dir", type=Path, required=True)
    cluster.add_argument("--output-dir", type=Path, required=True)
    cluster.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "extract":
        result = extract_features(
            image_csv=args.images,
            output_dir=args.output_dir,
            image_root=args.image_root,
            model_name=args.model,
            repo=args.repo,
            layer=args.layer,
            image_size=args.image_size,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device=args.device,
            torch_home=args.torch_home,
        )
    else:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        result = group_features(args.feature_dir, args.output_dir, GroupingConfig.from_dict(values))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

