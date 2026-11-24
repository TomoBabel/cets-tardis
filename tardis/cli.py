"""Command line of the Tardis <-> CETS converters (``cets-tardis``).

Examples::

    cets-tardis semantic to-cets --semantic Predictions/TS_01_semantic.mrc \\
        --tomo-id TS_01 -o TS_01_semantic_cets.yaml
    cets-tardis semantic from-cets -i TS_01_semantic_cets.yaml --output-dir out
    cets-tardis instances to-cets --instances Predictions/TS_01_instances.csv \\
        --tomo-id TS_01 --semantic-annotation-id tardis_membrane_semantic_TS_01 \\
        -o TS_01_instances_cets.yaml
    cets-tardis instances from-cets -i TS_01_instances_cets.yaml --output-dir out

``to-cets`` writes a CETS yaml file with the annotation under ``segmentation`` (the semantic
SegmentationMask3D) or ``coordinates`` (the instances: a PointSet3D for membranes, a
Spline3D for filaments), which is what ``from-cets`` reads. The target, when not given, is
read from the prediction_log.txt that Tardis writes next to its predictions.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import PointSet3D, SegmentationMask3D, Spline3D
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    check_outputs,
    prepare_outputs,
    run_cli,
)
from cets_data_model.utils.yaml_io import load_cets_object, write_cets_yaml
from tardis.constants import (
    CSV_EXT,
    INSTANCES_FILTER_SUFFIX,
    INSTANCES_SUFFIX,
    TardisTarget,
)
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation

SEGMENTATION_KEY = "segmentation"
COORDINATES_KEY = "coordinates"


def _target(value: str) -> TardisTarget:
    """Tardis target from its name or its value (case insensitive)."""
    for target in TardisTarget:
        if value.lower() in (target.name.lower(), target.value.lower()):
            return target
    raise argparse.ArgumentTypeError(
        f"invalid target '{value}' (choose from "
        f"{', '.join(t.value for t in TardisTarget)})"
    )


def _tomo_name(args: argparse.Namespace, source_tomogram_id: Optional[str]) -> str:
    """--tomo-name, or the id of the tomogram the annotation was obtained from."""
    tomo_name = args.tomo_name or source_tomogram_id
    if not tomo_name:
        raise ValueError(
            f"{args.input} has no source tomogram id to name the files with: provide "
            f"--tomo-name."
        )
    return str(tomo_name)


def _add_to_cets_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tomo-id",
        required=True,
        help="Id of the CETS tomogram it was obtained from (source_tomogram_id; it also "
        "names the coordinate systems).",
    )
    parser.add_argument(
        "--target",
        type=_target,
        metavar="{" + ",".join(t.value for t in TardisTarget) + "}",
        help="Segmented target. Default: the one of the prediction_log.txt next to the "
        "input file.",
    )
    parser.add_argument(
        "--annotation-id",
        help="Id of the annotation, unique within the region (needed when several Tardis "
        "runs on the same tomogram are converted). Default: generated from the target "
        "and the tomogram id.",
    )


def _add_from_cets_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the file in (natively, the Predictions directory).",
    )
    parser.add_argument(
        "--tomo-name",
        help="Name of the tomogram in the Tardis file names (the input tomogram file name "
        "without extension). Default: the source tomogram id of the annotation.",
    )


# ------------------------------------------------------------------------- semantic
def semantic_to_cets(args: argparse.Namespace) -> None:
    mask = TardisSemanticSegmentation().tardis_to_cets(
        args.semantic,
        args.tomo_id,
        pixel_size=args.pixel_size,
        target=args.target,
        annotation_id=args.annotation_id,
    )
    out = write_cets_yaml(args.output, {SEGMENTATION_KEY: mask})
    print(f"CETS semantic segmentation written -> {out}")


def semantic_from_cets(args: argparse.Namespace) -> None:
    mask = load_cets_object(args.input, SEGMENTATION_KEY, SegmentationMask3D)
    out = TardisSemanticSegmentation().cets_to_tardis(
        mask,
        args.output_dir,
        _tomo_name(args, mask.source_tomogram_id),
        copy=args.copy,
    )
    print(f"Tardis semantic mask written -> {out}")


def _add_semantic(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types, "semantic", "Semantic segmentation (<tomo>_semantic.mrc)."
    )

    to_cets = directions.add_parser(
        TO_CETS, help="Tardis semantic mask -> CETS yaml (segmentation)."
    )
    to_cets.add_argument(
        "--semantic",
        required=True,
        type=Path,
        help="Semantic mask (<tomo>_semantic.mrc).",
    )
    _add_to_cets_common(to_cets)
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px). Default: the one of the mrc header.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=semantic_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> Tardis semantic mask (<tomo>_semantic.mrc)."
    )
    add_input_yaml(from_cets, "CETS yaml with the semantic segmentation.")
    _add_from_cets_common(from_cets)
    from_cets.add_argument(
        "--copy",
        action="store_true",
        help="Copy the mask instead of linking it.",
    )
    from_cets.set_defaults(func=semantic_from_cets)


# ------------------------------------------------------------------------- instances
def instances_to_cets(args: argparse.Namespace) -> None:
    instances = TardisInstanceSegmentation().tardis_to_cets(
        args.instances,
        args.tomo_id,
        pixel_size=args.pixel_size,
        target=args.target,
        semantic_annotation_id=args.semantic_annotation_id,
        annotation_id=args.annotation_id,
    )
    if instances is None:
        raise ValueError(f"Tardis did not find any instance in {args.instances}.")
    out = write_cets_yaml(args.output, {COORDINATES_KEY: instances})
    print(f"CETS instances written -> {out}")


def _load_instances(yaml_file: Path) -> PointSet3D | Spline3D:
    """The instances annotation of a CETS yaml file: a PointSet3D or a Spline3D (told
    apart by their annotation_type)."""
    errors = []
    for model_cls in (PointSet3D, Spline3D):
        try:
            return load_cets_object(yaml_file, COORDINATES_KEY, model_cls)
        except ValueError as e:
            errors.append(str(e))
    raise ValueError(
        f"{yaml_file} does not contain a PointSet3D nor a Spline3D: {' '.join(errors)}"
    )


def instances_from_cets(args: argparse.Namespace) -> None:
    instances = _load_instances(args.input)
    tomo_name = _tomo_name(args, instances.source_tomogram_id)
    suffix = INSTANCES_FILTER_SUFFIX if args.filtered else INSTANCES_SUFFIX
    outputs = prepare_outputs(args.output_dir / f"{tomo_name}{suffix}{CSV_EXT}")
    TardisInstanceSegmentation().cets_to_tardis(
        instances,
        args.output_dir,
        tomo_name,
        pixel_size=args.pixel_size,
        filtered=args.filtered,
    )
    check_outputs(outputs)


def _add_instances(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "instances",
        "Instance segmentation (<tomo>_instances[_filter].csv): membranes (point "
        "clouds) or filaments (splines).",
    )

    to_cets = directions.add_parser(
        TO_CETS,
        help="Tardis instances csv -> CETS yaml (coordinates: PointSet3D for "
        "membranes, Spline3D for filaments).",
    )
    to_cets.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Instances csv (<tomo>_instances.csv or <tomo>_instances_filter.csv).",
    )
    _add_to_cets_common(to_cets)
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px) used by Tardis. Default: the header of the semantic "
        "mask next to the csv, else of the input tomogram.",
    )
    to_cets.add_argument(
        "--semantic-annotation-id",
        help="Id of the semantic mask annotation of the same run, the instances were "
        "computed from (derived_from_annotation_id).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=instances_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS,
        help="CETS yaml -> Tardis instances csv (<tomo>_instances[_filter].csv).",
    )
    add_input_yaml(
        from_cets, "CETS yaml with the instances (a PointSet3D or a Spline3D)."
    )
    _add_from_cets_common(from_cets)
    from_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px) to write the coordinates in Å. Default: the one of "
        "the annotation.",
    )
    from_cets.add_argument(
        "--filtered",
        action="store_true",
        help="The instances are the length-filtered filaments: write "
        "<tomo>_instances_filter.csv.",
    )
    from_cets.set_defaults(func=instances_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-tardis",
        description="Converts Tardis files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{semantic,instances}", required=True
    )
    _add_semantic(data_types)
    _add_instances(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
