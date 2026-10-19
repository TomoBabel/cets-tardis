import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from cets_data_model.models.models import (
    AnnotatedObject,
    CoordinateSystem,
    Axis,
    AxisType,
    AxisUnit,
    Scale,
    SegmentationMask3D,
    PointSet3D,
    Spline3D,
)
from tardis.constants import (
    INSTANCES_CSV_HEADER,
    PREDICTION_LOG,
    PREDICTION_LOG_TARGET_KEY,
    TARGET_FEATURE_TYPES,
    TardisTarget,
)


# The yaml files are named after the annotation id, which already contains the target, the
# tomogram id and the kind of output, and is unique within the region.
def write_semantic_yaml(mask: SegmentationMask3D, output_directory: Path) -> None:
    _write_obj_yaml(mask, output_directory / f"{mask.id}_tardis_to_cets.yaml")


def write_instances_yaml(
    instances: PointSet3D | Spline3D, output_directory: Path
) -> None:
    _write_obj_yaml(instances, output_directory / f"{instances.id}_tardis_to_cets.yaml")


def _write_obj_yaml(
    cets_obj: SegmentationMask3D | PointSet3D | Spline3D, yaml_file: Path
) -> None:
    metadata_dict = cets_obj.model_dump(mode="json")
    with open(yaml_file, "w") as f:
        yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
    print(f"yaml file successfully written! -> {yaml_file}")


def gen_coordinate_systems(
    name: str, ndim: int = 2
) -> Tuple[CoordinateSystem, CoordinateSystem]:
    """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

    Names follow the CETS proposal convention ``{name}_array`` / ``{name}_physical``. The array
    system is pixel/array coords (unit ``pixel``); the physical system is in Å. ``ndim`` is 2 for
    2-D images (x, y) and 3 for volumes (x, y, z).
    """
    axes = ("x", "y", "z")[:ndim]
    array_cs = CoordinateSystem(
        name=f"{name}_array",
        axes=[
            Axis(name=a, axis_type=AxisType.array, axis_unit=AxisUnit.pixel)
            for a in axes
        ],
    )
    physical_cs = CoordinateSystem(
        name=f"{name}_physical",
        axes=[
            Axis(name=a, axis_type=AxisType.space, axis_unit=AxisUnit.angstrom)
            for a in axes
        ],
    )
    return array_cs, physical_cs


def gen_array_to_physical(
    pixel_size: float, array_cs_name: str, physical_cs_name: str, ndim: int = 2
) -> Scale:
    """The single canonical ``array_to_physical`` transformation for an image: a Scale mapping
    pixel/array coordinates to physical (Å) coordinates by the (isotropic) pixel/voxel size.
    The spec requires exactly one such transformation per image."""
    return Scale(
        scale=[pixel_size] * ndim,
        name="array_to_physical",
        input=array_cs_name,
        output=physical_cs_name,
    )


def validate_file(filename: os.PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p


def gen_annotated_object(target: TardisTarget) -> AnnotatedObject:
    """The CETS description of the object segmented by Tardis for the given target."""
    return AnnotatedObject(feature_type=TARGET_FEATURE_TYPES[target])


def parse_prediction_log(log_file: os.PathLike) -> TardisTarget:
    """Reads the segmentation target from the prediction_log.txt that Tardis writes next to
    its predictions (line ``Predict: <target>``)."""
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(PREDICTION_LOG_TARGET_KEY):
                target = line[len(PREDICTION_LOG_TARGET_KEY) :].strip()
                try:
                    return TardisTarget(target)
                except ValueError:
                    raise ValueError(
                        f"Unsupported Tardis segmentation target '{target}'. Only the 3D "
                        f"targets are supported: {[t.value for t in TardisTarget]}"
                    ) from None
    raise ValueError(
        f"No line starting with '{PREDICTION_LOG_TARGET_KEY}' was found in {log_file}"
    )


def resolve_target(prediction_file: Path, target: TardisTarget | None) -> TardisTarget:
    """Returns the given target or, if None, the one read from the prediction_log.txt located
    in the same directory as the prediction file."""
    if target is not None:
        return target
    log_file = prediction_file.parent / PREDICTION_LOG
    if not log_file.is_file():
        raise FileNotFoundError(
            f"The segmentation target was not provided and no {PREDICTION_LOG} was found "
            f"next to {prediction_file} to read it from."
        )
    return parse_prediction_log(log_file)


def read_instances_csv(
    csv_file: os.PathLike,
) -> Tuple[List[int], List[List[float]]]:
    """Reads a Tardis instances csv, keeping the row order (Tardis writes the points grouped by
    instance and, for filaments, in their order along the spline).

    :returns: the instance id of each point and the point coordinates [[x, y, z], ...] in Å,
    referred to the bottom left corner of the tomogram.
    """
    with open(csv_file, "r") as f:
        header = [col.strip() for col in f.readline().split(",")]
        rows = [line for line in f if line.strip()]
    if header != INSTANCES_CSV_HEADER:
        raise ValueError(
            f"Unexpected header in {csv_file}: {header}. Expected: {INSTANCES_CSV_HEADER}"
        )
    if not rows:
        # Tardis writes only the header when no instances are found
        return [], []
    data = np.loadtxt(rows, delimiter=",", ndmin=2)
    instance_ids = [int(row[0]) for row in data]
    coords = [[float(row[1]), float(row[2]), float(row[3])] for row in data]
    return instance_ids, coords
