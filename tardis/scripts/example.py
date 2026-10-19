"""Converts the Tardis outputs found in a directory into CETS metadata."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation
from tardis.utils.discovery import TardisPrediction, find_tardis_predictions
from tardis.utils.utils import write_instances_yaml, write_semantic_yaml

### TARDIS TO CETS #################################################################
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
# Any directory containing Tardis outputs (searched recursively): each Predictions directory
# written by Tardis (identified by its prediction_log.txt), with the <tomo>_semantic.mrc,
# <tomo>_instances.csv and, for filaments, <tomo>_instances_filter.csv files.
tardis_dir = Path("/home/jjimenez/ScipionUserData/projects/czii_tardis_actin")


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return number


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--sampling-rate",
    type=positive_float,
    default=None,
    help="Sampling rate (Å/px) of the segmented tomograms. If given, it is used for all the "
    "outputs; otherwise, the one used by Tardis (--correct_px) is read from the header of the semantic mask or, else, of the input tomogram.",
)
sampling_rate: Optional[float] = parser.parse_args().sampling_rate

predictions = find_tardis_predictions(tardis_dir)
if not predictions:
    raise SystemExit(f"No Tardis outputs were found in {tardis_dir}")


def get_annotation_ids(
    predictions: List[TardisPrediction],
) -> Dict[Tuple[Path, str], str]:
    """The converters' default ids depend on the target, the tomogram and the kind of output,
    so the outputs of several runs on the same tomogram would collide. In that case, the
    Predictions directory (relative to tardis_dir) is appended to keep the ids unique within
    the region. Returns {(predictions dir, kind): id}, kind being semantic or instances."""
    default_ids = {}
    for p in predictions:
        if p.semantic_mrc:
            default_ids[(p.predictions_dir, "semantic")] = (
                TardisSemanticSegmentation.get_annotation_id(p.tomo_id, p.target)
            )
        csv_file = p.preferred_instances_csv
        if csv_file:
            default_ids[(p.predictions_dir, "instances")] = (
                TardisInstanceSegmentation.get_annotation_id(
                    p.tomo_id,
                    p.target,
                    TardisInstanceSegmentation.is_filtered(csv_file),
                )
            )
    counts = Counter(default_ids.values())
    annotation_ids = {}
    for (predictions_dir, kind), default_id in default_ids.items():
        if counts[default_id] > 1:
            rel_dir = predictions_dir.relative_to(tardis_dir).as_posix()
            annotation_ids[(predictions_dir, kind)] = (
                f"{default_id}__{rel_dir.replace('/', '_')}"
            )
        else:
            annotation_ids[(predictions_dir, kind)] = default_id
    return annotation_ids


annotation_ids = get_annotation_ids(predictions)
semantic_converter = TardisSemanticSegmentation()
instances_converter = TardisInstanceSegmentation()
for prediction in predictions:
    tomo_id = prediction.tomo_id
    pixel_size = sampling_rate or prediction.get_pixel_size()

    # Semantic segmentation (-out mrc_*) -> SegmentationMask3D (Region.annotations)
    semantic_id = None
    if prediction.semantic_mrc:
        semantic_mask = semantic_converter.tardis_to_cets(
            semantic_mrc=prediction.semantic_mrc,
            tomo_id=tomo_id,
            pixel_size=pixel_size,
            target=prediction.target,
            annotation_id=annotation_ids[(prediction.predictions_dir, "semantic")],
        )
        write_semantic_yaml(semantic_mask, scratch_dir)
        semantic_id = semantic_mask.id

    # Instance segmentation (-out *_csv) -> a single Spline3D (filaments) or PointSet3D
    # (membranes) with the instance id of each point (Region.annotations), derived from the
    # semantic mask of the same run.
    instances_csv = prediction.preferred_instances_csv
    if instances_csv:
        if pixel_size is None:
            print(
                f"WARNING: skipping {instances_csv}: the voxel size used by Tardis is not "
                f"recorded in it, nor in a semantic mask or input tomogram; use --sampling-rate."
            )
            continue
        instances = instances_converter.tardis_to_cets(
            instances_csv=instances_csv,
            tomo_id=tomo_id,
            pixel_size=pixel_size,
            target=prediction.target,
            semantic_annotation_id=semantic_id,
            annotation_id=annotation_ids[(prediction.predictions_dir, "instances")],
        )
        if instances is not None:
            write_instances_yaml(instances, scratch_dir)
