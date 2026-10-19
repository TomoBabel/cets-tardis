import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cets_data_model.utils.image_utils import get_mrc_info
from tardis.constants import (
    CSV_EXT,
    FILAMENT_TARGETS,
    INSTANCES_FILTER_SUFFIX,
    INSTANCES_SUFFIX,
    MRC_EXT,
    PREDICTION_LOG,
    SEMANTIC_SUFFIX,
    TardisTarget,
)
from tardis.utils.utils import parse_prediction_log

# Extensions of the input tomograms read by Tardis that are mrc files
TOMOGRAM_EXTS = (MRC_EXT, ".rec")


@dataclass(frozen=True)
class TardisPrediction:
    """The Tardis outputs of a tomogram, written in a Predictions directory."""

    tomo_id: str
    predictions_dir: Path
    target: TardisTarget
    semantic_mrc: Optional[Path] = None
    instances_csv: Optional[Path] = None
    instances_filter_csv: Optional[Path] = None

    @property
    def preferred_instances_csv(self) -> Optional[Path]:
        """For filaments, the length-filtered and connected splines if they were written;
        otherwise, all the predicted instances."""
        if self.target in FILAMENT_TARGETS and self.instances_filter_csv:
            return self.instances_filter_csv
        return self.instances_csv

    @property
    def input_tomogram(self) -> Optional[Path]:
        """The tomogram segmented by Tardis, which writes the Predictions directory next to
        its input file(s)."""
        return find_input_tomogram(self.predictions_dir, self.tomo_id)

    def get_pixel_size(self) -> Optional[float]:
        """Voxel size (Å) used by Tardis (see ``find_pixel_size``). None if unknown."""
        return find_pixel_size(self.predictions_dir, self.tomo_id)


def find_input_tomogram(predictions_dir: os.PathLike, tomo_id: str) -> Optional[Path]:
    """The tomogram segmented by Tardis, which writes the Predictions directory next to its
    input file(s). None if it is not there."""
    for ext in TOMOGRAM_EXTS:
        tomogram = Path(predictions_dir).parent / f"{tomo_id}{ext}"
        if tomogram.is_file():
            return tomogram
    return None


def find_pixel_size(predictions_dir: os.PathLike, tomo_id: str) -> Optional[float]:
    """Voxel size (Å) used by Tardis for a tomogram, which the instances csv does not record:
    the one written in the header of its semantic mask (--correct_px), or else the one of the
    input tomogram header. None if none of them is found or has it."""
    semantic_mrc = Path(predictions_dir) / f"{tomo_id}{SEMANTIC_SUFFIX}{MRC_EXT}"
    for mrc_file in (
        semantic_mrc if semantic_mrc.is_file() else None,
        find_input_tomogram(predictions_dir, tomo_id),
    ):
        if mrc_file is not None:
            pixel_size = get_mrc_info(mrc_file).apix_x
            if pixel_size:
                return pixel_size
    return None


def find_tardis_predictions(root: os.PathLike) -> List[TardisPrediction]:
    """Recursively finds the Tardis outputs under ``root``: every directory containing a
    prediction_log.txt (written by Tardis in its Predictions output directory) is read, and its
    <tomo>_semantic.mrc, <tomo>_instances.csv and <tomo>_instances_filter.csv files are
    grouped by tomogram. Any other file is ignored.

    :returns: the predictions, sorted by directory and tomogram id.
    """
    predictions = []
    for log_file in sorted(Path(root).rglob(PREDICTION_LOG)):
        predictions_dir = log_file.parent
        target = parse_prediction_log(log_file)
        files: Dict[str, Dict[str, Path]] = {}
        for path in sorted(predictions_dir.iterdir()):
            for suffix, ext, key in (
                (SEMANTIC_SUFFIX, MRC_EXT, "semantic_mrc"),
                (INSTANCES_FILTER_SUFFIX, CSV_EXT, "instances_filter_csv"),
                (INSTANCES_SUFFIX, CSV_EXT, "instances_csv"),
            ):
                end = f"{suffix}{ext}"
                if path.is_file() and path.name.endswith(end):
                    files.setdefault(path.name[: -len(end)], {})[key] = path
                    break
        for tomo_id, tomo_files in sorted(files.items()):
            predictions.append(
                TardisPrediction(
                    tomo_id=tomo_id,
                    predictions_dir=predictions_dir,
                    target=target,
                    semantic_mrc=tomo_files.get("semantic_mrc"),
                    instances_csv=tomo_files.get("instances_csv"),
                    instances_filter_csv=tomo_files.get("instances_filter_csv"),
                )
            )
    return predictions
