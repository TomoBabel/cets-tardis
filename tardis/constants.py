from enum import Enum

from cets_data_model.models.models import SegmentedFeatureType

MRC_EXT = ".mrc"
CSV_EXT = ".csv"

# Tardis output files (written in <out_dir>/Predictions/)
SEMANTIC_SUFFIX = "_semantic"
INSTANCES_SUFFIX = "_instances"
# For filaments, Tardis also writes <tomo>_instances_filter.csv: the splines filtered by
# length (--filter_by_length) and connected (--connect_splines / --connect_cylinder).
INSTANCES_FILTER_SUFFIX = "_instances_filter"
PREDICTION_LOG = "prediction_log.txt"
PREDICTION_LOG_TARGET_KEY = "Predict:"

# Header of the Tardis instances csv: instance id + coordinates in Å, referred to the
# bottom left corner of the tomogram (voxel index * pixel size).
INSTANCES_CSV_HEADER = ["IDs", "X [A]", "Y [A]", "Z [A]"]


class TardisTarget(str, Enum):
    """3D segmentation targets of Tardis, as written in its prediction_log.txt."""

    membrane = "Membrane"
    microtubule = "Microtubule"
    actin = "Actin"
    general_filament = "General_filament"


# CETS SegmentedFeatureType of the object segmented for each target
TARGET_FEATURE_TYPES = {
    TardisTarget.membrane: SegmentedFeatureType.membrane,
    TardisTarget.microtubule: SegmentedFeatureType.microtubule,
    TardisTarget.actin: SegmentedFeatureType.actin_filament,
    TardisTarget.general_filament: SegmentedFeatureType.filament,
}

# Value of the target voxels in the Tardis semantic masks (the background is 0)
SEMANTIC_LABEL_VALUE = 1

# Targets whose instances are ordered points along a filament (splines). The rest of the
# targets (membranes) produce unordered surface point clouds.
FILAMENT_TARGETS = (
    TardisTarget.microtubule,
    TardisTarget.actin,
    TardisTarget.general_filament,
)
