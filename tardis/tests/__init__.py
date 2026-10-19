from pathlib import Path

import mrcfile
import numpy as np

from tardis.constants import INSTANCES_CSV_HEADER, PREDICTION_LOG

# Excerpts of real Tardis 0.3.10 outputs (prediction logs verbatim, first rows of two instances
# of each csv), from the scipion-em-tardis test projects TestTardisMembraneSeg and
# TestTardisActinSeg.
TEST_DATA_DIR = Path(__file__).parent / "test_data"
REAL_MEMBRANE_PREDICTIONS = TEST_DATA_DIR / "membrane" / "Predictions"
REAL_MEMBRANE_TOMO_ID = "emd_10439"
REAL_MEMBRANE_PIXEL_SIZE = 27.36  # --correct_px used in the run
REAL_ACTIN_PREDICTIONS = TEST_DATA_DIR / "actin" / "Predictions"
REAL_ACTIN_TOMO_ID = "TS_00004_sq_df_sorted"
REAL_ACTIN_PIXEL_SIZE = 26.96  # --correct_px used in the run

TOMO_ID = "TS_01"
PIXEL_SIZE = 5.4
# Mask dimensions in x, y, z (the mrc data is stored as z, y, x)
MASK_DIMS = (12, 10, 8)
# [instance id, x, y, z] with the coordinates in Å, as written by Tardis
INSTANCES_ROWS = [
    [0, 0.0, 0.0, 0.0],
    [0, 5.4, 10.8, 16.2],
    [0, 10.8, 21.6, 32.4],
    [1, 54.0, 54.0, 27.0],
    [1, 59.4, 54.0, 27.0],
]


def write_prediction_log(directory: Path, target: str) -> None:
    with open(directory / PREDICTION_LOG, "w") as f:
        f.write(
            "Directory: TS_01.mrc \n"
            "Output Format: mrc_csv \n"
            f"Predict: {target} \n"
            "Device: cuda:0 \n"
        )


def write_semantic_mrc(directory: Path) -> Path:
    x, y, z = MASK_DIMS
    data = np.zeros((z, y, x), dtype=np.int8)
    data[2:4, 3:6, 4:9] = 1
    mrc_file = directory / f"{TOMO_ID}_semantic.mrc"
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = PIXEL_SIZE
    return mrc_file


def write_instances_csv(directory: Path, rows: list[list[float]] | None = None) -> Path:
    csv_file = directory / f"{TOMO_ID}_instances.csv"
    with open(csv_file, "w") as f:
        f.write(",".join(INSTANCES_CSV_HEADER) + "\n")
        for row in INSTANCES_ROWS if rows is None else rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return csv_file
