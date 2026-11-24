"""Test data: real Tardis 0.3.10 (tardis_em-pytorch) runs on three tomograms, one per kind
of target, in ``data/<dataset>``:

- ``<tomo>.mrc``: the segmented tomogram.
- ``Predictions/``: what Tardis wrote for it (``--output_format mrc_csv``): the
  ``prediction_log.txt``, the semantic mask (``<tomo>_semantic.mrc``), all the instances
  (``<tomo>_instances.csv``) and, for filaments, the length-filtered ones
  (``<tomo>_instances_filter.csv``); and the CETS yaml files of the semantic mask and of the
  instances (the filtered ones for filaments), as written by ``write_semantic_yaml`` and
  ``write_instances_yaml``: the inputs of the CETS -> Tardis tests. The path of the mask in
  its yaml is relative to ``data``, so the masks are read from there.

The other files the tests need (outputs without a log, without a semantic mask, with no
instances, ...) are derived from these in a temporary directory.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tardis.constants import INSTANCES_CSV_HEADER, PREDICTION_LOG, TardisTarget

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Dataset:
    """A real Tardis run, with the values read from its files."""

    folder: str  # in DATA_DIR
    tomo_id: str  # name of the tomogram file, prefix of the Tardis files
    target: TardisTarget
    pixel_size: float  # header of the tomogram and the mask (Å/voxel)
    grid: tuple[int, int, int]  # x, y, z
    semantic_voxels: int  # voxels of the target in the mask
    n_instances: int  # in <tomo>_instances.csv
    n_points: int
    n_filter_instances: Optional[int] = None  # in <tomo>_instances_filter.csv
    n_filter_points: Optional[int] = None

    @property
    def root(self) -> Path:
        return DATA_DIR / self.folder

    @property
    def tomogram(self) -> Path:
        return self.root / f"{self.tomo_id}.mrc"

    @property
    def predictions(self) -> Path:
        return self.root / "Predictions"

    @property
    def log(self) -> Path:
        return self.predictions / PREDICTION_LOG

    @property
    def semantic_mrc(self) -> Path:
        return self.predictions / f"{self.tomo_id}_semantic.mrc"

    @property
    def instances_csv(self) -> Path:
        return self.predictions / f"{self.tomo_id}_instances.csv"

    @property
    def instances_filter_csv(self) -> Optional[Path]:
        if self.n_filter_instances is None:
            return None
        return self.predictions / f"{self.tomo_id}_instances_filter.csv"

    @property
    def is_filament(self) -> bool:
        return self.n_filter_instances is not None

    @property
    def semantic_id(self) -> str:
        return f"tardis_{self.target.name}_semantic_{self.tomo_id}"

    @property
    def instances_id(self) -> str:
        """Id of the instances of the yaml: the filtered ones for filaments."""
        filtered = "_filtered" if self.is_filament else ""
        return f"tardis_{self.target.name}_instances{filtered}_{self.tomo_id}"

    @property
    def yaml_instances_csv(self) -> Path:
        """The csv the instances of the yaml were converted from."""
        return self.instances_filter_csv or self.instances_csv

    @property
    def semantic_yaml(self) -> Path:
        return self.predictions / f"{self.semantic_id}_tardis_to_cets.yaml"

    @property
    def instances_yaml(self) -> Path:
        return self.predictions / f"{self.instances_id}_tardis_to_cets.yaml"

    @property
    def relative_semantic_path(self) -> str:
        """Path of the mask in its yaml file, relative to DATA_DIR."""
        return f"./{self.folder}/Predictions/{self.semantic_mrc.name}"


ACTIN = Dataset(
    folder="actin_filaments",
    tomo_id="TS_00004_sq_df_sorted",
    target=TardisTarget.actin,
    pixel_size=26.96,
    grid=(464, 464, 312),
    semantic_voxels=568584,
    n_instances=387,
    n_points=4456,
    n_filter_instances=254,
    n_filter_points=3411,
)
MEMBRANES = Dataset(
    folder="membranes",
    tomo_id="emd_10439",
    target=TardisTarget.membrane,
    pixel_size=54.72,
    grid=(232, 232, 125),
    semantic_voxels=191081,
    n_instances=26,
    n_points=4690,
)
MICROTUBULES = Dataset(
    folder="microtubules",
    tomo_id="GMPCPP_S1_tomo1_rec",
    target=TardisTarget.microtubule,
    pixel_size=31.6,
    grid=(512, 512, 100),
    semantic_voxels=52178,
    n_instances=25,
    n_points=852,
    n_filter_instances=16,
    n_filter_points=847,
)
DATASETS = (ACTIN, MEMBRANES, MICROTUBULES)

# First row of each instances csv, as Tardis wrote it
FIRST_ROWS = {
    ACTIN.instances_csv: "0.0,2750.3926173157233,8735.033710929603,5553.241833207144",
    MEMBRANES.instances_csv: "0.0,1751.04,9630.72,2298.24",
    MICROTUBULES.instances_csv: (
        "0.0,11755.94845330786,9101.366556694014,2335.7953853277513"
    ),
}


def all_csv_files() -> List[Path]:
    """Every instances csv of the test data."""
    files = []
    for dataset in DATASETS:
        files.append(dataset.instances_csv)
        if dataset.instances_filter_csv:
            files.append(dataset.instances_filter_csv)
    return files


def log_numbers(dataset: Dataset, key: str) -> List[int]:
    """The numbers of the lines ``<key> Prediction: ...`` of the prediction log of a dataset
    (the voxels of the semantic mask, the instances of each csv)."""
    text = dataset.log.read_text()
    return [
        int(float(n)) for n in re.findall(rf"{key} Prediction: .*?: ([\d.]+)", text)
    ]


def copy_predictions(
    dataset: Dataset,
    directory: Path,
    csv_files: bool = True,
    log: bool = True,
    semantic: bool = False,
    tomogram: bool = False,
) -> Path:
    """A Predictions directory under ``directory`` with some of the files of a dataset:
    the csv files and the log are copied, and the mrc files (big) linked. Returns it."""
    predictions = directory / "Predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    if csv_files:
        for csv_file in (dataset.instances_csv, dataset.instances_filter_csv):
            if csv_file:
                shutil.copyfile(csv_file, predictions / csv_file.name)
    if log:
        shutil.copyfile(dataset.log, predictions / PREDICTION_LOG)
    if semantic:
        (predictions / dataset.semantic_mrc.name).symlink_to(dataset.semantic_mrc)
    if tomogram:
        (directory / dataset.tomogram.name).symlink_to(dataset.tomogram)
    return predictions


def write_log(directory: Path, target: str) -> Path:
    """A prediction_log.txt of a run with another target."""
    log = directory / PREDICTION_LOG
    log.write_text(MEMBRANES.log.read_text().replace("Membrane", target))
    return log


def write_empty_csv(csv_file: Path) -> Path:
    """An instances csv of a run in which Tardis found no instance: only the header."""
    csv_file.write_text(",".join(INSTANCES_CSV_HEADER) + "\n")
    return csv_file
