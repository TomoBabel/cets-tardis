# cets-tardis

Converters between [TARDIS](https://github.com/SMLC-NYSBC/TARDIS) segmentation outputs and
CETS, the cryo-electron tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | TARDIS files (in `<out_dir>/Predictions/`) | CETS objects | Directions |
|---|---|---|---|
| Semantic segmentation | `<tomo>_semantic.mrc` | `SegmentationMask3D` (`semantic`, target = 1) | TARDIS → CETS, CETS → TARDIS |
| Instances of filaments (microtubules, actin, general filaments) | `<tomo>_instances.csv`, `<tomo>_instances_filter.csv` (length-filtered) | `Spline3D` (one spline per instance) | TARDIS → CETS, CETS → TARDIS |
| Instances of membranes | `<tomo>_instances.csv` | `PointSet3D` (point cloud of each membrane) | TARDIS → CETS, CETS → TARDIS |

The segmented target (membrane, microtubule, actin, general filament) is read from the
`prediction_log.txt` TARDIS writes next to its outputs, and becomes the annotated object of the
annotation.

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `tardis` package and the `cets-tardis` command. TARDIS itself is not needed.

## Command line

```text
cets-tardis {semantic,instances} {to-cets,from-cets} [options]
```

```bash
# TARDIS -> CETS
cets-tardis semantic to-cets --semantic Predictions/TS_03_semantic.mrc --tomo-id TS_03 -o semantic.yaml
cets-tardis instances to-cets --instances Predictions/TS_03_instances.csv --tomo-id TS_03 \
    --semantic-annotation-id tardis_microtubule_semantic_TS_03 -o instances.yaml

# CETS -> TARDIS file names
cets-tardis semantic from-cets -i semantic.yaml --output-dir out/Predictions
cets-tardis instances from-cets -i instances.yaml --output-dir out/Predictions
```

`to-cets` writes a CETS yaml file with the entry `segmentation` (semantic mask) or
`coordinates` (instances); `from-cets` reads it. Every sub-command has `--help`. Errors print a
one-line message and exit with code 1; put `--debug` before the data type to also print the
traceback.

| `to-cets` option | Meaning |
|---|---|
| `--semantic` / `--instances` (required) | The mask or the csv. |
| `--tomo-id` (required) | Id of the CETS tomogram it was obtained from (`source_tomogram_id`). It also names the coordinate systems (`<tomo id>_array`, `<tomo id>_physical`), so it must be the id of the CETS tomogram. |
| `--target {Membrane,Microtubule,Actin,General_filament}` | Segmented target. Default: the one of the `prediction_log.txt` next to the input. |
| `--pixel-size` | Sampling rate (Å) used by TARDIS. Default for masks: the mrc header. Default for instances: the header of the semantic mask next to the csv, else of the input tomogram. An error is raised if it cannot be found. |
| `--semantic-annotation-id` | For instances: id of the semantic mask annotation of the same run, which they were computed from (`derived_from_annotation_id`). |
| `--annotation-id` | Id of the annotation, unique within its region (needed when several runs on the same tomogram are converted). Default: `tardis_<target>_semantic_<tomo id>` / `tardis_<target>_instances[_filtered]_<tomo id>`. |

| `from-cets` option | Meaning |
|---|---|
| `--output-dir` (required) | Directory to write the file in (natively, the `Predictions` directory). |
| `--tomo-name` | Tomogram name in the file names (the input tomogram file name without extension). Default: the `source_tomogram_id`. |
| `--pixel-size` | For instances: sampling rate (Å) to write the coordinates in Å. Default: the one of the annotation. |
| `--filtered` | For instances: write `<tomo>_instances_filter.csv` (the length-filtered filaments). |
| `--copy` | For masks: copy the file instead of linking it (it is always copied if its header sampling rate is not the CETS one). |

## Python API

```python
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation

mask = TardisSemanticSegmentation().tardis_to_cets("Predictions/TS_03_semantic.mrc", "TS_03")
instances = TardisInstanceSegmentation().tardis_to_cets(
    "Predictions/TS_03_instances.csv", "TS_03", semantic_annotation_id=mask.id
)
TardisInstanceSegmentation().cets_to_tardis(instances, "out/Predictions", tomo_name="TS_03")
```

`tardis.utils.discovery.find_tardis_predictions` finds the TARDIS outputs of a directory.

## Conventions

- **Instances.** Each row of the csv is a point `[instance id, x, y, z]`. All the instances of
  a tomogram go into one annotation, whose `instance_ids` keep the instance of each point. For
  filaments, the points sharing an instance id form one spline, in the order of the file.
- **Positions.** TARDIS writes the coordinates in Å from the bottom left corner of the tomogram
  (voxel index × pixel size). In CETS they are voxel positions in the tomogram (`origin3D`),
  with the `array_to_physical` scale to Å; they are multiplied back by the pixel size when
  writing (the last digit may change).
- **Masks are not converted.** CETS points to the TARDIS mrc files; `from-cets` links them, or
  copies them when their header sampling rate is not the CETS one.

## Limitations

- The target is not written back: TARDIS records it in `prediction_log.txt`, not in the csv or
  the mask.
- The masks are assumed to be on the voxel grid of the tomogram given with `--tomo-id`; this is
  not checked.
- Other TARDIS outputs (e.g. Amira spatial graphs, instance masks) are not converted.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest tardis/tests/tests_*.py    # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy tardis
```

### Tests

| File | What it tests |
|---|---|
| `tardis/tests/tests_cets_tardis.py` | TARDIS → CETS: the consistency of the test data (sizes, pixel sizes, voxels and instances against `prediction_log.txt`), the discovery of the runs, the prediction log, the semantic masks, the instances of every csv (raw and length-filtered) and the pixel size of the instances (given, semantic mask or input tomogram). |
| `tardis/tests/tests_cets_to_tardis.py` | CETS → TARDIS: the CETS yaml files of the test data give back the files TARDIS wrote; round trips of every csv; masks linked or copied. |
| `tardis/tests/tests_discovery.py` | `find_tardis_predictions` on directory layouts built by the tests. |
| `tardis/tests/tests_cli.py` | Every `cets-tardis` sub-command on the three runs, in both directions (`from-cets` on the yaml files of the test data), plus the round trips, the options, the errors and the usage errors. |

```bash
pytest tardis/tests/tests_cli.py                    # one file
pytest tardis/tests/tests_cli.py -k semantic        # some tests
```

All the tests use real TARDIS 0.3.10 runs (`--output_format mrc_csv`), one per kind of target,
in `tardis/tests/data` (`DATA_DIR` in `tardis/tests/__init__.py`, which also describes each run:
`ACTIN`, `MEMBRANES`, `MICROTUBULES`):

| Folder | Tomogram | Target | Voxel size (Å) | Size (x, y, z) | Instances (`_instances.csv` / `_instances_filter.csv`) |
|---|---|---|---|---|---|
| `actin_filaments` | `TS_00004_sq_df_sorted` | Actin | 26.96 | 464 x 464 x 312 | 387 / 254 |
| `membranes` | `emd_10439` | Membrane | 54.72 | 232 x 232 x 125 | 26 / - |
| `microtubules` | `GMPCPP_S1_tomo1_rec` | Microtubule | 31.6 | 512 x 512 x 100 | 25 / 16 |

Each folder has the tomogram (`<tomo>.mrc`) and the `Predictions` directory TARDIS wrote:
`prediction_log.txt`, `<tomo>_semantic.mrc`, `<tomo>_instances.csv` and, for filaments,
`<tomo>_instances_filter.csv`. It also has the CETS yaml files of the semantic mask and of the
instances (the length-filtered ones for filaments), written by `write_semantic_yaml` and
`write_instances_yaml`: the inputs of the `from-cets` tests. The `path` of the masks in their
yaml files is relative to `tardis/tests/data`, so the tests read them from that directory.

The other files the tests need (runs without a log, without a semantic mask, with no instances,
...) are derived from these in a temporary directory by the helpers of
`tardis/tests/__init__.py`; the files in `tardis/tests/data` are never modified by the tests. A
test checks that the yaml files are still what the converters give for the TARDIS files; if
the converters change, write them again and make the path of the masks relative to
`tardis/tests/data` again.
