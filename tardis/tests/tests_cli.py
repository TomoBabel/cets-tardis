# -*- coding: utf-8 -*-
"""Tests of the ``cets-tardis`` command line, on the real Tardis runs of the test data and
their CETS yaml files (see ``tardis.tests``), in both directions."""

import contextlib
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    AnnotationType,
    PointSet3D,
    SegmentationMask3D,
    SegmentedFeatureType,
    Spline3D,
)
from cets_data_model.utils.yaml_io import load_cets_object
from tardis.cli import main
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation
from tardis.tests import (
    ACTIN,
    DATA_DIR,
    DATASETS,
    MEMBRANES,
    MICROTUBULES,
    Dataset,
    all_csv_files,
    copy_predictions,
    write_empty_csv,
)
from tardis.utils.utils import read_instances_csv


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-tardis with the given arguments; returns the exit code and what it printed
    to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file under a directory, to check that it is not modified."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def dataset_of(csv_file: Path) -> Dataset:
    return next(d for d in DATASETS if csv_file.parent == d.predictions)


class CetsTardisCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_tardis_cli"))
        self.yaml = self.dir / "cets.yaml"
        self.out_dir = self.dir / "out" / "Predictions"  # created by the command

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-tardis, checks its exit code and returns what it printed to stdout
        and stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return out + err

    def check_same_csv(self, written: Path, original: Path) -> None:
        self.assertEqual(written.name, original.name)
        ids, coords = read_instances_csv(written)
        original_ids, original_coords = read_instances_csv(original)
        self.assertEqual(ids, original_ids)
        np.testing.assert_allclose(coords, original_coords, rtol=1e-12)

    def _semantic_to_cets(self, mask_file: Path, tomo_id: str, *extra: str) -> str:
        return self.assert_run(
            ["semantic", "to-cets", "--semantic", str(mask_file), "--tomo-id", tomo_id]
            + [*extra, "-o", str(self.yaml)]
        )

    def _instances_args(self, csv_file: Path, tomo_id: str, *extra: str) -> List[str]:
        return [
            "instances",
            "to-cets",
            "--instances",
            str(csv_file),
            "--tomo-id",
            tomo_id,
            *extra,
            "-o",
            str(self.yaml),
        ]

    @staticmethod
    def _load_instances(yaml_file: Path, dataset: Dataset) -> PointSet3D | Spline3D:
        model_cls = Spline3D if dataset.is_filament else PointSet3D
        return load_cets_object(yaml_file, "coordinates", model_cls)

    # ------------------------------------------------------------------ semantic
    def test_semantic_to_cets(self) -> None:
        """Every mask: the target of the log and the pixel size of the header, under the
        key ``segmentation``"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                output = self._semantic_to_cets(dataset.semantic_mrc, dataset.tomo_id)
                self.assertIn(str(self.yaml), output)
                mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
                self.assertEqual(
                    mask,
                    TardisSemanticSegmentation().tardis_to_cets(
                        dataset.semantic_mrc, dataset.tomo_id
                    ),
                )
                self.assertEqual(mask.id, dataset.semantic_id)
                self.assertEqual(mask.path, str(dataset.semantic_mrc.resolve()))
                self.assertEqual((mask.width, mask.height, mask.depth), dataset.grid)
                self.assertAlmostEqual(
                    mask.coordinate_transformations[0].scale[0],
                    dataset.pixel_size,
                    places=5,
                )

    def test_semantic_options(self) -> None:
        """The target and the pixel size given override the log and the header"""
        self._semantic_to_cets(
            MEMBRANES.semantic_mrc,
            "tomo_1",
            "--target",
            "microtubule",
            "--pixel-size",
            "27.36",
            "--annotation-id",
            "sem_1",
        )
        mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
        self.assertEqual(mask.id, "sem_1")
        self.assertEqual(mask.source_tomogram_id, "tomo_1")
        self.assertEqual(
            [cs.name for cs in mask.coordinate_systems],
            ["tomo_1_array", "tomo_1_physical"],
        )
        self.assertEqual(
            mask.annotated_object.feature_type, SegmentedFeatureType.microtubule
        )
        self.assertEqual(mask.coordinate_transformations[0].scale, [27.36] * 3)

    def test_semantic_from_cets_yaml(self) -> None:
        """The yaml files of the test data: their paths are relative to ``data``"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                with contextlib.chdir(DATA_DIR):
                    self.assert_run(
                        ["semantic", "from-cets", "-i", str(dataset.semantic_yaml)]
                        + ["--output-dir", str(self.out_dir)]
                    )
                # Default name: the source tomogram id; linked
                out = self.out_dir / dataset.semantic_mrc.name
                self.assertTrue(out.is_symlink())
                self.assertEqual(out.resolve(), dataset.semantic_mrc.resolve())
        # From another directory the mask is not found
        with contextlib.chdir(self.dir):
            output = self.assert_run(
                ["semantic", "from-cets", "-i", str(MEMBRANES.semantic_yaml)]
                + ["--output-dir", str(self.dir / "other")],
                1,
            )
        self.assertIn(MEMBRANES.semantic_mrc.name, output)
        self.assertFalse((self.dir / "other").exists())

    def test_semantic_round_trip(self) -> None:
        self._semantic_to_cets(MEMBRANES.semantic_mrc, MEMBRANES.tomo_id)
        from_cets = ["semantic", "from-cets", "-i", str(self.yaml), "--output-dir"]
        self.assert_run([*from_cets, str(self.out_dir)])
        out = self.out_dir / MEMBRANES.semantic_mrc.name
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), MEMBRANES.semantic_mrc.resolve())
        self.assert_run([*from_cets, str(self.out_dir), "--tomo-name", "a", "--copy"])
        out = self.out_dir / "a_semantic.mrc"
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_bytes(), MEMBRANES.semantic_mrc.read_bytes())

    def test_semantic_from_cets_other_voxel_size(self) -> None:
        """Copied (whole) with the header updated"""
        self._semantic_to_cets(
            MEMBRANES.semantic_mrc, MEMBRANES.tomo_id, "--pixel-size", "27.36"
        )
        self.assert_run(
            [
                "semantic",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.dir),
            ]
        )
        out = self.dir / MEMBRANES.semantic_mrc.name
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 27.36, places=4)
        self.assertEqual(out.stat().st_size, MEMBRANES.semantic_mrc.stat().st_size)

    # ------------------------------------------------------------------ instances
    def test_instances_to_cets(self) -> None:
        """Every csv, under the key ``coordinates``: membranes -> PointSet3D, filaments ->
        Spline3D; the pixel size of the semantic mask next to the csv"""
        converter = TardisInstanceSegmentation()
        for csv_file in all_csv_files():
            with self.subTest(csv_file.name):
                dataset = dataset_of(csv_file)
                self.assert_run(
                    self._instances_args(
                        csv_file,
                        dataset.tomo_id,
                        "--semantic-annotation-id",
                        dataset.semantic_id,
                    )
                )
                instances = self._load_instances(self.yaml, dataset)
                self.assertEqual(
                    instances,
                    converter.tardis_to_cets(
                        csv_file,
                        dataset.tomo_id,
                        semantic_annotation_id=dataset.semantic_id,
                    ),
                )
                self.assertEqual(
                    instances.annotation_type,
                    AnnotationType.spline_3D
                    if dataset.is_filament
                    else AnnotationType.point_set_3D,
                )
                self.assertEqual(
                    instances.derived_from_annotation_id, dataset.semantic_id
                )
                ids, coords = read_instances_csv(csv_file)
                self.assertEqual(instances.instance_ids, ids)
                np.testing.assert_allclose(
                    np.array(instances.origin3D) * dataset.pixel_size, coords, rtol=1e-6
                )

    def test_instances_to_cets_gives_the_yaml_files(self) -> None:
        """The same annotations as the yaml files of the test data"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                self.assert_run(
                    self._instances_args(
                        dataset.yaml_instances_csv,
                        dataset.tomo_id,
                        "--semantic-annotation-id",
                        dataset.semantic_id,
                    )
                )
                self.assertEqual(
                    self._load_instances(self.yaml, dataset),
                    self._load_instances(dataset.instances_yaml, dataset),
                )

    def test_instances_options(self) -> None:
        """The target, pixel size and id given"""
        self.assert_run(
            self._instances_args(
                MEMBRANES.instances_csv,
                MEMBRANES.tomo_id,
                "--target",
                "Actin",
                "--pixel-size",
                "1",
                "--annotation-id",
                "a1",
            )
        )
        splines = load_cets_object(self.yaml, "coordinates", Spline3D)
        self.assertEqual(splines.id, "a1")
        self.assertEqual(
            splines.annotated_object.feature_type, SegmentedFeatureType.actin_filament
        )
        self.assertIsNone(splines.derived_from_annotation_id)
        _, coords = read_instances_csv(MEMBRANES.instances_csv)
        np.testing.assert_allclose(splines.origin3D, coords)

    def test_instances_pixel_size_from_the_tomogram(self) -> None:
        """Without the semantic mask, the input tomogram next to the Predictions dir"""
        predictions = copy_predictions(MICROTUBULES, self.dir / "run", tomogram=True)
        self.assert_run(
            self._instances_args(
                predictions / MICROTUBULES.instances_csv.name, MICROTUBULES.tomo_id
            )
        )
        splines = load_cets_object(self.yaml, "coordinates", Spline3D)
        self.assertAlmostEqual(
            splines.coordinate_transformations[0].scale[0],
            MICROTUBULES.pixel_size,
            places=5,
        )

    def test_instances_from_cets_yaml(self) -> None:
        """The yaml files of the test data give back the csv files Tardis wrote (the
        filtered ones for filaments)"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                out_dir = self.out_dir / dataset.folder
                args = ["instances", "from-cets", "-i", str(dataset.instances_yaml)]
                args += ["--output-dir", str(out_dir)]
                if dataset.is_filament:
                    args.append("--filtered")
                self.assert_run(args)
                (written,) = list(out_dir.iterdir())
                self.check_same_csv(written, dataset.yaml_instances_csv)

    def test_instances_round_trip(self) -> None:
        """Every csv -> CETS -> csv: the same rows"""
        for csv_file in all_csv_files():
            with self.subTest(csv_file.name):
                dataset = dataset_of(csv_file)
                self.assert_run(self._instances_args(csv_file, dataset.tomo_id))
                args = ["instances", "from-cets", "-i", str(self.yaml), "--output-dir"]
                args.append(str(self.out_dir))
                if TardisInstanceSegmentation.is_filtered(csv_file):
                    args.append("--filtered")
                self.assert_run(args)
                self.check_same_csv(self.out_dir / csv_file.name, csv_file)

    def test_instances_from_cets_options(self) -> None:
        """--tomo-name names the file; another pixel size scales the coordinates"""
        from_cets = ["instances", "from-cets", "-i", str(MEMBRANES.instances_yaml)]
        self.assert_run(
            from_cets
            + [
                "--output-dir",
                str(self.out_dir),
                "--tomo-name",
                "t",
                "--pixel-size",
                "1",
            ]
        )
        _, coords = read_instances_csv(self.out_dir / "t_instances.csv")
        _, original = read_instances_csv(MEMBRANES.instances_csv)
        np.testing.assert_allclose(
            coords, np.array(original) / MEMBRANES.pixel_size, rtol=1e-6
        )
        # Written again over the existing file, with the pixel size of the annotation
        self.assert_run(
            from_cets + ["--output-dir", str(self.out_dir), "--tomo-name", "t"]
        )
        _, coords = read_instances_csv(self.out_dir / "t_instances.csv")
        np.testing.assert_allclose(coords, original, rtol=1e-12)

    # ------------------------------------------------------------------ errors
    def test_errors_give_exit_code_1(self) -> None:
        # No target given nor prediction_log.txt
        predictions = copy_predictions(MEMBRANES, self.dir / "no_log", log=False)
        csv_file = predictions / MEMBRANES.instances_csv.name
        output = self.assert_run(
            self._instances_args(csv_file, MEMBRANES.tomo_id, "--pixel-size", "5"), 1
        )
        self.assertIn("prediction_log.txt", output)
        # No pixel size: no semantic mask nor tomogram
        predictions = copy_predictions(MEMBRANES, self.dir / "no_mrc")
        csv_file = predictions / MEMBRANES.instances_csv.name
        self.assert_run(self._instances_args(csv_file, MEMBRANES.tomo_id), 1)
        # Tardis found no instance
        write_empty_csv(csv_file)
        self.assert_run(
            self._instances_args(csv_file, MEMBRANES.tomo_id, "--pixel-size", "5"), 1
        )
        self.assertFalse(self.yaml.exists())
        # Missing input files
        self.assert_run(
            self._instances_args(
                self.dir / "a_instances.csv", "a", "--pixel-size", "5"
            ),
            1,
        )
        self.assert_run(
            ["semantic", "to-cets", "--semantic", str(self.dir / "a.mrc")]
            + ["--tomo-id", "a", "-o", str(self.yaml)],
            1,
        )
        self.assert_run(
            ["instances", "from-cets", "-i", str(self.dir / "a.yaml")]
            + ["--output-dir", str(self.out_dir)],
            1,
        )
        # A semantic mask is not an instances annotation, and the other way round
        self.assert_run(
            ["instances", "from-cets", "-i", str(MEMBRANES.semantic_yaml)]
            + ["--output-dir", str(self.out_dir)],
            1,
        )
        self.assert_run(
            ["semantic", "from-cets", "-i", str(ACTIN.instances_yaml)]
            + ["--output-dir", str(self.out_dir)],
            1,
        )
        self.assertFalse(self.out_dir.exists())

    def test_debug_prints_the_traceback(self) -> None:
        args = ["semantic", "from-cets", "-i", str(self.dir / "a.yaml")]
        args += ["--output-dir", str(self.out_dir)]
        self.assertNotIn("Traceback", self.assert_run(args, 1))
        self.assertIn("Traceback", self.assert_run(["--debug", *args], 1))

    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(DATA_DIR)
        for dataset in DATASETS:
            out_dir = self.dir / dataset.folder
            self._semantic_to_cets(dataset.semantic_mrc, dataset.tomo_id)
            self.assert_run(
                ["semantic", "from-cets", "-i", str(self.yaml), "--output-dir"]
                + [str(out_dir), "--copy"]
            )
            with contextlib.chdir(DATA_DIR):
                self.assert_run(
                    ["semantic", "from-cets", "-i", str(dataset.semantic_yaml)]
                    + ["--output-dir", str(out_dir), "--tomo-name", "linked"]
                )
            self.assert_run(
                self._instances_args(dataset.instances_csv, dataset.tomo_id)
            )
            self.assert_run(
                ["instances", "from-cets", "-i", str(dataset.instances_yaml)]
                + ["--output-dir", str(out_dir)]
            )
        self.assertEqual(tree_digest(DATA_DIR), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["other", "to-cets"],
            ["semantic"],
            ["semantic", "other"],
            # Required arguments
            ["semantic", "to-cets", "--tomo-id", "a", "-o", str(self.yaml)],
            ["semantic", "to-cets", "--semantic", str(MEMBRANES.semantic_mrc)],
            ["semantic", "from-cets", "-i", str(MEMBRANES.semantic_yaml)],
            ["instances", "to-cets", "--instances", str(MEMBRANES.instances_csv)],
            ["instances", "from-cets", "--output-dir", str(self.out_dir)],
            # Invalid values
            self._instances_args(MEMBRANES.instances_csv, "a", "--target", "Other"),
            self._instances_args(MEMBRANES.instances_csv, "a", "--pixel-size", "x"),
            # Options of the other direction
            ["semantic", "to-cets", "--semantic", str(MEMBRANES.semantic_mrc)]
            + ["--tomo-id", "a", "-o", str(self.yaml), "--copy"],
            ["instances", "from-cets", "-i", str(MEMBRANES.instances_yaml)]
            + ["--output-dir", str(self.out_dir), "--target", "Membrane"],
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as raised:
                    run(args)
                self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.yaml.exists())
        self.assertFalse(self.out_dir.exists())

    def test_target_names(self) -> None:
        """The target can be given by its name or its value, in any case"""
        for target in ("membrane", "Membrane", "MEMBRANE", "general_filament"):
            with self.subTest(target):
                self.assert_run(
                    self._instances_args(
                        MEMBRANES.instances_csv, "a", "--target", target
                    )
                )

    def test_help(self) -> None:
        help_args: List[List[str]] = [
            [],
            ["semantic"],
            ["instances"],
            ["semantic", "to-cets"],
            ["semantic", "from-cets"],
            ["instances", "to-cets"],
            ["instances", "from-cets"],
        ]
        for args in help_args:
            with self.subTest(args=args):
                out = io.StringIO()
                with (
                    contextlib.redirect_stdout(out),
                    self.assertRaises(SystemExit) as e,
                ):
                    main([*args, "--help"])
                self.assertEqual(e.exception.code, 0)
                self.assertIn("usage: cets-tardis", out.getvalue())
