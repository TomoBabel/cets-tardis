# -*- coding: utf-8 -*-
"""CETS -> Tardis, on the real Tardis runs of the test data and their CETS yaml files (see
``tardis.tests``)."""

import contextlib
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np
import yaml

from cets_data_model.models.models import (
    PointSet3D,
    SegmentationMask3D,
    SegmentationType,
    Spline3D,
)
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation
from tardis.tests import (
    DATA_DIR,
    DATASETS,
    MEMBRANES,
    MICROTUBULES,
    Dataset,
    all_csv_files,
)
from tardis.utils.utils import read_instances_csv


def load_yaml(yaml_file: Path) -> dict:
    with open(yaml_file) as f:
        return yaml.safe_load(f)


def semantic_yaml(dataset: Dataset) -> SegmentationMask3D:
    return SegmentationMask3D.model_validate(load_yaml(dataset.semantic_yaml))


def instances_yaml(dataset: Dataset) -> PointSet3D | Spline3D:
    model_cls = Spline3D if dataset.is_filament else PointSet3D
    return model_cls.model_validate(load_yaml(dataset.instances_yaml))


def dataset_of(csv_file: Path) -> Dataset:
    return next(d for d in DATASETS if csv_file.parent == d.predictions)


class CetsToTardisBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_tardis"))
        self.out_dir = self.test_dir / "out" / "Predictions"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def check_same_csv(self, written: Path, original: Path) -> None:
        """Same name, rows and number format. Going through voxels may change the last
        digit of the values."""
        self.assertEqual(written.name, original.name)
        ids, coords = read_instances_csv(written)
        original_ids, original_coords = read_instances_csv(original)
        self.assertEqual(ids, original_ids)
        np.testing.assert_allclose(coords, original_coords, rtol=1e-12)
        with open(written) as f, open(original) as f_original:
            self.assertEqual(f.readline(), f_original.readline())
            # Float instance ids, as Tardis writes them
            self.assertTrue(f.readline().startswith("0.0,"))


class CetsYamlFilesTest(TestCase):
    def test_yaml_files_are_up_to_date(self) -> None:
        """The yaml files of the test data are what the converters give for the Tardis
        files (the mask with its path relative to ``data``)"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                mask = TardisSemanticSegmentation().tardis_to_cets(
                    dataset.semantic_mrc, dataset.tomo_id
                )
                self.assertEqual(
                    semantic_yaml(dataset),
                    mask.model_copy(update={"path": dataset.relative_semantic_path}),
                )
                instances = TardisInstanceSegmentation().tardis_to_cets(
                    dataset.yaml_instances_csv,
                    dataset.tomo_id,
                    semantic_annotation_id=mask.id,
                )
                self.assertEqual(instances_yaml(dataset), instances)


class CetsToTardisInstancesTest(CetsToTardisBaseTest):
    def test_instances_yaml_to_csv(self) -> None:
        """The instances of the yaml files give back the csv files Tardis wrote"""
        converter = TardisInstanceSegmentation()
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                out = converter.cets_to_tardis(
                    instances_yaml(dataset),
                    self.out_dir / dataset.folder,
                    dataset.tomo_id,
                    filtered=dataset.is_filament,
                )
                self.check_same_csv(out, dataset.yaml_instances_csv)

    def test_real_csv_roundtrip(self) -> None:
        """Every csv -> CETS -> csv: the same rows"""
        converter = TardisInstanceSegmentation()
        for csv_file in all_csv_files():
            with self.subTest(csv_file.name):
                dataset = dataset_of(csv_file)
                instances = converter.tardis_to_cets(csv_file, dataset.tomo_id)
                assert instances is not None
                out = converter.cets_to_tardis(
                    instances,
                    self.out_dir,
                    dataset.tomo_id,
                    filtered=converter.is_filtered(csv_file),
                )
                self.check_same_csv(out, csv_file)

    def test_tomo_name(self) -> None:
        out = TardisInstanceSegmentation().cets_to_tardis(
            instances_yaml(MICROTUBULES), self.out_dir, "tomo_a"
        )
        self.assertEqual(out, self.out_dir / "tomo_a_instances.csv")

    def test_single_instance_without_ids(self) -> None:
        instances = instances_yaml(MEMBRANES)
        out = TardisInstanceSegmentation().cets_to_tardis(
            instances.model_copy(update={"instance_ids": None}),
            self.out_dir,
            MEMBRANES.tomo_id,
        )
        ids, _ = read_instances_csv(out)
        self.assertEqual(ids, [0] * MEMBRANES.n_points)

    def test_pixel_size(self) -> None:
        """The given pixel size is used; without any, the csv cannot be written"""
        converter = TardisInstanceSegmentation()
        instances = instances_yaml(MEMBRANES)
        out = converter.cets_to_tardis(
            instances, self.out_dir, MEMBRANES.tomo_id, pixel_size=1.0
        )
        _, coords = read_instances_csv(out)
        np.testing.assert_allclose(coords, instances.origin3D, rtol=1e-12)
        _, original = read_instances_csv(MEMBRANES.instances_csv)
        np.testing.assert_allclose(
            np.array(coords) * MEMBRANES.pixel_size, original, rtol=1e-6
        )
        without_scale = instances.model_copy(update={"coordinate_transformations": []})
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(without_scale, self.test_dir / "a", "a")
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(instances, self.test_dir / "b", "b", pixel_size=0)

    def test_invalid_instances(self) -> None:
        converter = TardisInstanceSegmentation()
        instances = instances_yaml(MICROTUBULES)
        no_points = instances.model_copy(update={"origin3D": []})
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(no_points, self.test_dir / "a", "a")
        other_ids = instances.model_copy(
            update={"instance_ids": instances.instance_ids[:-1]}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(other_ids, self.test_dir / "b", "b")
        self.assertFalse((self.test_dir / "a").exists())


class CetsToTardisSemanticTest(CetsToTardisBaseTest):
    def test_semantic_yaml_to_mask(self) -> None:
        """The path of the yaml files is relative to ``data``: linked to the real masks"""
        converter = TardisSemanticSegmentation()
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                with contextlib.chdir(DATA_DIR):
                    out = converter.cets_to_tardis(
                        semantic_yaml(dataset), self.out_dir, dataset.tomo_id
                    )
                self.assertEqual(out, self.out_dir / dataset.semantic_mrc.name)
                self.assertTrue(out.is_symlink())
                self.assertEqual(out.resolve(), dataset.semantic_mrc.resolve())

    def test_relative_path_from_another_directory(self) -> None:
        with contextlib.chdir(self.test_dir), self.assertRaises(FileNotFoundError):
            TardisSemanticSegmentation().cets_to_tardis(
                semantic_yaml(MEMBRANES), self.out_dir, MEMBRANES.tomo_id
            )
        self.assertFalse(self.out_dir.exists())

    def test_linked_and_copied(self) -> None:
        converter = TardisSemanticSegmentation()
        mask = converter.tardis_to_cets(MEMBRANES.semantic_mrc, MEMBRANES.tomo_id)
        out = converter.cets_to_tardis(mask, self.out_dir, "tomo_a")
        self.assertEqual(out.name, "tomo_a_semantic.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), MEMBRANES.semantic_mrc.resolve())
        # Again: already there
        self.assertEqual(converter.cets_to_tardis(mask, self.out_dir, "tomo_a"), out)
        # Copied on request
        out = converter.cets_to_tardis(
            mask, self.test_dir / "copy", "tomo_a", copy=True
        )
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_bytes(), MEMBRANES.semantic_mrc.read_bytes())

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        converter = TardisSemanticSegmentation()
        mask = converter.tardis_to_cets(
            MEMBRANES.semantic_mrc, MEMBRANES.tomo_id, pixel_size=27.36
        )
        out = converter.cets_to_tardis(mask, self.out_dir, MEMBRANES.tomo_id)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 27.36, places=4)
        # The whole file is copied, not only its header
        self.assertEqual(out.stat().st_size, MEMBRANES.semantic_mrc.stat().st_size)
        self.assertEqual(
            mrcfile.read(out).tobytes(), mrcfile.read(MEMBRANES.semantic_mrc).tobytes()
        )
        with mrcfile.open(MEMBRANES.semantic_mrc, header_only=True) as mrc:
            self.assertAlmostEqual(
                float(mrc.voxel_size.x), MEMBRANES.pixel_size, places=4
            )

    def test_existing_output(self) -> None:
        self.out_dir.mkdir(parents=True)
        (self.out_dir / MEMBRANES.semantic_mrc.name).write_text("other")
        mask = TardisSemanticSegmentation().tardis_to_cets(
            MEMBRANES.semantic_mrc, MEMBRANES.tomo_id
        )
        with self.assertRaises(FileExistsError):
            TardisSemanticSegmentation().cets_to_tardis(
                mask, self.out_dir, MEMBRANES.tomo_id
            )

    def test_not_a_semantic_mask(self) -> None:
        converter = TardisSemanticSegmentation()
        mask = converter.tardis_to_cets(MEMBRANES.semantic_mrc, MEMBRANES.tomo_id)
        instance_mask = mask.model_copy(
            update={"segmentation_type": SegmentationType.instance.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(instance_mask, self.out_dir, "tomo_a")
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(
                mask.model_copy(update={"path": None}), self.out_dir, "tomo_a"
            )
