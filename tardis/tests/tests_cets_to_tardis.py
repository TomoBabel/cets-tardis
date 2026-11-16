# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import SegmentationType
from tardis.constants import TardisTarget
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation
from tardis.tests import (
    INSTANCES_ROWS,
    PIXEL_SIZE,
    REAL_ACTIN_PIXEL_SIZE,
    REAL_ACTIN_PREDICTIONS,
    REAL_ACTIN_TOMO_ID,
    REAL_MEMBRANE_PIXEL_SIZE,
    REAL_MEMBRANE_PREDICTIONS,
    REAL_MEMBRANE_TOMO_ID,
    TOMO_ID,
    write_instances_csv,
    write_semantic_mrc,
)
from tardis.utils.utils import read_instances_csv


class CetsToTardisBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_tardis"))
        self.out_dir = self.test_dir / "out" / "Predictions"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _check_same_csv(self, written: Path, original: Path) -> None:
        ids, coords = read_instances_csv(written)
        original_ids, original_coords = read_instances_csv(original)
        self.assertEqual(ids, original_ids)
        np.testing.assert_allclose(coords, original_coords, rtol=1e-12)
        # Same header and number format: float instance ids
        with open(written) as f:
            self.assertEqual(f.readline().strip(), "IDs,X [A],Y [A],Z [A]")
            self.assertTrue(f.readline().startswith(f"{float(original_ids[0])},"))


class CetsToTardisInstancesTest(CetsToTardisBaseTest):
    def test_instances_roundtrip(self) -> None:
        """csv -> CETS -> csv gives back the original rows"""
        csv_file = write_instances_csv(self.test_dir)
        converter = TardisInstanceSegmentation()
        for target in (TardisTarget.microtubule, TardisTarget.membrane):
            instances = converter.tardis_to_cets(
                csv_file, TOMO_ID, PIXEL_SIZE, target=target
            )
            assert instances is not None
            out = converter.cets_to_tardis(
                instances, self.out_dir / target.name, TOMO_ID
            )
            self.assertEqual(out.name, f"{TOMO_ID}_instances.csv")
            self._check_same_csv(out, csv_file)

    def test_real_outputs_roundtrip(self) -> None:
        converter = TardisInstanceSegmentation()
        for predictions, tomo_id, pixel_size, suffix in (
            (
                REAL_MEMBRANE_PREDICTIONS,
                REAL_MEMBRANE_TOMO_ID,
                REAL_MEMBRANE_PIXEL_SIZE,
                "_instances.csv",
            ),
            (
                REAL_ACTIN_PREDICTIONS,
                REAL_ACTIN_TOMO_ID,
                REAL_ACTIN_PIXEL_SIZE,
                "_instances.csv",
            ),
            (
                REAL_ACTIN_PREDICTIONS,
                REAL_ACTIN_TOMO_ID,
                REAL_ACTIN_PIXEL_SIZE,
                "_instances_filter.csv",
            ),
        ):
            csv_file = predictions / f"{tomo_id}{suffix}"
            instances = converter.tardis_to_cets(csv_file, tomo_id, pixel_size)
            assert instances is not None
            out = converter.cets_to_tardis(
                instances,
                self.out_dir,
                tomo_id,
                filtered=converter.is_filtered(csv_file),
            )
            self.assertEqual(out.name, csv_file.name)
            self._check_same_csv(out, csv_file)

    def test_single_instance_without_ids(self) -> None:
        csv_file = write_instances_csv(self.test_dir)
        converter = TardisInstanceSegmentation()
        instances = converter.tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE, target=TardisTarget.membrane
        )
        assert instances is not None
        out = converter.cets_to_tardis(
            instances.model_copy(update={"instance_ids": None}), self.out_dir, TOMO_ID
        )
        ids, _ = read_instances_csv(out)
        self.assertEqual(ids, [0] * len(INSTANCES_ROWS))

    def test_pixel_size(self) -> None:
        """The given pixel size is used; without any, the csv cannot be written"""
        csv_file = write_instances_csv(self.test_dir)
        converter = TardisInstanceSegmentation()
        instances = converter.tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE, target=TardisTarget.membrane
        )
        assert instances is not None
        out = converter.cets_to_tardis(
            instances, self.out_dir, TOMO_ID, pixel_size=2 * PIXEL_SIZE
        )
        _, coords = read_instances_csv(out)
        np.testing.assert_allclose(
            coords, 2 * np.array([row[1:] for row in INSTANCES_ROWS])
        )
        without_scale = instances.model_copy(update={"coordinate_transformations": []})
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(without_scale, self.out_dir / "other", TOMO_ID)
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(instances, self.out_dir, TOMO_ID, pixel_size=0.0)


class CetsToTardisSemanticTest(CetsToTardisBaseTest):
    def test_linked_with_the_tardis_name(self) -> None:
        converter = TardisSemanticSegmentation()
        mask_file = write_semantic_mrc(self.test_dir)
        mask = converter.tardis_to_cets(
            mask_file, TOMO_ID, target=TardisTarget.membrane
        )
        out = converter.cets_to_tardis(mask, self.out_dir, "tomo_a")
        self.assertEqual(out.name, "tomo_a_semantic.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), mask_file.resolve())

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        converter = TardisSemanticSegmentation()
        mask_file = write_semantic_mrc(self.test_dir)
        mask = converter.tardis_to_cets(
            mask_file, TOMO_ID, pixel_size=10.8, target=TardisTarget.membrane
        )
        out = converter.cets_to_tardis(mask, self.out_dir, "tomo_a")
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.8, places=5)
        with mrcfile.open(mask_file, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), PIXEL_SIZE, places=5)

    def test_not_a_semantic_mask(self) -> None:
        converter = TardisSemanticSegmentation()
        mask = converter.tardis_to_cets(
            write_semantic_mrc(self.test_dir), TOMO_ID, target=TardisTarget.membrane
        )
        instance_mask = mask.model_copy(
            update={"segmentation_type": SegmentationType.instance.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_tardis(instance_mask, self.out_dir, "tomo_a")
