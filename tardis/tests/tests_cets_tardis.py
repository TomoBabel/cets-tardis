# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from cets_data_model.models.models import (
    AnnotationType,
    PointSet3D,
    SegmentationMask3D,
    SegmentationType,
    SegmentedFeatureType,
    Spline3D,
)
from tardis.constants import TardisTarget
from tardis.converters.instances import TardisInstanceSegmentation
from tardis.converters.semantic import TardisSemanticSegmentation
from tardis.tests import (
    REAL_ACTIN_PIXEL_SIZE,
    REAL_ACTIN_PREDICTIONS,
    REAL_ACTIN_TOMO_ID,
    REAL_MEMBRANE_PIXEL_SIZE,
    REAL_MEMBRANE_PREDICTIONS,
    REAL_MEMBRANE_TOMO_ID,
    INSTANCES_ROWS,
    MASK_DIMS,
    PIXEL_SIZE,
    TOMO_ID,
    write_instances_csv,
    write_prediction_log,
    write_semantic_mrc,
)
from tardis.utils.utils import parse_prediction_log, write_instances_yaml

CETS_TARDIS = "cets_tardis"


class CetsTardisBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix=CETS_TARDIS))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _check_frame(
        self, annotation: SegmentationMask3D | PointSet3D | Spline3D
    ) -> None:
        cs_names = [cs.name for cs in annotation.coordinate_systems or []]
        self.assertEqual(cs_names, [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"])
        transformations = annotation.coordinate_transformations or []
        self.assertEqual(len(transformations), 1)
        self.assertEqual(transformations[0].name, "array_to_physical")
        # The mrc header stores the pixel size as float32
        for value in transformations[0].scale:
            self.assertAlmostEqual(value, PIXEL_SIZE, places=5)


class CetsTardisPredictionLogTest(CetsTardisBaseTest):
    def test_parse_target(self) -> None:
        write_prediction_log(self.test_dir, TardisTarget.microtubule.value)
        target = parse_prediction_log(self.test_dir / "prediction_log.txt")
        self.assertEqual(target, TardisTarget.microtubule)

    def test_2d_target_not_supported(self) -> None:
        write_prediction_log(self.test_dir, "Membrane2D")
        with self.assertRaises(ValueError):
            parse_prediction_log(self.test_dir / "prediction_log.txt")


class CetsTardisSemanticTest(CetsTardisBaseTest):
    def test_semantic_to_cets(self) -> None:
        write_prediction_log(self.test_dir, TardisTarget.membrane.value)
        mrc_file = write_semantic_mrc(self.test_dir)
        mask = TardisSemanticSegmentation().tardis_to_cets(mrc_file, TOMO_ID)
        self.assertEqual(mask.annotation_type, AnnotationType.segmentation_mask_3D)
        self.assertEqual(mask.id, f"tardis_membrane_semantic_{TOMO_ID}")
        self.assertEqual(mask.source_tomogram_id, TOMO_ID)
        self.assertEqual(mask.path, str(mrc_file.resolve()))
        self.assertEqual((mask.width, mask.height, mask.depth), MASK_DIMS)
        self.assertEqual(mask.segmentation_type, SegmentationType.semantic)
        self.assertEqual(
            mask.annotated_object.feature_type, SegmentedFeatureType.membrane
        )
        self.assertEqual(len(mask.labels), 1)
        self.assertEqual(mask.labels[0].label_value, 1)
        self.assertEqual(mask.labels[0].annotated_object, mask.annotated_object)
        self._check_frame(mask)

    def test_semantic_without_log_or_target(self) -> None:
        mrc_file = write_semantic_mrc(self.test_dir)
        with self.assertRaises(FileNotFoundError):
            TardisSemanticSegmentation().tardis_to_cets(mrc_file, TOMO_ID)

    def test_explicit_target_and_pixel_size(self) -> None:
        mrc_file = write_semantic_mrc(self.test_dir)
        mask = TardisSemanticSegmentation().tardis_to_cets(
            mrc_file, TOMO_ID, pixel_size=PIXEL_SIZE, target=TardisTarget.actin
        )
        self.assertEqual(mask.id, f"tardis_actin_semantic_{TOMO_ID}")
        self._check_frame(mask)


class CetsTardisInstancesTest(CetsTardisBaseTest):
    def _check_instances(
        self, instances: PointSet3D | Spline3D | None, expected_type: type
    ) -> None:
        self.assertIsInstance(instances, expected_type)
        assert instances is not None
        self.assertEqual(instances.source_tomogram_id, TOMO_ID)
        # All the instances in a single annotation, keeping the csv row order
        self.assertEqual(
            instances.instance_ids, [int(row[0]) for row in INSTANCES_ROWS]
        )
        expected = [[c / PIXEL_SIZE for c in row[1:]] for row in INSTANCES_ROWS]
        self.assertEqual(len(instances.origin3D or []), len(expected))
        for coord, expected_coord in zip(instances.origin3D or [], expected):
            for value, expected_value in zip(coord, expected_coord):
                self.assertAlmostEqual(value, expected_value)
        self._check_frame(instances)

    def test_filament_instances_to_splines(self) -> None:
        write_prediction_log(self.test_dir, TardisTarget.microtubule.value)
        csv_file = write_instances_csv(self.test_dir)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE
        )
        self._check_instances(instances, Spline3D)
        assert instances is not None
        self.assertEqual(instances.id, f"tardis_microtubule_instances_{TOMO_ID}")
        self.assertEqual(
            instances.annotated_object.feature_type, SegmentedFeatureType.microtubule
        )
        self.assertIsNone(instances.derived_from_annotation_id)

    def test_membrane_instances_to_point_sets(self) -> None:
        write_prediction_log(self.test_dir, TardisTarget.membrane.value)
        csv_file = write_instances_csv(self.test_dir)
        semantic_id = TardisSemanticSegmentation.get_annotation_id(
            TOMO_ID, TardisTarget.membrane
        )
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE, semantic_annotation_id=semantic_id
        )
        self._check_instances(instances, PointSet3D)
        assert instances is not None
        self.assertEqual(
            instances.annotated_object.feature_type, SegmentedFeatureType.membrane
        )
        self.assertEqual(instances.derived_from_annotation_id, semantic_id)

    def test_no_instances(self) -> None:
        csv_file = write_instances_csv(self.test_dir, rows=[])
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE, target=TardisTarget.actin
        )
        self.assertIsNone(instances)

    def test_instances_yaml(self) -> None:
        csv_file = write_instances_csv(self.test_dir)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, PIXEL_SIZE, target=TardisTarget.general_filament
        )
        assert instances is not None
        self.assertEqual(
            instances.annotated_object.feature_type, SegmentedFeatureType.filament
        )
        write_instances_yaml(instances, self.test_dir)
        with open(self.test_dir / f"{instances.id}_tardis_to_cets.yaml") as f:
            reloaded = Spline3D.model_validate(yaml.safe_load(f))
        self.assertEqual(reloaded, instances)

    def test_invalid_pixel_size(self) -> None:
        csv_file = write_instances_csv(self.test_dir)
        with self.assertRaises(ValueError):
            TardisInstanceSegmentation().tardis_to_cets(
                csv_file, TOMO_ID, 0.0, target=TardisTarget.actin
            )

    def test_annotation_id_override(self) -> None:
        csv_file = write_instances_csv(self.test_dir)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file,
            TOMO_ID,
            PIXEL_SIZE,
            target=TardisTarget.actin,
            annotation_id="run_2",
        )
        assert instances is not None
        self.assertEqual(instances.id, "run_2")


class CetsTardisInstancesPixelSizeTest(CetsTardisBaseTest):
    """The sampling rate of the instances: used if given, else read from the semantic mask or
    the input tomogram headers"""

    def _scale(self, instances: PointSet3D | Spline3D | None) -> float:
        assert instances is not None
        return instances.coordinate_transformations[0].scale[0]

    def test_from_semantic_mask(self) -> None:
        write_semantic_mrc(self.test_dir)  # <tomo>_semantic.mrc, header: PIXEL_SIZE
        csv_file = write_instances_csv(self.test_dir)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, target=TardisTarget.actin
        )
        self.assertAlmostEqual(self._scale(instances), PIXEL_SIZE, places=5)

    def test_from_input_tomogram(self) -> None:
        predictions_dir = self.test_dir / "Predictions"
        predictions_dir.mkdir()
        csv_file = write_instances_csv(predictions_dir)
        tomogram = write_semantic_mrc(self.test_dir).rename(
            self.test_dir / f"{TOMO_ID}.mrc"
        )
        self.assertTrue(tomogram.is_file())
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, target=TardisTarget.actin
        )
        self.assertAlmostEqual(self._scale(instances), PIXEL_SIZE, places=5)

    def test_given_overrides_headers(self) -> None:
        write_semantic_mrc(self.test_dir)
        csv_file = write_instances_csv(self.test_dir)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, TOMO_ID, pixel_size=10.0, target=TardisTarget.actin
        )
        self.assertEqual(self._scale(instances), 10.0)

    def test_not_found(self) -> None:
        csv_file = write_instances_csv(self.test_dir)
        with self.assertRaises(ValueError):
            TardisInstanceSegmentation().tardis_to_cets(
                csv_file, TOMO_ID, target=TardisTarget.actin
            )

    def test_invalid_given(self) -> None:
        mrc_file = write_semantic_mrc(self.test_dir)
        with self.assertRaises(ValueError):
            TardisSemanticSegmentation().tardis_to_cets(
                mrc_file, TOMO_ID, pixel_size=-1.0, target=TardisTarget.actin
            )


class CetsTardisRealOutputsTest(CetsTardisBaseTest):
    """Excerpts of real Tardis outputs: float instance ids ("0.0"), the target read from the
    real prediction_log.txt and the raw and length-filtered csv files of filaments."""

    def test_real_membrane_instances(self) -> None:
        csv_file = REAL_MEMBRANE_PREDICTIONS / f"{REAL_MEMBRANE_TOMO_ID}_instances.csv"
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, REAL_MEMBRANE_TOMO_ID, REAL_MEMBRANE_PIXEL_SIZE
        )
        self.assertIsInstance(instances, PointSet3D)
        assert instances is not None
        self.assertEqual(
            instances.id, f"tardis_membrane_instances_{REAL_MEMBRANE_TOMO_ID}"
        )
        self.assertEqual(instances.instance_ids, [0] * 5 + [1] * 5)
        # Tardis membrane coordinates are voxel indices * pixel size
        for value, expected in zip(instances.origin3D[0], [447.0, 58.0, 87.0]):
            self.assertAlmostEqual(value, expected)
        for coord in instances.origin3D:
            for value in coord:
                self.assertAlmostEqual(value, round(value))

    def test_real_actin_raw_and_filtered_instances(self) -> None:
        converter = TardisInstanceSegmentation()
        raw = converter.tardis_to_cets(
            REAL_ACTIN_PREDICTIONS / f"{REAL_ACTIN_TOMO_ID}_instances.csv",
            REAL_ACTIN_TOMO_ID,
            REAL_ACTIN_PIXEL_SIZE,
        )
        filtered = converter.tardis_to_cets(
            REAL_ACTIN_PREDICTIONS / f"{REAL_ACTIN_TOMO_ID}_instances_filter.csv",
            REAL_ACTIN_TOMO_ID,
            REAL_ACTIN_PIXEL_SIZE,
        )
        assert raw is not None and filtered is not None
        for instances in (raw, filtered):
            self.assertIsInstance(instances, Spline3D)
            self.assertEqual(
                instances.annotated_object.feature_type,
                SegmentedFeatureType.actin_filament,
            )
        # Both files can be converted into the same region, so they need different ids
        self.assertEqual(raw.id, f"tardis_actin_instances_{REAL_ACTIN_TOMO_ID}")
        self.assertEqual(
            filtered.id, f"tardis_actin_instances_filtered_{REAL_ACTIN_TOMO_ID}"
        )
        self.assertIn("length-filtered", filtered.name)
        self.assertEqual(raw.instance_ids, [0] * 4 + [1] * 5)
        self.assertEqual(filtered.instance_ids, [0] * 3 + [1] * 2)
