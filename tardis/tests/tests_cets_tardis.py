# -*- coding: utf-8 -*-
"""Tardis -> CETS, on the real Tardis runs of the test data (see ``tardis.tests``)."""

import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np
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
    ACTIN,
    DATA_DIR,
    DATASETS,
    FIRST_ROWS,
    MEMBRANES,
    MICROTUBULES,
    Dataset,
    copy_predictions,
    log_numbers,
    write_empty_csv,
    write_log,
)
from tardis.utils.discovery import find_tardis_predictions
from tardis.utils.utils import (
    parse_prediction_log,
    read_instances_csv,
    write_instances_yaml,
    write_semantic_yaml,
)

FEATURE_TYPES = {
    TardisTarget.actin: SegmentedFeatureType.actin_filament,
    TardisTarget.membrane: SegmentedFeatureType.membrane,
    TardisTarget.microtubule: SegmentedFeatureType.microtubule,
}


class CetsTardisBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_tardis"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def check_frame(
        self,
        annotation: SegmentationMask3D | PointSet3D | Spline3D,
        dataset: Dataset,
        pixel_size: float | None = None,
    ) -> None:
        """Coordinate systems named after the tomogram and the pixel size as scale"""
        cs_names = [cs.name for cs in annotation.coordinate_systems or []]
        self.assertEqual(
            cs_names, [f"{dataset.tomo_id}_array", f"{dataset.tomo_id}_physical"]
        )
        transformations = annotation.coordinate_transformations or []
        self.assertEqual(len(transformations), 1)
        self.assertEqual(transformations[0].name, "array_to_physical")
        self.assertEqual(transformations[0].input, f"{dataset.tomo_id}_array")
        self.assertEqual(transformations[0].output, f"{dataset.tomo_id}_physical")
        # The mrc headers store the pixel size as float32
        for value in transformations[0].scale:
            self.assertAlmostEqual(value, pixel_size or dataset.pixel_size, places=5)


class TestDataTest(TestCase):
    """The test data is the one the tests expect, and consistent"""

    def test_grids_and_pixel_sizes(self) -> None:
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                for mrc_file in (dataset.tomogram, dataset.semantic_mrc):
                    with mrcfile.open(mrc_file, header_only=True, permissive=True) as m:
                        header = m.header
                        self.assertEqual(
                            (int(header.nx), int(header.ny), int(header.nz)),
                            dataset.grid,
                        )
                        self.assertAlmostEqual(
                            float(m.voxel_size.x), dataset.pixel_size, places=5
                        )

    def test_semantic_masks_match_the_logs(self) -> None:
        """Binary masks with the number of voxels of the log"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                data = mrcfile.read(dataset.semantic_mrc)
                self.assertEqual(set(np.unique(data)), {0, 1})
                self.assertEqual(int(data.sum()), dataset.semantic_voxels)
                self.assertEqual(
                    log_numbers(dataset, "Semantic"), [dataset.semantic_voxels]
                )

    def test_instances_match_the_logs(self) -> None:
        """Number of instances and points of each csv; the log gives the instances of
        every csv written (raw and, for filaments, filtered)"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                self.assertEqual(parse_prediction_log(dataset.log), dataset.target)
                expected = [(dataset.n_instances, dataset.n_points)]
                if dataset.instances_filter_csv:
                    expected.append(
                        (dataset.n_filter_instances or 0, dataset.n_filter_points or 0)
                    )
                csv_files = [dataset.instances_csv, dataset.instances_filter_csv]
                found = []
                for csv_file in csv_files[: len(expected)]:
                    assert csv_file is not None
                    ids, _ = read_instances_csv(csv_file)
                    # Ids 0..n-1, grouped
                    self.assertEqual(sorted(set(ids)), list(range(len(set(ids)))))
                    self.assertEqual(ids, sorted(ids))
                    found.append((len(set(ids)), len(ids)))
                self.assertEqual(found, expected)
                self.assertEqual(
                    log_numbers(dataset, "Instance"), [n for n, _ in expected]
                )

    def test_instances_are_inside_the_tomograms(self) -> None:
        for dataset in DATASETS:
            for csv_file in [dataset.instances_csv, dataset.instances_filter_csv]:
                if csv_file is None:
                    continue
                with self.subTest(csv_file.name):
                    _, coords = read_instances_csv(csv_file)
                    voxels = np.array(coords) / dataset.pixel_size
                    self.assertTrue((voxels > -0.5).all())
                    self.assertTrue((voxels < np.array(dataset.grid) - 0.5).all())

    def test_find_tardis_predictions(self) -> None:
        """The three runs are found, with their files"""
        predictions = find_tardis_predictions(DATA_DIR)
        by_id = {p.tomo_id: p for p in predictions}
        self.assertEqual(len(predictions), len(DATASETS))
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                prediction = by_id[dataset.tomo_id]
                self.assertEqual(prediction.target, dataset.target)
                self.assertEqual(prediction.semantic_mrc, dataset.semantic_mrc)
                self.assertEqual(prediction.instances_csv, dataset.instances_csv)
                self.assertEqual(
                    prediction.instances_filter_csv, dataset.instances_filter_csv
                )
                # Filaments: the filtered instances, as in the yaml files
                self.assertEqual(
                    prediction.preferred_instances_csv, dataset.yaml_instances_csv
                )
                self.assertEqual(prediction.input_tomogram, dataset.tomogram)
                self.assertAlmostEqual(
                    prediction.get_pixel_size() or 0, dataset.pixel_size, places=5
                )


class CetsTardisPredictionLogTest(CetsTardisBaseTest):
    def test_targets_of_the_real_logs(self) -> None:
        for dataset in DATASETS:
            self.assertEqual(parse_prediction_log(dataset.log), dataset.target)

    def test_other_targets(self) -> None:
        log = write_log(self.test_dir, "General_filament")
        self.assertEqual(parse_prediction_log(log), TardisTarget.general_filament)
        # 2D targets are not supported
        with self.assertRaises(ValueError):
            parse_prediction_log(write_log(self.test_dir, "Membrane2D"))
        # No target
        log.write_text("Directory: emd_10439.mrc\nDevice: cuda:0\n")
        with self.assertRaises(ValueError):
            parse_prediction_log(log)


class CetsTardisSemanticTest(CetsTardisBaseTest):
    def test_real_semantic_masks(self) -> None:
        """Target from the log, pixel size and size from the header"""
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                mask = TardisSemanticSegmentation().tardis_to_cets(
                    dataset.semantic_mrc, dataset.tomo_id
                )
                self.assertEqual(
                    mask.annotation_type, AnnotationType.segmentation_mask_3D
                )
                self.assertEqual(mask.id, dataset.semantic_id)
                self.assertEqual(
                    mask.name,
                    f"Tardis {dataset.target.value} semantic segmentation for "
                    f"{dataset.tomo_id}",
                )
                self.assertEqual(mask.source_tomogram_id, dataset.tomo_id)
                self.assertEqual(mask.path, str(dataset.semantic_mrc.resolve()))
                self.assertEqual((mask.width, mask.height, mask.depth), dataset.grid)
                self.assertEqual(mask.segmentation_type, SegmentationType.semantic)
                self.assertEqual(
                    mask.annotated_object.feature_type, FEATURE_TYPES[dataset.target]
                )
                self.assertEqual(len(mask.labels), 1)
                self.assertEqual(mask.labels[0].label_value, 1)
                self.assertEqual(mask.labels[0].annotated_object, mask.annotated_object)
                self.assertIsNone(mask.derived_from_annotation_id)
                self.check_frame(mask, dataset)

    def test_explicit_target_pixel_size_and_id(self) -> None:
        mask = TardisSemanticSegmentation().tardis_to_cets(
            MEMBRANES.semantic_mrc,
            MEMBRANES.tomo_id,
            pixel_size=27.36,
            target=TardisTarget.actin,
            annotation_id="run_2",
        )
        self.assertEqual(mask.id, "run_2")
        self.assertEqual(
            mask.annotated_object.feature_type, SegmentedFeatureType.actin_filament
        )
        self.check_frame(mask, MEMBRANES, pixel_size=27.36)

    def test_without_log_or_target(self) -> None:
        predictions = copy_predictions(
            MEMBRANES, self.test_dir, csv_files=False, log=False
        )
        mask_file = predictions / MEMBRANES.semantic_mrc.name
        shutil.copyfile(MEMBRANES.semantic_mrc, mask_file)
        with self.assertRaises(FileNotFoundError):
            TardisSemanticSegmentation().tardis_to_cets(mask_file, MEMBRANES.tomo_id)
        mask = TardisSemanticSegmentation().tardis_to_cets(
            mask_file, MEMBRANES.tomo_id, target=TardisTarget.membrane
        )
        self.assertEqual(mask.id, MEMBRANES.semantic_id)

    def test_invalid_input(self) -> None:
        converter = TardisSemanticSegmentation()
        with self.assertRaises(ValueError):
            converter.tardis_to_cets(
                MEMBRANES.semantic_mrc, MEMBRANES.tomo_id, pixel_size=-1.0
            )
        with self.assertRaises(ValueError):  # Not an mrc file
            converter.tardis_to_cets(MEMBRANES.instances_csv, MEMBRANES.tomo_id)
        with self.assertRaises(FileNotFoundError):
            converter.tardis_to_cets(self.test_dir / "a.mrc", MEMBRANES.tomo_id)


class CetsTardisInstancesTest(CetsTardisBaseTest):
    def check_instances(
        self,
        instances: PointSet3D | Spline3D | None,
        dataset: Dataset,
        csv_file: Path,
    ) -> None:
        """The points of the csv, in its order, in voxels"""
        self.assertIsInstance(
            instances, Spline3D if dataset.is_filament else PointSet3D
        )
        assert instances is not None
        self.assertEqual(
            instances.annotation_type,
            AnnotationType.spline_3D
            if dataset.is_filament
            else AnnotationType.point_set_3D,
        )
        ids, coords = read_instances_csv(csv_file)
        self.assertEqual(instances.instance_ids, ids)
        np.testing.assert_allclose(
            np.array(instances.origin3D) * dataset.pixel_size, coords, rtol=1e-6
        )
        self.assertEqual(instances.source_tomogram_id, dataset.tomo_id)
        self.assertEqual(
            instances.annotated_object.feature_type, FEATURE_TYPES[dataset.target]
        )
        self.check_frame(instances, dataset)

    def test_real_instances(self) -> None:
        """Every csv: target from the log, pixel size from the semantic mask"""
        converter = TardisInstanceSegmentation()
        for dataset in DATASETS:
            with self.subTest(dataset.folder):
                instances = converter.tardis_to_cets(
                    dataset.instances_csv, dataset.tomo_id
                )
                self.check_instances(instances, dataset, dataset.instances_csv)
                assert instances is not None
                self.assertEqual(
                    instances.id,
                    f"tardis_{dataset.target.name}_instances_{dataset.tomo_id}",
                )
                self.assertEqual(
                    instances.name,
                    f"Tardis {dataset.target.value} instances for {dataset.tomo_id}",
                )
                self.assertEqual(len(set(instances.instance_ids)), dataset.n_instances)
                self.assertIsNone(instances.derived_from_annotation_id)

    def test_real_filtered_instances(self) -> None:
        """The filtered filaments get another id and name, to coexist in a region"""
        converter = TardisInstanceSegmentation()
        for dataset in (ACTIN, MICROTUBULES):
            with self.subTest(dataset.folder):
                csv_file = dataset.instances_filter_csv
                assert csv_file is not None
                self.assertTrue(converter.is_filtered(csv_file))
                self.assertFalse(converter.is_filtered(dataset.instances_csv))
                self.assertEqual(converter.get_tomo_name(csv_file), dataset.tomo_id)
                instances = converter.tardis_to_cets(
                    csv_file,
                    dataset.tomo_id,
                    semantic_annotation_id=dataset.semantic_id,
                )
                self.check_instances(instances, dataset, csv_file)
                assert instances is not None
                self.assertEqual(instances.id, dataset.instances_id)
                self.assertIn("(length-filtered)", instances.name)
                self.assertEqual(
                    len(set(instances.instance_ids)), dataset.n_filter_instances
                )
                self.assertEqual(
                    instances.derived_from_annotation_id, dataset.semantic_id
                )

    def test_first_points(self) -> None:
        """Values read from the files: the Å of the csv over the pixel size"""
        converter = TardisInstanceSegmentation()
        for csv_file, row in FIRST_ROWS.items():
            dataset = next(d for d in DATASETS if d.instances_csv == csv_file)
            instances = converter.tardis_to_cets(csv_file, dataset.tomo_id)
            assert instances is not None
            values = [float(v) for v in row.split(",")]
            self.assertEqual(instances.instance_ids[0], int(values[0]))
            np.testing.assert_allclose(
                instances.origin3D[0],
                np.array(values[1:]) / dataset.pixel_size,
                rtol=1e-6,
            )
        # Membrane points are voxels
        membranes = converter.tardis_to_cets(MEMBRANES.instances_csv, MEMBRANES.tomo_id)
        assert membranes is not None
        np.testing.assert_allclose(membranes.origin3D[0], [32.0, 176.0, 42.0])
        np.testing.assert_allclose(
            membranes.origin3D, np.round(membranes.origin3D), atol=1e-5
        )

    def test_explicit_target_and_id(self) -> None:
        instances = TardisInstanceSegmentation().tardis_to_cets(
            MEMBRANES.instances_csv,
            MEMBRANES.tomo_id,
            target=TardisTarget.general_filament,
            annotation_id="run_2",
        )
        self.assertIsInstance(instances, Spline3D)
        assert instances is not None
        self.assertEqual(instances.id, "run_2")
        self.assertEqual(
            instances.annotated_object.feature_type, SegmentedFeatureType.filament
        )

    def test_without_log_or_target(self) -> None:
        predictions = copy_predictions(MICROTUBULES, self.test_dir, log=False)
        csv_file = predictions / MICROTUBULES.instances_csv.name
        with self.assertRaises(FileNotFoundError):
            TardisInstanceSegmentation().tardis_to_cets(
                csv_file, MICROTUBULES.tomo_id, pixel_size=MICROTUBULES.pixel_size
            )

    def test_no_instances(self) -> None:
        """Tardis writes only the header"""
        predictions = copy_predictions(MEMBRANES, self.test_dir, csv_files=False)
        csv_file = write_empty_csv(predictions / MEMBRANES.instances_csv.name)
        instances = TardisInstanceSegmentation().tardis_to_cets(
            csv_file, MEMBRANES.tomo_id, pixel_size=MEMBRANES.pixel_size
        )
        self.assertIsNone(instances)

    def test_invalid_csv(self) -> None:
        csv_file = self.test_dir / "a_instances.csv"
        csv_file.write_text(MEMBRANES.instances_csv.read_text().replace("IDs", "Id", 1))
        with self.assertRaises(ValueError):
            TardisInstanceSegmentation().tardis_to_cets(
                csv_file, "a", pixel_size=1.0, target=TardisTarget.membrane
            )

    def test_yaml_files(self) -> None:
        """write_*_yaml: one object per file, named after the annotation id"""
        mask = TardisSemanticSegmentation().tardis_to_cets(
            MICROTUBULES.semantic_mrc, MICROTUBULES.tomo_id
        )
        instances = TardisInstanceSegmentation().tardis_to_cets(
            MICROTUBULES.instances_filter_csv or Path(),
            MICROTUBULES.tomo_id,
            semantic_annotation_id=mask.id,
        )
        assert instances is not None
        write_semantic_yaml(mask, self.test_dir)
        write_instances_yaml(instances, self.test_dir)
        self.assertEqual(
            sorted(p.name for p in self.test_dir.iterdir()),
            sorted(
                p.name
                for p in (MICROTUBULES.semantic_yaml, MICROTUBULES.instances_yaml)
            ),
        )
        with open(self.test_dir / MICROTUBULES.semantic_yaml.name) as f:
            self.assertEqual(SegmentationMask3D.model_validate(yaml.safe_load(f)), mask)
        with open(self.test_dir / MICROTUBULES.instances_yaml.name) as f:
            self.assertEqual(Spline3D.model_validate(yaml.safe_load(f)), instances)


class CetsTardisInstancesPixelSizeTest(CetsTardisBaseTest):
    """The sampling rate of the instances, which the csv does not record: the given one,
    else the one of the semantic mask header, else the one of the input tomogram"""

    @staticmethod
    def scale(instances: PointSet3D | Spline3D | None) -> float:
        assert instances is not None
        return instances.coordinate_transformations[0].scale[0]

    def convert(self, predictions: Path, pixel_size: float | None = None) -> float:
        return self.scale(
            TardisInstanceSegmentation().tardis_to_cets(
                predictions / MEMBRANES.instances_csv.name,
                MEMBRANES.tomo_id,
                pixel_size=pixel_size,
            )
        )

    def test_from_semantic_mask(self) -> None:
        predictions = copy_predictions(MEMBRANES, self.test_dir, semantic=True)
        self.assertAlmostEqual(self.convert(predictions), MEMBRANES.pixel_size, 5)

    def test_from_input_tomogram(self) -> None:
        predictions = copy_predictions(MEMBRANES, self.test_dir, tomogram=True)
        self.assertAlmostEqual(self.convert(predictions), MEMBRANES.pixel_size, 5)

    def test_semantic_mask_first(self) -> None:
        """A mask of another sampling rate than the tomogram (not with the real data)"""
        predictions = copy_predictions(MEMBRANES, self.test_dir, tomogram=True)
        mask = predictions / MEMBRANES.semantic_mrc.name
        shutil.copyfile(MEMBRANES.semantic_mrc, mask)
        with mrcfile.mmap(mask, mode="r+", permissive=True) as mrc:
            mrc.voxel_size = 27.36
        self.assertAlmostEqual(self.convert(predictions), 27.36, places=5)

    def test_given_overrides_headers(self) -> None:
        predictions = copy_predictions(
            MEMBRANES, self.test_dir, semantic=True, tomogram=True
        )
        self.assertEqual(self.convert(predictions, pixel_size=10.0), 10.0)

    def test_not_found(self) -> None:
        predictions = copy_predictions(MEMBRANES, self.test_dir)
        with self.assertRaises(ValueError):
            self.convert(predictions)

    def test_invalid_given(self) -> None:
        with self.assertRaises(ValueError):
            TardisInstanceSegmentation().tardis_to_cets(
                MEMBRANES.instances_csv, MEMBRANES.tomo_id, pixel_size=0.0
            )
