# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np

from tardis.constants import TardisTarget
from tardis.tests import write_prediction_log
from tardis.utils.discovery import find_tardis_predictions


class TardisDiscoveryTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_tardis_discovery"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _touch(self, rel_path: str) -> Path:
        path = self.test_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def _write_mrc(self, rel_path: str, voxel_size: float) -> Path:
        path = self.test_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.zeros((2, 3, 4), dtype=np.int8))
            mrc.voxel_size = voxel_size
        return path

    def test_find_predictions(self) -> None:
        """Every Predictions directory (with its prediction_log.txt) is read and its files
        grouped by tomogram. Tardis run on a directory writes several tomograms in it."""
        preds = self.test_dir / "run" / "Predictions"
        preds.mkdir(parents=True)
        write_prediction_log(preds, TardisTarget.actin.value)
        sem_a = self._touch("run/Predictions/TS_01_semantic.mrc")
        raw_a = self._touch("run/Predictions/TS_01_instances.csv")
        filt_a = self._touch("run/Predictions/TS_01_instances_filter.csv")
        raw_b = self._touch("run/Predictions/TS_02_instances.csv")
        # Ignored: files of other tools and outputs without a prediction_log.txt
        self._touch("run/project/tomograms/volume_1/models/mgeneral_1.omd")
        self._touch("run/tomograms.vll")
        self._touch("other/TS_09_instances.csv")
        predictions = find_tardis_predictions(self.test_dir)
        self.assertEqual([p.tomo_id for p in predictions], ["TS_01", "TS_02"])
        first, second = predictions
        self.assertEqual(first.target, TardisTarget.actin)
        self.assertEqual(
            (first.semantic_mrc, first.instances_csv, first.instances_filter_csv),
            (sem_a, raw_a, filt_a),
        )
        # Filaments: the length-filtered splines are preferred when written
        self.assertEqual(first.preferred_instances_csv, filt_a)
        self.assertEqual(second.preferred_instances_csv, raw_b)
        self.assertIsNone(second.semantic_mrc)

    def test_membrane_prefers_all_instances(self) -> None:
        preds = self.test_dir / "Predictions"
        preds.mkdir()
        write_prediction_log(preds, TardisTarget.membrane.value)
        raw = self._touch("Predictions/T_instances.csv")
        self._touch("Predictions/T_instances_filter.csv")
        (prediction,) = find_tardis_predictions(self.test_dir)
        self.assertEqual(prediction.preferred_instances_csv, raw)

    def test_pixel_size(self) -> None:
        """From the semantic mask (written with --correct_px), else the input tomogram next to
        the Predictions directory, else unknown"""
        preds = self.test_dir / "Predictions"
        preds.mkdir()
        write_prediction_log(preds, TardisTarget.actin.value)
        self._touch("Predictions/T_instances.csv")
        (prediction,) = find_tardis_predictions(self.test_dir)
        self.assertIsNone(prediction.input_tomogram)
        self.assertIsNone(prediction.get_pixel_size())
        tomogram = self._write_mrc("T.mrc", 13.5)
        self.assertEqual(prediction.input_tomogram, tomogram)
        self.assertAlmostEqual(prediction.get_pixel_size() or 0, 13.5, places=5)
        self._write_mrc("Predictions/T_semantic.mrc", 27.0)
        (prediction,) = find_tardis_predictions(self.test_dir)
        self.assertAlmostEqual(prediction.get_pixel_size() or 0, 27.0, places=5)
