import os
from pathlib import Path

from cets_data_model.models.models import (
    PointSet3D,
    Spline3D,
    AnnotationType,
)
from tardis.constants import (
    CSV_EXT,
    FILAMENT_TARGETS,
    INSTANCES_FILTER_SUFFIX,
    INSTANCES_SUFFIX,
    TardisTarget,
)
from tardis.utils.discovery import find_pixel_size
from tardis.utils.utils import (
    validate_file,
    resolve_target,
    read_instances_csv,
    gen_annotated_object,
    gen_coordinate_systems,
    gen_array_to_physical,
)


class TardisInstanceSegmentation:
    def tardis_to_cets(
        self,
        instances_csv: os.PathLike,
        tomo_id: str,
        pixel_size: float | None = None,
        target: TardisTarget | None = None,
        semantic_annotation_id: str | None = None,
        annotation_id: str | None = None,
    ) -> PointSet3D | Spline3D | None:
        """Converts a Tardis instance segmentation (``<tomo>_instances.csv`` or, for filaments,
        ``<tomo>_instances_filter.csv``) into CETS metadata.

        Each row of the csv is a point [instance id, x, y, z]. All the instances of the
        tomogram are converted into a single annotation (stored under ``Region.annotations``)
        whose ``instance_ids`` keep the instance each point belongs to:

        * filaments (microtubules, actin, general filaments) -> a ``Spline3D`` holding a set of
          splines: the points sharing an instance id, in order, form one filament;
        * membranes -> a ``PointSet3D`` (unordered surface point cloud of each membrane).

        :param instances_csv: path to the Tardis instances csv file.
        :type instances_csv: os.PathLike.

        :param tomo_id: identifier of the tomogram the instances were obtained from. It is used
        as ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of
        the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px) of the segmented tomogram, i.e. the one used by
        Tardis (--correct_px). Tardis writes the coordinates in Å, but the csv does not record
        it, so when omitted it is read from the header of the semantic mask of the tomogram
        (<tomo>_semantic.mrc, next to the csv) or, else, of the input tomogram (next to the
        Predictions directory). A ValueError is raised if it cannot be found.
        :type pixel_size: float, optional. Defaults to None.

        :param target: segmented target (membrane, microtubule...). When omitted, it is read
        from the prediction_log.txt that Tardis writes next to the csv.
        :type target: TardisTarget, optional. Defaults to None.

        :param semantic_annotation_id: id of the semantic mask annotation obtained in the same
        Tardis run (see ``TardisSemanticSegmentation.get_annotation_id``), from which the
        instances were computed. It is stored as ``derived_from_annotation_id``.
        :type semantic_annotation_id: str, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations, so it has to be provided when several Tardis runs on the same
        tomogram are converted. When omitted, it is generated from the target, the tomogram id
        and whether the instances are the length-filtered ones (see ``get_annotation_id``).
        :type annotation_id: str, optional. Defaults to None.

        :returns: the instances annotation, or None if Tardis did not find any instance.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        csv_path = validate_file(instances_csv, expected_ext=CSV_EXT)
        if pixel_size is None:
            pixel_size = find_pixel_size(csv_path.parent, self.get_tomo_name(csv_path))
            if pixel_size is None:
                raise ValueError(
                    f"No pixel size was provided for {csv_path} and it could not be read "
                    f"from the semantic mask or the input tomogram headers."
                )
        target = resolve_target(csv_path, target)
        instance_ids, coords_angst = read_instances_csv(csv_path)
        if not instance_ids:
            return None

        # Tardis coordinates are voxel indices * pixel size, referred to the bottom left
        # corner of the tomogram, so dividing by the pixel size gives the position in the
        # tomogram array (voxel) frame without any offset.
        origin_3d = [[c / pixel_size for c in coord] for coord in coords_angst]
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        filtered = self.is_filtered(csv_path)
        name = (
            f"Tardis {target.value} instances{' (length-filtered)' if filtered else ''} "
            f"for {tomo_id}"
        )
        common_fields = dict(
            id=annotation_id or self.get_annotation_id(tomo_id, target, filtered),
            name=name,
            source_tomogram_id=tomo_id,
            annotated_object=gen_annotated_object(target),
            derived_from_annotation_id=semantic_annotation_id,
            origin3D=origin_3d,
            instance_ids=instance_ids,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[
                gen_array_to_physical(
                    pixel_size, array_cs.name, physical_cs.name, ndim=3
                )
            ],
        )
        if target in FILAMENT_TARGETS:
            return Spline3D(annotation_type=AnnotationType.spline_3D, **common_fields)
        return PointSet3D(annotation_type=AnnotationType.point_set_3D, **common_fields)

    @staticmethod
    def get_tomo_name(instances_csv: os.PathLike) -> str:
        """Name of the input tomogram in the Tardis file names (<tomo>_instances.csv or
        <tomo>_instances_filter.csv)."""
        name = Path(instances_csv).name
        for suffix in (INSTANCES_FILTER_SUFFIX, INSTANCES_SUFFIX):
            if name.endswith(f"{suffix}{CSV_EXT}"):
                return name[: -len(f"{suffix}{CSV_EXT}")]
        return Path(name).stem

    @staticmethod
    def is_filtered(instances_csv: os.PathLike) -> bool:
        """Whether the csv contains the length-filtered filaments
        (``<tomo>_instances_filter.csv``) instead of all the predicted instances."""
        return str(instances_csv).endswith(f"{INSTANCES_FILTER_SUFFIX}{CSV_EXT}")

    @staticmethod
    def get_annotation_id(tomo_id: str, target: TardisTarget, filtered: bool) -> str:
        """Default id of the instances annotation. Tardis writes both the raw and the
        length-filtered filaments, so they get different ids to coexist in the same region."""
        return (
            f"tardis_{target.name}_instances{'_filtered' if filtered else ''}_{tomo_id}"
        )
