import os

from cets_data_model.models.models import (
    SegmentationMask3D,
    SegmentationLabel,
    SegmentationType,
    AnnotationType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from tardis.constants import MRC_EXT, SEMANTIC_LABEL_VALUE, TardisTarget
from tardis.utils.utils import (
    validate_file,
    resolve_target,
    gen_annotated_object,
    gen_coordinate_systems,
    gen_array_to_physical,
)


class TardisSemanticSegmentation:
    def tardis_to_cets(
        self,
        semantic_mrc: os.PathLike,
        tomo_id: str,
        pixel_size: float | None = None,
        target: TardisTarget | None = None,
        annotation_id: str | None = None,
    ) -> SegmentationMask3D:
        """Converts a Tardis semantic segmentation (``<tomo>_semantic.mrc``) into CETS metadata.

        The semantic mask is a binary volume (1 = target, 0 = background) sampled on the same
        voxel grid as the segmented tomogram, so it is represented as a ``SegmentationMask3D``
        annotation (stored under ``Region.annotations``) that links back to the tomogram
        through ``source_tomogram_id``.

        :param semantic_mrc: path to the Tardis semantic mask.
        :type semantic_mrc: os.PathLike.

        :param tomo_id: identifier of the tomogram the mask was obtained from. It is used as
        ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of
        the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px) of the mask. If given, it is used; otherwise it
        is read from the mrc header, where Tardis writes it (--correct_px).
        :type pixel_size: float, optional. Defaults to None.

        :param target: segmented target (membrane, microtubule...). When omitted, it is read
        from the prediction_log.txt that Tardis writes next to the mask.
        :type target: TardisTarget, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations, so it has to be provided when several Tardis runs on the same
        tomogram are converted. Defaults to ``get_annotation_id(tomo_id, target)``.
        :type annotation_id: str, optional. Defaults to None.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        mask_path = validate_file(semantic_mrc, expected_ext=MRC_EXT)
        target = resolve_target(mask_path, target)
        mrc_info = get_mrc_info(mask_path)
        voxel_size = pixel_size or mrc_info.apix_x
        if not voxel_size:
            raise ValueError(
                f"No pixel size was provided and {mask_path} does not contain it in its "
                f"header."
            )
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        array_to_physical = gen_array_to_physical(
            voxel_size, array_cs.name, physical_cs.name, ndim=3
        )
        annotated_object = gen_annotated_object(target)
        return SegmentationMask3D(
            id=annotation_id or self.get_annotation_id(tomo_id, target),
            name=f"Tardis {target.value} semantic segmentation for {tomo_id}",
            annotation_type=AnnotationType.segmentation_mask_3D,
            source_tomogram_id=tomo_id,
            annotated_object=annotated_object,
            # Binary mask: the target voxels are 1 and the background 0
            segmentation_type=SegmentationType.semantic,
            labels=[
                SegmentationLabel(
                    label_value=SEMANTIC_LABEL_VALUE,
                    annotated_object=annotated_object,
                )
            ],
            path=str(mask_path),
            width=mrc_info.size_x,
            height=mrc_info.size_y,
            depth=mrc_info.size_z,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[array_to_physical],
        )

    @staticmethod
    def get_annotation_id(tomo_id: str, target: TardisTarget) -> str:
        """Id of the semantic mask annotation, so that the instance segmentation obtained in
        the same Tardis run can refer to it (derived_from_annotation_id)."""
        return f"tardis_{target.name}_semantic_{tomo_id}"
