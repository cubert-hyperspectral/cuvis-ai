from enum import Enum


class OutputFormat(Enum):
    """ "
    Describes the output format that is returned by NumpyData, CuvisData and LiveCuvisData.
    The output of the dataloader is always in the same order: (hsi data cube, labels dictionary, meta data dictionary).
    If a format is selected which omits either meta data or label data, None is returned at that index of the tuple.
    Possible values and their respective return formats:
        - Full: (cube, label data, meta data)
        - BoundingBox: (cube, Bounding Boxes, None)
        - SegmentationMask: (cube, Segmentation Mask, None)
        - Metadata: (cube, None, meta data)
        - BoundingBoxWithMeta: (cube, meta data, Bounding Boxes)
        - SegmentationMaskWithMeta: (cube, meta data, Segmentation Mask)
        - CustomFilter: output_lambda((cube, meta data, label data))
    """

    Full = 0
    BoundingBox = 1
    SegmentationMask = 2
    Metadata = 3
    BoundingBoxWithMeta = 4
    SegmentationMaskWithMeta = 5
    CustomFilter = 99
