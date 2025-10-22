import datetime
import json
import os.path as osp

import cv2 as cv
import numpy as np
import pycocotools


def get_shape_without_batch(array: np.ndarray, ignore=()):
    if isinstance(ignore, int):
        ignore = (ignore,)

    match array:
        case np.ndarray():
            ndim = array.ndim
        case tuple():
            ndim = len(array)
        case list():
            ndim = len(array)
        case _:
            raise ValueError("Unknown input type.")

    if ndim != 3 and ndim != 4:
        raise ValueError("Input array must be 3D or 4D.")

    match array:
        case np.ndarray():
            shape = array.shape if ndim == 3 else array.shape[1:]
        case tuple():
            shape = array if ndim == 3 else array[1:]
        case list():
            shape = array if ndim == 3 else array[1:]
        case _:
            raise ValueError("Unknown input type.")

    return tuple([-1 if i in ignore else shape[i] for i in [0, 1, 2]])


def check_array_shape(array: np.ndarray | tuple[int, int, int], wanted_shape: tuple[int, int, int]):
    match array:
        case np.ndarray():
            array_shape = array.shape
        case tuple():
            array_shape = array
        case list():
            array_shape = array
        case _:
            raise ValueError("Unknown input type.")

    ret = True
    for v, w in zip(array_shape, wanted_shape):
        ret &= w == -1 or v == w
    return ret


def flatten_spatial(array: np.ndarray):
    if array.ndim == 3:
        # Array is of shape [width, height, channels]
        return array.reshape(-1, array.shape[-1])
    elif array.ndim > 3:
        return array.reshape((*array.shape[:-3], -1, array.shape[-1]))
    else:
        raise ValueError("Input array must be 3D or higher.")


def flatten_batch_and_spatial(array: np.ndarray):
    if array.ndim == 3:
        # Array is of shape [width, height, channels]
        return array.reshape(-1, array.shape[2])
    elif array.ndim == 4:
        # Array is of shape [batch, width, height, channels]
        return array.reshape(-1, array.shape[3])
    else:
        raise ValueError("Input array must be 3D or 4D.")


def unflatten_batch_and_spatial(array: np.ndarray, orig_shape):
    if array.ndim != 2 and array.ndim != 1:
        raise ValueError("Input array must be 2D or 1D.")
    if array.shape[0] != np.prod(orig_shape[:-1]):
        raise ValueError("Input array and orig shape do not add up.")
    return array.reshape(*orig_shape[:-1], -1)


def unflatten_spatial(array: np.ndarray, orig_shape):
    if array.ndim != 3 and array.ndim != 2:
        raise ValueError("Input array must be 2D or 3D.")
    if array.shape[0] != np.prod(orig_shape[:-1]):
        raise ValueError("Input array and orig shape do not add up.")
    return array.reshape(*orig_shape[:-1], -1)


def flatten_labels(array: np.ndarray):
    if array.ndim == 2:
        # Array is of shape [width, height]
        return array.reshape(-1)
    elif array.ndim == 3:
        # Array is of shape [batch, width, height]
        return array.reshape(array.shape[0], -1)
    else:
        raise ValueError("Input array must be 2D or 3D.")


def unflatten_labels(array: np.ndarray, orig_shape):
    if array.ndim != 1 and array.ndim != 2:
        raise ValueError("Input array must be 1D or 2D.")
    return array.reshape(orig_shape)


def flatten_batch_and_labels(array: np.ndarray):
    if array.ndim == 2:
        # Array is of shape [width, height]
        return array.reshape(-1)
    elif array.ndim == 3:
        # Array is of shape [batch, width, height]
        return array.reshape(-1)
    else:
        raise ValueError("Input array must be 2D or 3D.")


def unflatten_batch_and_labels(array: np.ndarray, orig_shape):
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    return array.reshape(orig_shape)


def binary_mask_to_rle(binary_mask):
    """
    converts a binary mask to RLE shamelessly copied from https://stackoverflow.com/a/76990451
    :param binary_mask:
    :return:
    """
    rle = {"counts": [], "size": list(binary_mask.shape)}
    # fortran order needed to correctly convert to RLE
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    # +1 to "convert" from indices to length
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def gen_coco_labels(
    mask: np.ndarray,
    label_names: list,
    output_dir: str,
    name: str,
    img_name: str = None,
    single_object_per_label: bool = False,
):
    """
    generating coco labels from numpy image and mask, occluded objects can not be labeled correctly at the moment
    :param mask: mask should be a mask containing zeros for background and integers for labels
    :param label_names: list of labels
    :param output_dir: path where to save the json file, if the output file already exists, the generated labels will be appended
    :param name: name of the json file
    :param img: image for which these labels apply
    :param single_object_per_label: clarify if every object has its own label or if multiple objects share the same label. default = False

    :return:
    """

    now = datetime.datetime.now()
    out_ann_file = osp.realpath(osp.join(output_dir, name + "_annotations.json"))
    # if label file exists load and extend the file, create new data structure otherwise
    if osp.exists(out_ann_file):
        data = json.load(open(out_ann_file))

    else:
        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )
        for label_id, label in enumerate(label_names):
            data["categories"].append(
                dict(
                    supercategory=None,
                    id=label_id,
                    name=label,
                )
            )

    image_id = len(data["images"]) + 1

    data["images"].append(
        dict(
            license=0,
            url=None,
            file_name=img_name if img_name else str(image_id),
            height=mask.shape[0],
            width=mask.shape[1],
            date_captured=None,
            id=image_id,
        )
    )
    for label_id, label in enumerate(label_names, start=1):
        label_mask = mask.copy()
        # remove unwanted labels
        label_mask[label_mask != label_id] = 0
        label_mask[label_mask == label_id] = 1
        label_mask = label_mask.astype(np.uint8)

        pts = []
        # if all areas of one label describe the same object put them into one annotation with iscrowd 1 and RLE compressed
        if not single_object_per_label:
            segmentation = binary_mask_to_rle(label_mask)
            # make mask Fortran contiguous as pycocotools expects it that way
            label_mask = np.asfortranarray(label_mask.astype(np.uint8))
            label_mask = pycocotools.mask.encode(label_mask)
            area = float(pycocotools.mask.area(label_mask))
            bbox = pycocotools.mask.toBbox(label_mask).flatten().tolist()
            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=label_id,
                    segmentation=segmentation,
                    area=area,
                    bbox=bbox,
                    iscrowd=1,
                )
            )
        # if every contour is a different object but may be of the same "class" save every contour as an annotation and
        else:
            contours, _ = cv.findContours(label_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for el in contours:
                contour = el.squeeze()
                es = el.shape
                pts.append(el.reshape((es[0], es[-1])))
                # make contour Fortran contiguous as pycocotools expects it that way
                contour = np.asfortranarray(contour.astype(np.uint8))
                contour = pycocotools.mask.encode(contour)
                area = float(pycocotools.mask.area(contour))
                bbox = pycocotools.mask.toBbox(contour).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=label_id,
                        segmentation=contour,
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

    with open(out_ann_file, "w") as f:
        json.dump(data, f)
