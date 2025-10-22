import copy
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path

import cuvis
import numpy as np
import torch
from torchvision import tv_tensors

from cuvis_ai.data.coco_labels import COCOData
from cuvis_ai.data.metadata import get_meta_from_mesu, get_meta_from_session
from cuvis_ai.data.numpy_dataset import NumpyDataset
from cuvis_ai.data.output_format import OutputFormat
from cuvis_ai.tv_transforms import WavelengthList

debug_enabled = True


EXTENSION_SESSION = ".cu3s"
EXTENSION_LEGACY = ".cu3"


@lru_cache
def get_session_cube(path, idx, proc_mode, to_dtype: np.dtype):
    sess = cuvis.SessionFile(str(path))
    mesu = sess[idx]
    need_reprocess = bool(proc_mode is None)

    if mesu.cube is None:
        need_reprocess = True

    if need_reprocess and proc_mode is not None:
        pc = cuvis.ProcessingContext(sess)
        pc.processing_mode = proc_mode
        mesu = pc.apply(mesu)

    if mesu.cube is None:
        raise ValueError(f"Could not load Cube idx={idx} from SessionFile {path}.")  # nopep8

    cube_array = mesu.cube.array

    if cube_array.dtype != to_dtype:
        cube_array = cube_array.astype(to_dtype)

    cube_tensor = torch.from_numpy(cube_array)

    # Add batch/channel dimensions until we have (N, C, H, W)
    while cube_tensor.ndim < 4:
        cube_tensor = cube_tensor.unsqueeze(0)

    # Wrap as torchvision image
    cube = tv_tensors.Image(cube_tensor)

    return cube.to(memory_format=torch.channels_last)


@lru_cache
def get_legacy_cube(path, proc_mode, to_dtype: np.dtype):
    mesu = cuvis.Measurement.load(path)
    need_reprocess = bool(proc_mode is None)
    try:
        cube = mesu.data["cube"].array
    except KeyError:
        need_reprocess = True

    if need_reprocess:
        pc = cuvis.ProcessingContext(mesu)
        if proc_mode is not None:
            pc.processing_mode = proc_mode
        mesu = pc.apply(mesu)

    cube = mesu.data["cube"].array

    if cube.dtype != to_dtype:
        cube = cube.astype(to_dtype)
    cube = tv_tensors.Image(cube)
    while len(cube.shape) < 4:
        cube = cube.unsqueeze(0)


@lru_cache
def get_session_reference(path, reftype, to_dtype: np.dtype):
    try:
        cube = cuvis.SessionFile(path).get_reference(0, reftype).data["cube"].array
    except KeyError:
        sess = cuvis.SessionFile(path)
        mesu = sess.get_reference(0, reftype)
        pc = cuvis.ProcessingContext(sess)
        pc.processing_mode = cuvis.ProcessingMode.Raw
        mesu = pc.apply(mesu)
        cube = mesu.data["cube"].array

    if cube.dtype != to_dtype:
        cube = cube.astype(to_dtype)
    cube = tv_tensors.Image(cube)
    while len(cube.shape) < 4:
        cube = cube.unsqueeze(0)
    return cube.to(memory_format=torch.channels_last)


class CuvisDataset(NumpyDataset):
    """Representation for a set of Cuvis data cubes, their meta-data and labels.

    See :class:`NumpyData` for more details.


    Parameters
    ----------
    root : str, optional
        The absolute or relative path to the directory containing the HSI data.
    transforms : callable, optional
        A function/transforms that takes in an image and a label and returns the transformed versions of both.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    output_format : OutputFormat
        Enum value that controls the output format of the dataset. See :class:`OutputFormat`
    output_lambda : callable, optional
        Only used when :attr:`output_format` is set to `CustomFilter`. Before returning data, the full output of the dataset is passed through this function to allow for custom filtering.

    Notes
    -----
    :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.

    If :attr:`root` is not passed in the constructor, the :py:meth:`~CuvisDataset.initialize` or :py:meth:`~CuvisDataset.load` method has to be called with a root path before the dataset can be used.
    """

    def __init__(
        self,
        root: str | None = None,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Callable | None = None,
        force_proc_mode: cuvis.ProcessingMode | None = None,
    ):
        self.processing_mode = force_proc_mode
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            output_format=output_format,
            output_lambda=output_lambda,
        )

    def _load_directory(self, dir_path: str):
        dir_path = Path(dir_path)
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset_session = dir_path.glob(f"**/*{EXTENSION_SESSION}")

        fileset_legacy = dir_path.glob(f"**/*{EXTENSION_LEGACY}")

        for cur_path in fileset_session:
            self._load_session_file(cur_path)
        for cur_path in fileset_legacy:
            self._load_legacy_file(cur_path)

    def _load_session_file(self, filepath: Path):
        if debug_enabled:
            print("Found file:", filepath)

        labelpath = filepath.with_suffix(".json")
        has_labels = False  # labelpath.exists()

        crt_session = cuvis.SessionFile(str(filepath))

        if self.processing_mode is None:
            tmp_mesu = crt_session[0]
            self.processing_mode = tmp_mesu.processing_mode

        cube_count = len(crt_session)
        if debug_enabled:
            print("Session file has", cube_count, "cubes")

        # sess_meta = {}  # metadataInit(filepath, self.fileset_metadata)
        sess_meta = get_meta_from_session(crt_session, filepath)

        canvas_size = (sess_meta.shape[0], sess_meta.shape[1])

        if has_labels:
            coco = COCOData.from_path(labelpath)

        for idx in range(cube_count):
            cube_path = f"{filepath}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(partial(get_session_cube, str(filepath), idx, self.processing_mode))

            meta = copy.deepcopy(sess_meta)

            for k, v in meta.references.items():
                if not isinstance(v, str):
                    continue
                if Path(v).suffix == EXTENSION_SESSION and v == str(filepath):
                    meta.references[k] = partial(get_session_reference, str(v), k)
                if Path(v).suffix == EXTENSION_SESSION:
                    meta.references[k] = partial(
                        get_session_cube, str(v), 0, cuvis.ProcessingMode.Raw
                    )
                elif Path(v).suffix == EXTENSION_LEGACY:
                    meta.references[k] = partial(get_legacy_cube, str(v), cuvis.ProcessingMode.Raw)

            self.metas.append(meta)

            l = {}
            if has_labels:
                anns = coco.annotations.where(image_id=idx)[0]
                l = anns.to_torchvision(canvas_size)
                l["wavelength"] = WavelengthList(coco.images[idx].wavelength)
            self.labels.append(l)

    def _load_legacy_file(self, filepath: Path):
        if debug_enabled:
            print("Found file:", filepath)
        self.paths.append(filepath)
        labelpath = filepath.with_suffix(".json")
        has_labels = False  # labelpath.exists()

        mesu = cuvis.Measurement(filepath)

        meta = get_meta_from_mesu(mesu)

        canvas_size = (meta.shape[0], meta.shape[1])

        l = None
        if has_labels:
            coco = COCOData.from_path(labelpath)
            anns = coco.annotations.where(image_id=coco.image_ids[0])[0]
            l = anns.to_torchvision(canvas_size)
            l["wavelength"] = WavelengthList(coco.images[0].wavelength)
        self.labels.append(l)

        self.cubes.append(partial(get_legacy_cube, filepath, self.processing_mode))

        meta.flags = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        meta.references = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta.references[key] = partial(get_legacy_cube, val, cuvis.ProcessingMode.Raw)

        self.metas.append(meta)


# import cuvis
# from pathlib import Path
# from typing import Optional, Callable
# from torchvision import tv_tensors
# from .OutputFormat import OutputFormat
# import torch
# from torchvision.datasets import VisionDataset
# import yaml
# from ..tv_transforms import WavelengthList
# from pycocotools.coco import COCO
# import copy
# from cocolabels import COCOData
# import numpy as np
# from .metadata import Metadata, get_meta_from_session, get_meta_from_mesu, get_meta_from_path
# from functools import lru_cache, partial

# from cuvis.General import SDKException

# EXTENSION_SESSION = '.cu3s'
# EXTENSION_LEGACY = '.cu3'

# CUVIS_NON_CUBE_REFERENCES = (
#     cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)


# @lru_cache
# def get_session_cube(path, idx, proc_mode):
#     sess = cuvis.SessionFile(str(path))
#     mesu = sess[idx]
#     need_reprocess = bool(proc_mode is None)

#     if mesu.cube is None:
#         need_reprocess = True

#     if need_reprocess and proc_mode is not None:
#         pc = cuvis.ProcessingContext(sess)
#         pc.processing_mode = proc_mode
#         mesu = pc.apply(mesu)

#     if mesu.cube is None:
#         raise ValueError(f"Could not load Cube idx={idx} from SessionFile {path}.")  # nopep8
#     cube = tv_tensors.Image(mesu.cube)
#     return cube.to(memory_format=torch.channels_last)


# @lru_cache
# def get_session_reference(path, reftype):
#     sess = cuvis.SessionFile(str(path))
#     mesu = sess.get_reference(0, reftype)

#     if mesu.cube is None:
#         raise ValueError(f"Could not load Reference Cube {reftype} from SessionFile {path}.")  # nopep8
#     cube = tv_tensors.Image(mesu.cube)
#     return cube.to(memory_format=torch.channels_last)


# @lru_cache
# def get_legacy_cube(path, proc_mode):
#     mesu = cuvis.Measurement(str(path))
#     need_reprocess = bool(proc_mode is None)

#     if mesu.cube is None:
#         need_reprocess = True

#     if need_reprocess and proc_mode is not None:
#         pc = cuvis.ProcessingContext(mesu)
#         pc.processing_mode = proc_mode
#         mesu = pc.apply(mesu)

#     if mesu.cube is None:
#         raise ValueError(f"Could not load Cube from Legacy Measurement {path}.")  # nopep8
#     cube = tv_tensors.Image(mesu.cube)
#     return cube.to(memory_format=torch.channels_last)


# class CuvisDataset(VisionDataset):

#     def __init__(self, root: str = None,
#                  transforms: Optional[Callable] = None,
#                  transform: Optional[Callable] = None,
#                  target_transform: Optional[Callable] = None,
#                  output_format: OutputFormat = OutputFormat.Full,
#                  output_lambda: Optional[Callable] = None,
#                  force_proc_mode: Optional[cuvis.ProcessingMode] = None
#                  ):
#         self.processing_mode = force_proc_mode
#         super().__init__(root, transforms=transforms,
#                          transform=transform, target_transform=target_transform)
#         self.output_format = output_format
#         self.output_lambda = output_lambda

#         self._clear()
#         if root is None or not Path(root).exists():
#             raise RuntimeError(
#                 "Could not find root directory.")

#         self.root_dir = Path(root)

#         self.metadata_path = self.root_dir / "metadata.yaml"
#         if self.metadata_path.exists():
#             with open(self.metadata_path, 'r') as f:
#                 self.fileset_metadata = yaml.safe_load(f)
#         else:
#             self.metadata_path = None

#         self._load_directory(self.root_dir)
#         self.initialized = True

#     def _load_directory(self, directory: Path):

#         fileset_session = directory.glob(f'**/*{EXTENSION_SESSION}')
#         fileset_legacy = directory.glob(f'**/*{EXTENSION_LEGACY}')

#         for path in fileset_session:
#             self._load_session_file(path)
#         for path in fileset_legacy:
#             self._load_legacy_file(path)

#     def _load_session_file(self, session_path: Path):
#         session = cuvis.SessionFile(str(session_path))

#         if self.processing_mode is None:
#             tmp_mesu = session[0]
#             self.processing_mode = tmp_mesu.processing_mode

#         sess_meta = get_meta_from_session(session, session_path)

#         canvas_size = (sess_meta.shape[0], sess_meta.shape[1])

#         label_path = session_path.with_suffix('.json')
#         coco = COCOData.from_path(label_path) if label_path.exists() else None

#         for idx in range(len(session)):
#             cube_path = F"{session_path}:{idx}"
#             self.paths.append(cube_path)
#             self.cubes.append(partial(get_session_cube,
#                                       str(session_path), idx, self.processing_mode))

#             meta = copy.deepcopy(sess_meta)

#             for k, v in meta.references.items():
#                 if not isinstance(v, str):
#                     continue
#                 if Path(v).suffix == EXTENSION_SESSION and v == str(session_path):
#                     meta.references[k] = partial(get_session_reference,
#                                                  str(v), k)
#                 if Path(v).suffix == EXTENSION_SESSION:
#                     meta.references[k] = partial(get_session_cube,
#                                                  str(v), 0, cuvis.ProcessingMode.Raw)
#                 elif Path(v).suffix == EXTENSION_LEGACY:
#                     meta.references[k] = partial(get_legacy_cube,
#                                                  str(v), cuvis.ProcessingMode.Raw)

#             self.metas.append(meta)

#             l = {}
#             if coco is not None:
#                 anns = coco.annotations.where(image_id=idx)[0]
#                 l = anns.to_torchvision(canvas_size)
#                 l['wavelength'] = WavelengthList(coco.images[idx].wavelength)
#             self.labels.append(l)

#     def _load_legacy_file(self, legacy_path: Path):
#         self.paths.append(legacy_path)
#         labelpath = legacy_path.with_suffix(".json")

#         mesu = cuvis.Measurement(legacy_path)

#         meta = get_meta_from_mesu(mesu)

#         canvas_size = (meta.shape[0], meta.shape[1])

#         l = None
#         if labelpath.exists():
#             coco = COCOData.from_path(labelpath)
#             anns = coco.annotations.where(image_id=coco.image_ids[0])[0]
#             l = anns.to_torchvision(canvas_size)
#             l['wavelength'] = WavelengthList(coco.images[0].wavelength)
#         self.labels.append(l)

#         self.cubes.append(partial(get_legacy_cube,
#                                   legacy_path, self.processing_mode))

#         meta.flags = {}
#         for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
#             meta.flags[key] = val
#         meta.references = {}
#         for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
#             meta.references[key] = partial(get_legacy_cube,
#                                            val, cuvis.ProcessingMode.Raw)

#         self.metas.append(meta)

#     def _clear(self):
#         self.paths = []
#         self.cubes = []
#         self.metas = []
#         self.labels = []
#         self.fileset_metadata = {}
#         self.initialized = False
