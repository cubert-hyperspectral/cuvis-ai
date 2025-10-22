from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision import tv_tensors

from cuvis_ai.data.base_dataset import BaseDataset
from cuvis_ai.data.metadata import Metadata
from cuvis_ai.data.output_format import OutputFormat
from cuvis_ai.tv_transforms import WavelengthList

debug_enabled = True

NUMPY_EXTENSION = ".npy"


@lru_cache
def get_numpy(path, to_dtype: np.dtype):
    cube = np.load(path)
    if cube.dtype != to_dtype:
        cube = cube.astype(to_dtype)
    cube = tv_tensors.Image(cube)
    while len(cube.shape) < 4:
        cube = cube.unsqueeze(0)
    return cube.to(memory_format=torch.channels_last)


@lru_cache
def get_opencv(path, to_dtype: np.dtype):
    cube = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if cube.dtype != to_dtype:
        cube = cube.astype(to_dtype)
    cube = tv_tensors.Image(cube)
    while len(cube.shape) < 4:
        cube = cube.unsqueeze(0)
    return cube.to(memory_format=torch.channels_last)


class NumpyDataset(BaseDataset):
    """Representation for a set of data cubes, their meta-data and labels.

    This class is a subclass of torchvisions VisionDataset which is a subclass
    of torch.utils.data.Dataset.
    This class can be used anywhere these classes can be used, such as initializing
    a pytorch dataloader.

    Upon creation, the path set via :attr:`root` is scanned recursively for any compatible data.
    Any with the .npy extension are incorporated into the dataset.
    Metadata that is valid for the entire dataset is expected to be in the file `root`/metadata.yaml.
    See :class:`Metadata` for an explanation of the data format.
    Labels for any data file are expected in a file of the same name but with the .json extension in COCO label format.

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

    If :attr:`root` is not passed in the constructor, the :py:meth:`~NumpyDataset.initialize` or :py:meth:`~NumpyDataset.load` method has to be called with a root path before the dataset can be used.
    """

    def __init__(
        self,
        root: str | None = None,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Callable | None = None,
    ):
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            output_format=output_format,
            output_lambda=output_lambda,
        )
        self._clear()

        if root is not None:
            self.initialize(root)

    def initialize(self, root: str, force: bool = False):
        """Initialize the dataset by scanning the provided directory for data.
        Initialize will be called by the constructor if a root path is provided or by the load method.

        Parameters
        ----------
        root : str
            Path of the directory containing the data this dataset will represent.
        force : bool
            If True, the dataset will clear all currently held data and re-initialize with the provided root path.
        """
        if self.initialized:
            if force:
                self._clear()
            else:
                raise RuntimeError(
                    "Cannot initialize an already initialized dataset. Use force=True if this was intended."
                )
        self.root = Path(root)

        self.metadata_filepath = self.root / "metadata.yaml"

        if self.metadata_filepath.exists():
            self.fileset_metadata = yaml.safe_load(open(self.metadata_filepath))
        else:
            self.metadata_filepath = ""

        # Actual data collection
        self._load_directory(self.root)
        self.initialized = True

    def _clear(self):
        self.root = None
        self.paths = []
        self.cubes = []
        self.metas = []
        self.labels = []
        self.fileset_metadata = {}
        self.data_types: set = set()
        self.initialized = False

    def _load_directory(self, dir_path: str):
        dir_path = Path(dir_path)
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset = dir_path.glob(f"**/*{NUMPY_EXTENSION}")
        for cur_path in fileset:
            self._load_file(cur_path)

    def _load_file(self, filepath: Path):
        if debug_enabled:
            print("Found file:", filepath)

        self.paths.append(filepath)
        self.cubes.append(partial(get_numpy, filepath))

        # meta = {} # metadataInit(filepath, self.fileset_metadata)
        meta = Metadata()
        try:
            meta.wavelengths_nm = WavelengthList(meta.wavelengths_nm)
        except:
            pass

        temp_data = np.load(filepath)
        meta.shape = temp_data.shape
        meta.datatype = temp_data.dtype
        self.data_types.add(temp_data.dtype)

        try:
            refs = meta.references
            for t, v in refs.items():
                if isinstance(v, str) and Path(v).exists():
                    if Path(v).suffix == ".npy":
                        meta.references[t] = partial(get_numpy, v)
                    else:
                        meta.references[t] = partial(get_opencv, v)
        except KeyError:
            pass
        self.metas.append(meta)

        labelpath = filepath.with_suffix(".json")
        has_labels = False  # labelpath.exists()
        canvas_size = (meta.shape[0], meta.shape[1])
        l = None
        # if has_labels:
        #     coco = COCOData.from_path(labelpath)
        #     anns = coco.annotations.where(image_id=coco.image_ids[0])[0]
        #     l = anns.to_torchvision(canvas_size)
        #     l['wavelength'] = WavelengthList(coco.images[0].wavelength)
        self.labels.append(l)

    def __len__(self) -> int:
        """The number of data elements this data set holds."""
        return len(self.cubes)

    def __getitem__(self, idx: int | slice) -> tuple[np.ndarray, list[dict], list[dict]]:
        """Return data element number 'idx' in the selected :attr:`OutputFormat`.
        Default is `OutputFormat.Full`, tuple(cube, meta-data, labels):
        """
        data = self.get_data(idx)
        label = self.get_labels(idx)
        meta = self.get_metadata(idx)
        return self._get_return_shape(data, label, meta)

    def merge(self, other_dataset):
        """Merge another NumpyData dataset into this dataset."""
        if type(self) is type(other_dataset):
            self.paths.extend(other_dataset.paths)
            self.cubes.extend(other_dataset.cubes)
            self.labels.extend(other_dataset.labels)
            self.metas.extend(other_dataset.metas)
        else:
            raise TypeError(
                f"Cannot merge NumpyData with an object of type {type(other_dataset).__name__}"
            )

    def get_dataitems_datatypes(self) -> list:
        """Get a list with all datatypes detected when scanning the root folder."""
        return list(self.data_types)

    def get_data(self, idx: int | slice):
        """Get the cube at 'idx'."""
        loaded_cubes = list(
            map(
                lambda c: c(self.provide_datatype),
                [self.cubes[idx]] if isinstance(idx, int) else self.cubes[idx],
            )
        )
        # torchvision transforms don't yet respect the memory layout property of tensors. They assume NCHW while cubes are in NHWC
        return (
            self._apply_transform(torch.concatenate(loaded_cubes, dim=0).permute([0, 3, 1, 2]))
            .permute([0, 2, 3, 1])
            .numpy()
        )

    def get_metadata(self, idx: int | slice):
        """Get the meta-data for the cube at 'idx' in this dataset."""

        def transform_meta(m):
            m_out = deepcopy(m)
            try:
                m.wavelengths_nm = self._apply_transform(m.wavelengths_nm, True)
            except:
                pass
            for t, v in m.references.items():
                try:
                    refdata = v(self.provide_datatype)
                except:
                    refdata = v
                if isinstance(refdata, torch.Tensor):
                    refdata = (
                        self._apply_transform(refdata.permute([0, 3, 1, 2]))
                        .permute([0, 2, 3, 1])
                        .numpy()
                    )
                m_out.references[t] = refdata
            return m_out

        return list(
            map(transform_meta, [self.metas[idx]] if isinstance(idx, int) else self.metas[idx])
        )

    def get_labels(self, idx: int | slice):
        """Get the labels for the cube at 'idx' in this dataset."""
        labels = [self.labels[idx]] if isinstance(idx, int) else self.labels[idx]
        return list(map(self._apply_transform, labels, [True] * len(labels)))
