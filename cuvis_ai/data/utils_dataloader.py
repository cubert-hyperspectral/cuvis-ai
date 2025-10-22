import collections.abc
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from cuvis_ai.data.base_dataset import BaseDataset
from cuvis_ai.data.metadata import Metadata

container_abcs = collections.abc

string_classes = (str, bytes)


@staticmethod
def cuvis_collate(batch):
    r"""Puts each data field into a list or numpy array with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out).numpy()
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if re.search("[SaUO]", elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return np.stack(batch, axis=0)
        if elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(batch[0], int):
        return np.array(batch)
    elif isinstance(batch[0], float):
        return np.array(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: cuvis_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [cuvis_collate(samples) for samples in transposed]
    elif isinstance(batch[0], Metadata):
        return [b.dict() for b in batch]

    raise TypeError(error_msg.format(type(batch[0])))


def get_dataloader_from_dataset(
    dataset: BaseDataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=cuvis_collate,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
):
    return DataLoader(
        dataset,
        batch_size,
        shuffle,
        sampler,
        batch_sampler,
        num_workers,
        collate_fn,
        False,
        drop_last,
        timeout,
        worker_init_fn,
    )
