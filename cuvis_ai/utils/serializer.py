import pickle as pk
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from cuvis_ai.utils.filesystem import change_working_dir

# TODO make yaml serializer


class Serializer(ABC):
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    @abstractmethod
    def serialize(self, data: dict) -> None:
        pass

    @abstractmethod
    def load(self) -> dict:
        pass


class CuvisYamlDumper(yaml.SafeDumper):
    pass


class CuvisYamlLoader(yaml.SafeLoader):
    pass


def numpy_array_string_representer(dumper, data):
    return dumper.represent_scalar("!numpy.ndarray.str", np.array2string(data, separator=","))


def numpy_array_binary_representer(dumper, data):
    return dumper.represent_scalar("!numpy.ndarray.bin", np.array2string(data, separator=","))


def numpy_array_file_representer(dumper, data):
    tmp_filename = f"{uuid.uuid4()}.npy"

    np.save(tmp_filename, data)

    return dumper.represent_scalar("!numpy.ndarray.file", tmp_filename)


def numpy_arrray_string_constructor(loader, node):
    value = loader.construct_scalar(node)
    return np.fromstring(value.strip("[]"), sep=",")


def numpy_arrray_file_constructor(loader, node):
    value = loader.construct_scalar(node)
    return np.load(value)


def numpy_float32_representer(dumper, data):
    return dumper.represent_scalar("!numpy.float32", str(float(data)))


def numpy_float32_constructor(loader, node):
    value = loader.construct_scalar(node)
    return np.float32(value)


def numpy_float64_representer(dumper, data):
    return dumper.represent_scalar("!numpy.float64", str(float(data)))


def numpy_float64_constructor(loader, node):
    value = loader.construct_scalar(node)
    return np.float64(value)


CuvisYamlDumper.add_representer(np.ndarray, numpy_array_file_representer)
CuvisYamlDumper.add_representer(np.float32, numpy_float32_representer)
CuvisYamlDumper.add_representer(np.float64, numpy_float64_representer)
CuvisYamlLoader.add_constructor("!numpy.ndarray.str", numpy_arrray_string_constructor)
CuvisYamlLoader.add_constructor("!numpy.ndarray.file", numpy_arrray_file_constructor)
CuvisYamlLoader.add_constructor("!numpy.float32", numpy_float32_constructor)
CuvisYamlLoader.add_constructor("!numpy.float64", numpy_float64_constructor)


class YamlSerializer(Serializer):
    def __init__(self, data_dir: Path, filename: str = "main") -> None:
        super().__init__(data_dir)

        self.filename = filename

    def serialize(self, data: dict) -> None:
        with change_working_dir(self.data_dir):
            with open(f"{self.filename}.yml", "w") as f:
                yaml.dump(data, f, Dumper=CuvisYamlDumper, default_flow_style=False)

    def load(self) -> dict:
        with change_working_dir(self.data_dir):
            with open(f"{self.filename}.yml") as f:
                data = yaml.load(f, Loader=CuvisYamlLoader)
        return data


class OldSerializer:
    def __init__(self, serial_dir, *, pickle_inner=True) -> None:
        self.serial_dir = Path(serial_dir)
        self.pickle_inner = pickle_inner

    def serialize_node(self, node, name, *, inner_module: str, params_list: list[str]) -> str:
        pickle_name = f"{hash(node.__dict__[inner_module])}_{name}.pkl"
        if self.pickle_inner:
            pk.dump(node.__dict__[inner_module], open(self.serial_dir / pickle_name, "wb"))

        data = {"type": type(node).__name__, "id": node.id}
        if self.pickle_inner:
            data[f"{name}_object"] = pickle_name

        additional_params = {p: node.__dict__[p] for p in params_list}
        data = data | additional_params
        return yaml.dump(data, default_flow_style=False)

    def load_node(self, node, name, *, params: dict, inner_module: str, params_list: list[str]):
        node.id = params["id"]

        for p in params_list:
            node.__dict__[p] = params[p]

        if self.pickle_inner:
            pickle_name = params[f"{name}_object"]
            node.__dict__[inner_module] = pk.load(open(self.serial_dir / pickle_name, "rb"))

        node.initialized = True
