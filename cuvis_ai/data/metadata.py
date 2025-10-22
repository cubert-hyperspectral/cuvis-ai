from dataclasses import asdict, dataclass
from pathlib import Path

import cuvis
from dataclass_wizard import YAMLWizard

CUVIS_NON_CUBE_REFERENCES = (cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)


@dataclass
class Metadata(YAMLWizard):
    """
    The meta-data dictionary is a collection of meta-data for a data cube or dataset.

    This information can be extracted from cubes stored in the cu3s file format automatically.

    For all other data, a metadata.yaml file can be placed in the root data path to provide this meta data.
    The yaml file can either contain any of the attributes of this class directly or contain one or multiple 'fileset' entries.
    Each 'fileset' must contain a 'paths' entry, specifying a list of filepaths that this set of attributes is valid for.

    Possible entries
    ----------------
    name : str
        Name of the file that this object describes.
    shape : tuple
        Numpy shape describing the cube size as (columns, rows, channels)
    wavelengths_nm : list
        A list of wavelengths the cube contains. In the same order as the channels.
    bit_depth : int
        Bit depth of the source data the cube was computed from.
    references : Dict
        A dictionary containing filenames or links to the data references. e.g: white and dark cubes used to calculate reflectance data.
    integration_time_us : float
        The integration time (also exposure time) in microseconds used to record the data.
    framerate : float
        For video data. The number of measurements taken per second.
    flags : Dict
        Data dependend dictionary. Any additional flags associated with measurements. e.g: Overillumination, dead pixels, missing references or data, bad references, key frames, etc.
    processing_mode : str
        The processing mode the data was calculated with.
    """

    name: str
    shape: tuple[int, int, int]
    wavelengths_nm: list[float]
    bit_depth: int
    references: dict[str, str]
    integration_time_us: float
    framerate: float
    flags: dict[str, str]
    processing_mode: str

    from dataclasses import asdict, dataclass

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def get_meta_from_path(path: Path) -> Metadata | list[Metadata]:
    with open(path) as f:
        yaml_data = f.read()

    try:
        meta = Metadata.from_yaml(yaml_data)
        return meta
    except:
        metas = Metadata.from_yaml_list(yaml_data)
        return metas


def get_meta_from_session(session: cuvis.SessionFile, path: Path) -> Metadata:
    tmp_mesu = session[0]
    tmp_cube = tmp_mesu.cube

    shape = (tmp_cube.width, tmp_cube.height, tmp_cube.channels)
    wavelengths_nm = tmp_cube.wavelength
    # wavelengths_nm = WavelengthList(tmp_cube.wavelength)
    integration_time_us = int(tmp_mesu.integration_time * 1000)
    flags = {}
    references = {}
    for reftype in cuvis.ReferenceType:
        try:
            refmesu = session.get_reference(0, reftype)
        except cuvis.General.SDKException:
            refmesu = None
        if refmesu is None or reftype in CUVIS_NON_CUBE_REFERENCES:
            continue
        references[reftype.name] = str(path)
    try:
        framerate = session.fps
    except cuvis.General.SDKException:
        framerate = 0
    bit_depth = 0
    processing_mode = tmp_mesu.processing_mode

    meta = Metadata(
        name=str(path),
        shape=shape,
        wavelengths_nm=wavelengths_nm,
        bit_depth=bit_depth,
        references=references,
        integration_time_us=integration_time_us,
        framerate=framerate,
        flags=flags,
        processing_mode=processing_mode,
    )
    return meta


def get_meta_from_mesu(mesu: cuvis.Measurement) -> Metadata:
    tmp_cube = mesu.cube

    shape = (tmp_cube.width, tmp_cube.height, tmp_cube.channels)
    wavelengths_nm = tmp_cube.wavelength
    integration_time_us = int(mesu.integration_time * 1000)
    flags = {}
    references = {}
    framerate = 0
    bit_depth = 0

    meta = Metadata(
        shape=shape,
        wavelengths_nm=wavelengths_nm,
        bit_depth=bit_depth,
        references=references,
        integration_time_us=integration_time_us,
        framerate=framerate,
        flags=flags,
        processing_mode=None,
    )
    return meta
