from dataclasses import dataclass


@dataclass
class ChannelConfig:
    num_channels: int
    in_channels: int


def infer_channels_after_bandpass(datamodule, bandpass_cfg) -> ChannelConfig:
    """Infer post-bandpass channel count from a sample batch.

    Parameters
    ----------
    datamodule : object
        Datamodule with a train_dataloader() method returning batches with "wavelengths".
    bandpass_cfg : object
        Config with min_wavelength_nm and max_wavelength_nm fields.

    Returns
    -------
    ChannelConfig
        num_channels and in_channels set to the filtered channel count.
    """
    sample_batch = next(iter(datamodule.train_dataloader()))
    wavelengths = sample_batch["wavelengths"]
    keep_mask = wavelengths >= bandpass_cfg.min_wavelength_nm
    if bandpass_cfg.max_wavelength_nm is not None:
        keep_mask = keep_mask & (wavelengths <= bandpass_cfg.max_wavelength_nm)
    num_channels_after_bandpass = int(keep_mask.sum().item())
    return ChannelConfig(
        num_channels=num_channels_after_bandpass, in_channels=num_channels_after_bandpass
    )
