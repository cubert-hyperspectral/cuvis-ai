"""
Deep SVDD Channel Configuration Utilities.

This module provides utilities for inferring channel counts after bandpass
filtering for Deep SVDD networks. This is useful for automatically configuring
network architectures based on the data pipeline's preprocessing steps.

See Also
--------
cuvis_ai.node.preprocessors : Bandpass filtering nodes
cuvis_ai.anomaly.deep_svdd : Deep SVDD anomaly detection
"""

from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Configuration for network channel counts.

    Stores the number of input and output channels for network layers,
    typically determined after bandpass filtering.

    Attributes
    ----------
    num_channels : int
        Total number of channels in the network.
    in_channels : int
        Number of input channels to the network.
    """

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
