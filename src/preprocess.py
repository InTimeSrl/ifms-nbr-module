"""
preprocess.py -- Radiometric correction (DN -> reflectance) and cloud masking (SCL).

Used by:
  - baseline.py        (pre-campaign baseline construction)
  - pipeline.py        (operational continuous monitoring)
  - RGB output         (true-colour composite for visualisation)

Sentinel-2 Collection 1 L2A (Processing Baseline >= 05.00):
    reflectance = DN * 0.0001 - 0.1

The SCL band (Scene Classification Layer) is a classification map
(0-11, uint8) and does not require radiometric correction.

NBR bands (B8A, B12, SCL): all at 20 m, no resampling required.
RGB bands (B4, B3, B2): native 10 m; the composite is produced at 10 m,
  the SCL mask (20 m) is resampled to 10 m with nearest-neighbour.

Ref: technical specification, phases 4 (Radiometric Correction) and 6 (Cloud Masking).
"""

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# Sentinel-2 Collection 1 L2A radiometric fallbacks
# Used ONLY if scale/offset are absent from the scene STAC JSON.
# ---------------------------------------------------------------------------
_DEFAULT_SCALE = 0.0001
_DEFAULT_OFFSET = -0.1


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_band_calibration(meta, asset_key):
    """Extract scale and offset from the STAC JSON for a band.

    Parameters
    ----------
    meta : dict
        Scene metadata (from data_io.load_metadata).
    asset_key : str
        STAC asset key (e.g. "nir08", "swir22", "red").

    Returns
    -------
    scale : float
    offset : float
    """
    asset = meta.get("assets", {}).get(asset_key, {})
    scale = asset.get("scale", _DEFAULT_SCALE)
    offset = asset.get("offset", _DEFAULT_OFFSET)
    return scale, offset


def dn_to_reflectance(dn, scale=_DEFAULT_SCALE, offset=_DEFAULT_OFFSET):
    """Convert a DN array to surface reflectance.

    Formula: rho = DN * scale + offset
    The scale/offset values should be read from the scene STAC JSON
    via get_band_calibration().

    Parameters
    ----------
    dn : np.ndarray
        2D array of Digital Numbers (uint16).
    scale : float
        Multiplicative factor (default: 0.0001).
    offset : float
        Additive offset (default: -0.1).

    Returns
    -------
    np.ndarray
        Surface reflectance (float32).
    """
    return dn.astype("float32") * scale + offset


def scl_mask(scl, mask_classes=None):
    """Generate a boolean mask from SCL values.

    Parameters
    ----------
    scl : np.ndarray
        2D Scene Classification Layer array (uint8, values 0-11).
    mask_classes : list[int], optional
        SCL classes to mask (invalid pixels).
        If None, uses config.SCL_MASK_CLASSES.

    Returns
    -------
    np.ndarray (bool)
        True = valid pixel (not masked), False = pixel to discard.
    """
    if mask_classes is None:
        mask_classes = config.SCL_MASK_CLASSES
    return ~np.isin(scl, mask_classes)


def prepare_bands(nir_raw, swir_raw, scl, meta=None):
    """Full preprocessing: DN -> reflectance + SCL mask + nodata.

    Steps:
    1. Mask nodata pixels (DN == 0) on NIR and SWIR
    2. Mask invalid pixels from SCL (clouds, shadows, water, snow)
    3. Convert DN -> reflectance (scale/offset read from STAC JSON)

    Parameters
    ----------
    nir_raw : np.ndarray
        Band B8A (NIR, 20 m) as DN uint16.
    swir_raw : np.ndarray
        Band B12 (SWIR, 20 m) as DN uint16.
    scl : np.ndarray
        Band SCL (Scene Classification Layer, 20 m) uint8.
    meta : dict, optional
        Scene STAC metadata. If provided, scale/offset are read from
        meta["assets"][<band>]. If None, uses C1 defaults.

    Returns
    -------
    nir : np.ndarray (float32)
        NIR reflectance.
    swir : np.ndarray (float32)
        SWIR reflectance.
    valid_mask : np.ndarray (bool)
        True = valid pixel for NBR computation.
    """
    nodata = config.NODATA_DN
    mask_nodata = (nir_raw == nodata) | (swir_raw == nodata)
    mask_scl = scl_mask(scl)
    valid_mask = mask_scl & ~mask_nodata

    nir_s, nir_o = get_band_calibration(meta, "nir08") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    swir_s, swir_o = get_band_calibration(meta, "swir22") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)

    nir = dn_to_reflectance(nir_raw, nir_s, nir_o)
    swir = dn_to_reflectance(swir_raw, swir_s, swir_o)

    return nir, swir, valid_mask


def prepare_rgb(red_raw, green_raw, blue_raw, scl=None, meta=None):
    """Full RGB band preprocessing: DN -> reflectance + optional mask.

    RGB bands (B4, B3, B2) are at native 10 m; the SCL is at 20 m.
    If scl is provided and has different dimensions (20 m), it is resampled
    to 10 m with nearest-neighbour before applying the mask.

    Parameters
    ----------
    red_raw : np.ndarray
        Band B4 (Red, 10 m) as DN uint16.
    green_raw : np.ndarray
        Band B3 (Green, 10 m) as DN uint16.
    blue_raw : np.ndarray
        Band B2 (Blue, 10 m) as DN uint16.
    scl : np.ndarray, optional
        SCL band (20 m) uint8. If provided, invalid pixels are masked
        (set to 0 in the composite).
    meta : dict, optional
        Scene STAC metadata (for per-band scale/offset).

    Returns
    -------
    rgb : np.ndarray (float32)
        Array (3, H, W) with reflectance [R, G, B], clipped to [0, 1].
    valid_mask : np.ndarray (bool)
        Valid pixel mask at RGB resolution (H, W).
        If scl is None, all non-nodata pixels are valid.
    """
    nodata = config.NODATA_DN
    # AND: a pixel is nodata only if ALL bands are zero (tile borders).
    # With OR, single bands at low DN (e.g. B2 on dark vegetation) would punch
    # holes in valid pixels across the entire raster.
    mask_nodata = (red_raw == nodata) & (green_raw == nodata) & (blue_raw == nodata)

    r_s, r_o = get_band_calibration(meta, "red") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    g_s, g_o = get_band_calibration(meta, "green") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    b_s, b_o = get_band_calibration(meta, "blue") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)

    red = dn_to_reflectance(red_raw, r_s, r_o)
    green = dn_to_reflectance(green_raw, g_s, g_o)
    blue = dn_to_reflectance(blue_raw, b_s, b_o)

    # Clip to [0, 1] for visualisation
    rgb = np.clip(np.array([red, green, blue]), 0.0, 1.0)

    valid_mask = ~mask_nodata

    if scl is not None:
        mask_scl = scl_mask(scl)
        # If SCL is at 20 m and RGB at 10 m, resample with nearest-neighbour
        if mask_scl.shape != valid_mask.shape:
            mask_scl = _upsample_nearest(mask_scl, valid_mask.shape)
        valid_mask = valid_mask & mask_scl

    return rgb, valid_mask


def _upsample_nearest(arr, target_shape):
    """Resample a 2D array to target_shape with nearest-neighbour.

    Parameters
    ----------
    arr : np.ndarray
        Source 2D array (e.g. 20 m).
    target_shape : tuple
        (height, width) target (e.g. 10 m).

    Returns
    -------
    np.ndarray
        Resampled array.
    """
    row_idx = np.linspace(0, arr.shape[0] - 1, target_shape[0]).round().astype(int)
    col_idx = np.linspace(0, arr.shape[1] - 1, target_shape[1]).round().astype(int)
    return arr[np.ix_(row_idx, col_idx)]
