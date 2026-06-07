"""
indices.py -- Spectral index computation for burned area detection.

Indices implemented:
  - NBR  = (NIR - SWIR) / (NIR + SWIR)
  - dNBR = NBR_pre - NBR_post
  - RBR  = dNBR / (NBR_pre + 1.001)

Used by:
  - baseline.py   (compute_nbr for composite construction)
  - pipeline.py   (compute_dnbr / compute_rbr for continuous monitoring)

The choice between dNBR and RBR is configured via config.INDEX_MODE.

Ref: Key & Benson 2006 (dNBR), Parks et al. 2014 (RBR).
"""

import numpy as np


def compute_nbr(nir, swir):
    """Normalized Burn Ratio: NBR = (NIR - SWIR) / (NIR + SWIR).

    Parameters
    ----------
    nir : np.ndarray (float32)
        Band B8A reflectance (from preprocess.prepare_bands).
    swir : np.ndarray (float32)
        Band B12 reflectance (from preprocess.prepare_bands).

    Returns
    -------
    np.ndarray (float32)
        NBR values, typically in [-1, 1]. Division by zero -> 0.
    """
    denom = nir + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = np.where(np.abs(denom) > 1e-6, (nir - swir) / denom, 0.0)
    return np.clip(nbr, -1.0, 1.0).astype("float32")


def compute_dnbr(nbr_pre, nbr_post):
    """Differenced NBR: dNBR = NBR_pre - NBR_post.

    Positive values indicate vegetation loss (potential fire).

    Parameters
    ----------
    nbr_pre : np.ndarray (float32)
        Pre-event NBR (baseline or previous).
    nbr_post : np.ndarray (float32)
        Post-event NBR (current scene).

    Returns
    -------
    np.ndarray (float32)
    """
    return (nbr_pre - nbr_post).astype("float32")


def compute_rbr(nbr_pre, nbr_post):
    """Relativized Burn Ratio: RBR = dNBR / (NBR_pre + 1.001).

    Normalises dNBR relative to the pre-event value, reducing dependence
    on initial vegetation cover. Useful in heterogeneous areas
    (mix of vegetation and bare soil).

    The 1.001 offset avoids division by zero (theoretical NBR_pre = -1).

    Parameters
    ----------
    nbr_pre : np.ndarray (float32)
        Pre-event NBR.
    nbr_post : np.ndarray (float32)
        Post-event NBR.

    Returns
    -------
    np.ndarray (float32)

    Ref: Parks, Dillon, Miller (2014), Int. J. Wildland Fire.
    """
    dnbr = nbr_pre - nbr_post
    return (dnbr / (nbr_pre + 1.001)).astype("float32")
