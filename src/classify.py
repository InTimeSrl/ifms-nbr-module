"""
classify.py -- Severity classification from dNBR/RBR (7 USGS classes).

Classes (from config.SEVERITY_CLASSES):
  1: Enhanced Regrowth, High    (dNBR < -0.251)
  2: Enhanced Regrowth, Low     (-0.251 <= dNBR < -0.101)
  3: Unburned                   (-0.101 <= dNBR <  0.099)
  4: Low Severity               ( 0.099 <= dNBR <  0.269)
  5: Moderate-Low Severity      ( 0.269 <= dNBR <  0.439)
  6: Moderate-High Severity     ( 0.439 <= dNBR <  0.659)
  7: High Severity              ( 0.659 <= dNBR)

Invalid pixels (valid_mask=False) remain at 0 (nodata).

Used by pipeline.py (process_scene) when a fire is detected.

Ref: Key & Benson 2006, USGS FIREMON.
"""

import numpy as np

from . import config


def classify_severity(delta, valid_mask):
    """Classify severity pixel-by-pixel from dNBR (or RBR).

    Parameters
    ----------
    delta : np.ndarray (float32)
        dNBR or RBR map (positive values = vegetation loss).
    valid_mask : np.ndarray (bool)
        True = valid pixel, False = cloud/shadow/nodata.

    Returns
    -------
    np.ndarray (uint8)
        Severity map: 0 = nodata, 1-7 = USGS classes.
    """
    severity = np.zeros(delta.shape, dtype="uint8")

    for class_id, info in config.SEVERITY_CLASSES.items():
        lo = info["min"]
        hi = info["max"]
        if lo is None and hi is not None:
            mask = delta < hi
        elif lo is not None and hi is None:
            mask = delta >= lo
        else:
            mask = (delta >= lo) & (delta < hi)
        severity[mask & valid_mask] = class_id

    return severity
