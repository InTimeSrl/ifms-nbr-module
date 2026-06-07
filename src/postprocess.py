"""
postprocess.py -- Morphological filters and burnt area polygon vectorisation.

Operations:
  1. Sieve: remove isolated patches < MIN_PATCH_PIXELS
  2. Fill holes: fill internal holes < HOLE_FILL_PIXELS
  3. Vectorize by severity: dissolve by severity class (3-7)
     -> MultiPolygon feature with attributes area_ha, dnbr_mean, event metadata
  4. RGB HONC: Highlight Optimized Natural Color composite for visualisation

Ref: technical specification, post-processing phase.
"""

import logging
from pathlib import Path

import numpy as np
from rasterio.features import sieve, shapes
from scipy.ndimage import binary_fill_holes, label, maximum_filter
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from . import config
from . import data_io
from . import preprocess

logger = logging.getLogger(__name__)


def morphological_filter(severity, valid_mask):
    """Apply morphological filters to the severity map.

    Steps:
    1. Sieve: remove burnt patches (severity >= 4) with fewer than
       MIN_PATCH_PIXELS contiguous pixels.
    2. Fill holes: fill internal holes in burnt areas smaller than
       HOLE_FILL_PIXELS pixels.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Severity map (0=nodata, 1-7=USGS classes).
    valid_mask : np.ndarray (bool)
        Valid pixel mask.

    Returns
    -------
    np.ndarray (uint8)
        Filtered severity map.
    """
    out = severity.copy()

    # --- Sieve: remove burnt patches that are too small ---
    # Works on the binary burnt mask (severity >= 4) for the sieve,
    # then maps original classes back only where they survive.
    burnt = (out >= 4).astype("uint8")
    if burnt.any():
        burnt_sieved = sieve(burnt, size=config.MIN_PATCH_PIXELS, connectivity=8)
        # Zero out classes in pixels removed by the sieve
        removed = (burnt == 1) & (burnt_sieved == 0)
        out[removed] = 3  # Unburned

    # --- Fill holes: fill small internal holes in burnt areas ---
    burnt = (out >= 4)
    if burnt.any():
        filled = _fill_small_holes(burnt, max_pixels=config.HOLE_FILL_PIXELS)
        # Pixels added by fill: assign the most common neighbour class
        new_pixels = filled & ~burnt & valid_mask
        if new_pixels.any():
            out[new_pixels] = _assign_neighbor_class(out, new_pixels)

    return out


def vectorize_by_severity(severity, dnbr, profile, meta=None):
    """Dissolve by severity class -> 1 MultiPolygon feature per class.

    Geometries in the native raster CRS (no reprojection).
    Includes classes 3-7 (Unburned + all burn severity classes).
    Classes 1-2 (Enhanced Regrowth) excluded: vegetation regrowth is a
    long-term phenomenon, not relevant in NRT monitoring.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Filtered severity map (classes 3-7).
    dnbr : np.ndarray (float32)
        Corresponding dNBR (or RBR) map.
    profile : dict
        Rasterio profile (contains transform and CRS).
    meta : dict, optional
        Global attributes to add to each feature:
        - event_id, detection_date, processing_mode, aoi_ref,
          cloud_cover_pct, index_mode, index_threshold

    Returns
    -------
    list[dict]
        List of GeoJSON features (one per severity class present).
    """
    if meta is None:
        meta = {}

    transform = profile["transform"]
    pixel_res = abs(transform.a)
    pixel_area_ha = (pixel_res * pixel_res) / 10000.0

    # Verify that there are burnt pixels (>= 4) before proceeding
    burnt_mask = (severity >= 4).astype("uint8")
    if not burnt_mask.any():
        return []

    features = []
    for class_id in sorted(config.SEVERITY_CLASSES.keys()):
        if class_id < 4:  # exclude ER-H, ER-L, Unburned
            continue
        class_mask = (severity == class_id)
        if not class_mask.any():
            continue

        # Dissolve: collect all geometries for this class and merge
        class_u8 = class_mask.astype("uint8")
        geoms = []
        for geom_dict, val in shapes(class_u8, mask=class_u8,
                                     transform=transform):
            if val == 1:
                geoms.append(shape(geom_dict))
        if not geoms:
            continue
        merged = unary_union(geoms)

        # dNBR statistics for this class
        dnbr_vals = dnbr[class_mask & np.isfinite(dnbr)]
        n_pixels = int(class_mask.sum())

        class_info = config.SEVERITY_CLASSES.get(class_id, {})
        props = {
            "event_id": meta.get("event_id", ""),
            "detection_date": meta.get("detection_date", ""),
            "closed_date": meta.get("closed_date", ""),
            "closure_reason": meta.get("closure_reason", ""),
            "tile": meta.get("tile", ""),
            "aoi_ref": meta.get("aoi_ref", ""),
            "index_mode": meta.get("index_mode", ""),
            "index_threshold": meta.get("index_threshold", None),
            "class_id": class_id,
            "class_label": class_info.get("label", ""),
            "area_ha": round(n_pixels * pixel_area_ha, 2),
            "dnbr_mean": round(float(dnbr_vals.mean()), 4) if len(dnbr_vals) else None,
            "n_detection_scenes": meta.get("n_detection_scenes", None),
        }

        features.append({
            "type": "Feature",
            "geometry": mapping(merged),
            "properties": props,
        })

    return features


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def filter_small_clusters(severity, pixel_res_m, min_area_ha):
    """Remove burnt clusters smaller than min_area_ha from the severity map.

    Retains 8-connected components with area >= min_area_ha;
    smaller components are zeroed out. Applied before vectorisation to
    ensure consistency between severity_final.tif, burnt_final and
    fire_footprint.

    Parameters
    ----------
    severity : np.ndarray (uint8, 2D)
        Severity map (0 = nodata, 1-7 = USGS classes).
    pixel_res_m : float
        Pixel resolution in metres (typically 20.0 for Sentinel-2).
    min_area_ha : float
        Minimum cluster area (ha) to retain.

    Returns
    -------
    np.ndarray (uint8)
        Filtered severity map.
    """
    burnt = (severity >= 4)
    if not burnt.any():
        return severity.copy()

    pixel_area_ha = (pixel_res_m * pixel_res_m) / 10_000.0
    min_px = int(np.ceil(min_area_ha / pixel_area_ha))

    structure = np.ones((3, 3), dtype=bool)
    labeled, _ = label(burnt, structure=structure)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background

    keep_ids = np.where(counts >= min_px)[0]
    keep_mask = np.isin(labeled, keep_ids)

    out = severity.copy()
    out[burnt & ~keep_mask] = 0
    return out


def _fill_small_holes(mask, max_pixels):
    """Fill internal holes in a binary mask if smaller than max_pixels.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Binary mask (True = burnt area).
    max_pixels : int
        Maximum hole size to fill.

    Returns
    -------
    np.ndarray (bool)
        Mask with small holes filled.
    """
    filled = binary_fill_holes(mask)
    if max_pixels is None:
        return filled

    # Identify filled holes and check their size
    holes = filled & ~mask
    if not holes.any():
        return filled

    labeled, n_holes = label(holes)
    out = mask.copy()
    for i in range(1, n_holes + 1):
        hole = labeled == i
        if hole.sum() <= max_pixels:
            out[hole] = True

    return out


def _assign_neighbor_class(severity, new_pixels):
    """Assign new pixels the class of the most common burnt neighbour.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Current severity map.
    new_pixels : np.ndarray (bool)
        Pixels just added by the fill.

    Returns
    -------
    np.ndarray (uint8)
        Classes to assign to new_pixels (same shape summed).
    """
    # Dilate burnt classes and take the highest value in the 3x3 neighbourhood
    burnt_classes = severity.copy()
    burnt_classes[burnt_classes < 4] = 0
    neighbor_class = maximum_filter(burnt_classes, size=3)
    return neighbor_class[new_pixels]


# ---------------------------------------------------------------------------
# RGB HONC composite (visualisation output, optional)
# ---------------------------------------------------------------------------

def save_rgb_composite(scene, aoi, scene_dir, out_dir):
    """Produce and save a Highlight Optimized Natural Color (HONC) RGB composite.

    Loads bands B4, B3, B2 at native 10 m, applies percentile stretch
    (p2-p98) + gamma 0.8, and saves a uint8 3-band GeoTIFF.

    Parameters
    ----------
    scene : dict
        Scene metadata.
    aoi : dict
        AOI dict.
    scene_dir : str or Path
        Local TIF folder (or None for remote COG read).
    out_dir : Path
        Scene output folder.

    Ref: Marko Repse, Sentinel Hub Custom Scripts (highlight_optimized_natural_color).
    """
    scene_id = scene["stac_item_id"]
    try:
        raster_crs = data_io.get_scene_crs(scene, scene_dir)
        bbox = data_io.get_aoi_bbox_raster(aoi, raster_crs)
        bands, profile_rgb = data_io.load_scene_bands(
            scene, scene_dir,
            asset_keys=["red", "green", "blue"],
            bbox=bbox,
        )

        rgb, valid_mask = preprocess.prepare_rgb(
            bands["red"], bands["green"], bands["blue"],
            scl=None, meta=scene,
        )

        # RGB true-colour: global percentile stretch (p2-p98) + gamma 0.8
        all_valid = rgb[:, valid_mask]          # (3, N)
        if all_valid.size > 0:
            lo = np.percentile(all_valid, 2)
            hi = np.percentile(all_valid, 98)
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        stretched = (rgb - lo) / (hi - lo)
        stretched = np.power(np.clip(stretched, 0.0, 1.0), 0.8)
        honc_uint8 = np.round(stretched * 255).astype(np.uint8)
        honc_uint8[:, ~valid_mask] = 0             # nodata pixels to zero only

        out_path = Path(out_dir) / f"{scene_id}_RGB_HONC.tif"
        data_io.write_rgb_geotiff(honc_uint8, profile_rgb, out_path)
        logger.info("RGB HONC saved: %s", out_path)
    except Exception:
        logger.warning("RGB HONC not produced for %s (RGB bands not available)", scene_id)
