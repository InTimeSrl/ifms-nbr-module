"""
baseline.py -- Pre-campaign retrospective baseline NBR.

One-shot: retrieves scenes from the look-back window, filters them,
computes NBR, builds a median composite with MAD anomaly filter, and
saves to disk (baseline_nbr, previous_nbr).
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config
from . import data_io
from . import preprocess, indices
from . import state as pipeline_state

logger = logging.getLogger(__name__)


def _nanmedian_no_allnan_warning(arr, axis=0):
    """Compute nanmedian silencing expected all-NaN slice warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


# ---------------------------------------------------------------------------
# Persistent NBR file paths for a tile
# ---------------------------------------------------------------------------

def nbr_paths(aoi, data_dir="data"):
    """Return the paths of persistent NBR files for an AOI.

    Parameters
    ----------
    aoi : dict
        AOI dict (from data_io.load_aoi).
    data_dir : str or Path
        Root folder for persistent data.

    Returns
    -------
    dict
        {"baseline": Path, "previous": Path}
    """
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "baseline": d / "baseline_nbr.tif",
        "previous": d / "previous_nbr.tif",
    }


def load_nbr(path):
    """Load an NBR raster from disk. Returns (array, profile) or (None, None)."""
    if Path(path).exists():
        return data_io.read_band(str(path))
    return None, None


def save_nbr(data, profile, path):
    """Save an NBR raster to disk."""
    data_io.write_geotiff(data, profile, path, dtype="float32")


# ---------------------------------------------------------------------------
# Filters and per-scene NBR computation
# ---------------------------------------------------------------------------

def _filter_baseline_scenes(scenes):
    """Filter pre-campaign scenes by quality (cloud cover, processing baseline).

    Pre-campaign scenes are not tracked by the watermark: the filter
    only checks radiometric quality criteria.
    """
    valid = []
    for scene in scenes:
        cc = scene.get("cloud_cover")
        if cc is not None and cc > config.MAX_CLOUD_COVER_PCT:
            continue
        pb = scene.get("processing_baseline", "99.99")
        if pb < config.MIN_PROCESSING_BASELINE:
            continue
        valid.append(scene)
    return valid


def compute_nbr_from_scene(scene, aoi, scene_dir):
    """Load a scene and return (nbr, nir, swir, valid_mask, profile) or None.

    Parameters
    ----------
    scene : dict
        Scene metadata.
    aoi : dict
        AOI dict.
    scene_dir : str or Path
        Local TIF folder.

    Returns
    -------
    tuple (nbr, nir, swir, valid_mask, profile) or None
        None only if the scene has no valid pixels at all (degenerate case).
        The operational quality filter (SCENE_VALID_SCL_PCT) is applied
        downstream in the monitoring loop.
    """
    # Detect actual CRS from the first raster band (not from metadata)
    raster_crs = data_io.get_scene_crs(scene, scene_dir)
    bbox = data_io.get_aoi_bbox_raster(aoi, raster_crs)
    asset_keys = ["nir08", "swir22", "scl"]
    bands, profile = data_io.load_scene_bands(
        scene, scene_dir, asset_keys=asset_keys, bbox=bbox,
    )
    nir, swir, valid_mask = preprocess.prepare_bands(
        bands["nir08"], bands["swir22"], bands["scl"], meta=scene,
    )
    if not valid_mask.any():
        return None

    nbr = indices.compute_nbr(nir, swir)
    return nbr, nir, swir, valid_mask, profile


# ---------------------------------------------------------------------------
# Retrospective baseline construction
# ---------------------------------------------------------------------------

def build_baseline(aoi, get_scenes_fn, scene_dir=None, data_dir="data"):
    """Build the retrospective NBR baseline for an AOI.

    Retrieves scenes from the pre-campaign look-back window, builds a
    median composite with MAD anomaly filter, and saves baseline_nbr.tif.
    Initialises previous_nbr.

    Parameters
    ----------
    aoi : dict
        AOI dict (from data_io.load_aoi).
    get_scenes_fn : callable
        Function to retrieve scenes: get_scenes_fn(date_from) -> list[dict].
        Must return scenes for the AOI starting from the given date.
    scene_dir : str or Path, optional
        Local folder with test scenes.
    data_dir : str, optional
        Persistent data folder (default: "data").

    Returns
    -------
    baseline_nbr : np.ndarray
        Median composite (with anomaly filter applied).
    profile : dict
        Rasterio profile of the composite.

    Raises
    ------
    RuntimeError
        If there are not enough cloud-free scenes in the look-back window.
    """
    aoi_name = aoi["name"]
    paths = nbr_paths(aoi, data_dir)
    logger.info("Building retrospective baseline for AOI '%s'", aoi_name)

    # --- Compute look-back window ---
    # CAMPAIGN_START_DATE = None -> use today as campaign start
    if config.CAMPAIGN_START_DATE is None:
        campaign_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        campaign_start = datetime.fromisoformat(config.CAMPAIGN_START_DATE)
    lookback_start = campaign_start - timedelta(days=config.BASELINE_LOOKBACK_DAYS)

    all_scenes = get_scenes_fn(date_from=lookback_start.isoformat())

    # Keep only scenes in the pre-campaign window
    campaign_str = config.CAMPAIGN_START_DATE
    pre_scenes = [
        s for s in all_scenes
        if s.get("datetime", s.get("date", "")) < campaign_str
    ]
    pre_scenes = _filter_baseline_scenes(pre_scenes)

    logger.info(
        "Baseline: %d pre-campaign scenes found (window %s -> %s)",
        len(pre_scenes), lookback_start.date(), campaign_start.date(),
    )

    if len(pre_scenes) < config.BASELINE_MIN_SCENES:
        raise RuntimeError(
            f"AOI '{aoi_name}': only {len(pre_scenes)} pre-campaign scenes found, "
            f"need at least {config.BASELINE_MIN_SCENES}. "
            f"Increase BASELINE_LOOKBACK_DAYS or check the data."
        )

    # --- Build NBR stack ---
    stack_list = []
    last_profile = None
    for i, scene in enumerate(pre_scenes, 1):
        logger.info(
            "  [%d/%d] downloading baseline scene: %s",
            i, len(pre_scenes), scene["stac_item_id"],
        )
        result = compute_nbr_from_scene(scene, aoi, scene_dir)
        if result is None:
            logger.info("  [%d/%d] skip %s: no valid pixels",
                        i, len(pre_scenes), scene["stac_item_id"])
            continue
        nbr, _nir, _swir, valid_mask, profile = result
        layer = np.full_like(nbr, np.nan)
        layer[valid_mask] = nbr[valid_mask]
        stack_list.append(layer)
        last_profile = profile

    if len(stack_list) < config.BASELINE_MIN_SCENES:
        raise RuntimeError(
            f"AOI '{aoi_name}': only {len(stack_list)} usable scenes "
            f"(after quality filter), need at least {config.BASELINE_MIN_SCENES}."
        )

    stack = np.array(stack_list, dtype="float32")  # (N, H, W)

    # --- MAD anomaly filter: discard anomalously low observations ---
    # For each pixel, compute the median and MAD (Median Absolute Deviation)
    # along the time axis. MAD is robust to outliers (50% breakdown point),
    # unlike standard deviation (0% breakdown point).
    # Ref: Leys et al. 2013, Rousseeuw & Croux 1993.
    pixel_median = _nanmedian_no_allnan_warning(stack, axis=0)
    pixel_mad = _nanmedian_no_allnan_warning(
        np.abs(stack - pixel_median[np.newaxis, :, :]), axis=0,
    )
    # Floor: if MAD < eps, observations are nearly identical -> skip filtering
    pixel_mad = np.maximum(pixel_mad, config.BASELINE_MAD_FLOOR)
    k = config.BASELINE_MAD_K
    threshold = pixel_median - k * pixel_mad
    anomaly_mask = stack < threshold[np.newaxis, :, :]
    n_anomalies = np.count_nonzero(anomaly_mask)
    if n_anomalies > 0:
        stack[anomaly_mask] = np.nan
        logger.info(
            "MAD anomaly filter: removed %d pixel/scene observations (%.2f%% of stack)",
            n_anomalies, 100.0 * n_anomalies / stack.size,
        )

    # --- Median composite ---
    baseline_nbr = _nanmedian_no_allnan_warning(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0)

    # --- Save to disk ---
    paths["baseline"].parent.mkdir(parents=True, exist_ok=True)
    save_nbr(baseline_nbr, last_profile, paths["baseline"])

    # previous_nbr = baseline_nbr (green reference)
    save_nbr(baseline_nbr.copy(), last_profile, paths["previous"])

    logger.info(
        "Baseline complete: %d scenes, median observations/pixel: %.0f",
        len(stack_list),
        np.median(count[count > 0]) if (count > 0).any() else 0,
    )
    return baseline_nbr, last_profile


# ---------------------------------------------------------------------------
# Baseline construction from pre-fetched metadata (operational STAC flow)
# ---------------------------------------------------------------------------

def build_baseline_from_metas(aoi, pre_metas, tile_id, tile_data_dir,
                              scene_dir=None, campaign_start=None):
    """Build the NBR baseline from a pre-fetched list of scene metadata.

    Used in the operational STAC flow where scenes have already been
    retrieved and filtered before this call (by main()).

    Skips the build if a baseline already exists on disk for this tile.

    Parameters
    ----------
    aoi : dict
        AOI dict (from data_io.load_aoi).
    pre_metas : list[dict]
        List of scene metadata (from data_io.query_stac + _filter_scenes).
    tile_id : str
        MGRS tile (e.g. T34SFG).
    tile_data_dir : str
        Persistent data folder for this tile.
    scene_dir : str or Path, optional
        Local TIF folder (None for remote COG reading).
    campaign_start : date, optional
        Campaign start date; saved to tile state if provided.

    Returns
    -------
    bool
        True if the baseline is available (already existed or just built).
    """
    paths = nbr_paths(aoi, tile_data_dir)
    existing, _ = load_nbr(paths["baseline"])
    if existing is not None:
        logger.info("Baseline tile %s already present, skipping build", tile_id)
        return True

    tile_metas = [m for m in pre_metas if _tile_id_from_meta(m) == tile_id]
    if not tile_metas:
        logger.warning("No pre-campaign scenes for tile %s", tile_id)
        return False

    logger.info("Building baseline tile %s: %d candidate scenes", tile_id, len(tile_metas))

    stack_list = []
    last_profile = None
    for meta in tile_metas:
        result = compute_nbr_from_scene(meta, aoi, scene_dir=scene_dir)
        if result is None:
            continue
        nbr, _nir, _swir, valid_mask, prof = result
        layer = np.full_like(nbr, np.nan)
        layer[valid_mask] = nbr[valid_mask]
        stack_list.append(layer)
        last_profile = prof

    if len(stack_list) < config.BASELINE_MIN_SCENES:
        logger.error(
            "Tile %s: only %d usable scenes (need %d) — baseline not built",
            tile_id, len(stack_list), config.BASELINE_MIN_SCENES,
        )
        return False

    stack = np.array(stack_list, dtype="float32")
    pixel_median = _nanmedian_no_allnan_warning(stack, axis=0)
    pixel_mad = _nanmedian_no_allnan_warning(
        np.abs(stack - pixel_median[np.newaxis, :, :]), axis=0,
    )
    pixel_mad = np.maximum(pixel_mad, config.BASELINE_MAD_FLOOR)
    threshold = pixel_median - config.BASELINE_MAD_K * pixel_mad
    anomaly_mask = stack < threshold[np.newaxis, :, :]
    n_anom = int(np.count_nonzero(anomaly_mask))
    if n_anom > 0:
        stack[anomaly_mask] = np.nan
        logger.info("MAD filter tile %s: removed %d pixel/scene observations (%.2f%%)",
                    tile_id, n_anom, 100.0 * n_anom / stack.size)

    baseline_nbr = _nanmedian_no_allnan_warning(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0).astype("float32")
    coverage = float((count > 0).mean())

    Path(tile_data_dir).mkdir(parents=True, exist_ok=True)
    save_nbr(baseline_nbr, last_profile, paths["baseline"])
    save_nbr(baseline_nbr.copy(), last_profile, paths["previous"])

    tile_state = pipeline_state.load_state(tile_data_dir)
    pipeline_state.mark_baseline_built(
        tile_state, len(stack_list), coverage * 100,
        [m["stac_item_id"] for m in tile_metas],
    )
    if campaign_start is not None:
        tile_state["baseline"]["campaign_start"] = str(campaign_start)
    pipeline_state.save_state(tile_state, tile_data_dir)

    logger.info("Baseline tile %s: %d scenes, coverage %.1f%%",
                tile_id, len(stack_list), coverage * 100)
    return True


def _tile_id_from_meta(meta):
    """Extract the MGRS tile ID from stac_item_id (e.g. S2A_T35SMC_... -> T35SMC)."""
    scene_id = meta.get("stac_item_id", "")
    parts = scene_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"
