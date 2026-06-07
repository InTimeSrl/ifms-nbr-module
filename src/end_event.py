"""
end_event.py -- Event closure and final perimeter production.

For each "open" event (managed by events.py) this module:
  1. Reads the accumulator rasters (burnt_count, obs_count, max_dnbr) from the event index.
  2. Computes the confirmed mask:
        confirmed = (burnt_count >= EVENT_MIN_DETECTIONS)
  3. Classifies the final severity on max_dnbr * confirmed.
  4. Saves the final products to:
        <output_dir>/<event_id>/
            severity_final.tif
        <output_dir>/<event_id>.gpkg  (layers: fire_footprint, burnt_final)
  5. Marks the event as "closed" in the index.

Main functions:
    close_event(event_id, output_dir, reason="manual")
    force_close_all_open_events(output_dir, reason="manual_force")
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio.transform
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union

from . import classify, config, data_io, events, postprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single event closure
# ---------------------------------------------------------------------------

def close_event(event_id, output_dir, reason="manual",
                current_nbr=None, valid_mask=None, previous_nbr_path=None,
                scene_date=None, aoi_mask=None):
    """Close an event and produce the final output products.

    Parameters
    ----------
    event_id : str
    output_dir : str or Path
        Tile output folder. Contains `events_index.json`,
        `<event_id>/` (sidecar rasters) and `<event_id>.gpkg` (event GPKG).
    reason : str
        Closure reason.
    current_nbr : np.ndarray, optional
        NBR of the current scene. Used to update previous_nbr.
    valid_mask : np.ndarray (bool), optional
        Valid pixel mask of the current scene. Invalid pixels are written
        as NaN in previous_nbr.
    previous_nbr_path : str or Path, optional
        Path to the previous_nbr.tif file to update in-place.
    scene_date : str, optional
        Date of the closing scene (ISO 8601).

    Returns
    -------
    dict
        Summary of the closed event, or None if the event does not exist.
    """
    state = events.load_index(output_dir).get(event_id)
    if state is None:
        logger.error("Event %s not found in %s", event_id, output_dir)
        return None

    if state.get("status") == "closed":
        logger.info("Event %s already closed, skipping", event_id)
        return None

    paths = events.event_paths(event_id, output_dir)

    # --- Load accumulators ---
    burnt_count, profile = data_io.read_band(str(paths["burnt_count"]))
    obs_count, _ = data_io.read_band(str(paths["obs_count"]))
    max_dnbr, _ = data_io.read_band(str(paths["max_dnbr"]))
    if burnt_count is None or obs_count is None or max_dnbr is None:
        logger.error("Incomplete accumulators for event %s", event_id)
        return None

    burnt_count = burnt_count.astype(np.int16)
    obs_count = obs_count.astype(np.int16)
    max_dnbr = max_dnbr.astype(np.float32)

    # --- Confirmed mask ---

    confirmed = (burnt_count >= config.EVENT_MIN_DETECTIONS)

    # --- Final severity ---
    # Classification on confirmed pixels (burnt_count >= EVENT_MIN_DETECTIONS).
    severity_final = classify.classify_severity(max_dnbr, confirmed)

    # --- Morphological filters on the final product ---
    # Sieve (removes patches < MIN_PATCH_PIXELS) + fill holes (< HOLE_FILL_PIXELS).
    # Same filters applied to per-scene outputs in pipeline.py.
    valid_mask_final = obs_count > 0
    severity_final = postprocess.morphological_filter(severity_final, valid_mask_final)

    # --- Remove burnt clusters below area threshold ---
    # Applied before the FP check: if no cluster exceeds the threshold,
    # the filtered area will be 0 ha and the event will be discarded as a false positive.
    _pixel_res_m = abs(profile["transform"].a)
    if config.OUTPUT_NOISE_FILTER_MIN_AREA_HA > 0:
        severity_final = postprocess.filter_small_clusters(
            severity_final, _pixel_res_m, config.OUTPUT_NOISE_FILTER_MIN_AREA_HA
        )

    # --- Check confirmed area: discard false positives ---
    # Uses area AFTER morphological filters, counting only fire classes (>= 4).
    # Do not use severity_final > 0 because the sieve converts removed pixels
    # to class 3 (Unburned), which survive filter_small_clusters but do not
    # represent real fire.
    pixel_res_check = abs(profile["transform"].a)
    pixel_area_ha_check = (pixel_res_check * pixel_res_check) / 10_000.0
    total_ha_check = round(int((severity_final >= 4).sum()) * pixel_area_ha_check, 2)
    if total_ha_check < config.MIN_ALERT_AREA_HA:
        n_valid = int(np.max(obs_count)) if obs_count is not None else 0
        logger.info(
            "  Event %s -> FALSE POSITIVE | closure reason: %s | "
            "confirmed area %.1f ha < threshold %.1f ha | valid scenes in window: %d",
            event_id, reason, total_ha_check, config.MIN_ALERT_AREA_HA, n_valid,
        )
        events.purge_event(event_id, output_dir)
        return {"event_id": event_id, "closure_reason": "false_positive",
                "purged": True,
                "total_burnt_ha": total_ha_check}

    # --- AOI mask on final products ---
    if aoi_mask is not None and aoi_mask.shape == severity_final.shape:
        severity_final = np.where(aoi_mask, severity_final, 0).astype(severity_final.dtype)
        max_dnbr = np.where(aoi_mask, max_dnbr, np.nan).astype(np.float32)

    # --- Output ---
    sidecar_dir = events.event_dir(event_id, output_dir)
    gpkg_path = Path(output_dir) / f"{event_id}.gpkg"

    # Clip severity_final to the extent of valid (non-nodata) pixels.
    # The original profile covers the full MGRS tile; the crop reduces the file
    # to the burnt area and ensures "Zoom to Layer" in QGIS works correctly.
    _rows, _cols = np.where(severity_final > 0)
    if _rows.size > 0:
        _r0, _r1 = int(_rows.min()), int(_rows.max()) + 1
        _c0, _c1 = int(_cols.min()), int(_cols.max()) + 1
        _sev_crop = severity_final[_r0:_r1, _c0:_c1]
        _t = profile["transform"]
        _west  = _t.c + _c0 * _t.a
        _north = _t.f + _r0 * _t.e
        _profile_crop = dict(profile)
        _profile_crop.update({
            "width":     _c1 - _c0,
            "height":    _r1 - _r0,
            "transform": rasterio.transform.from_origin(_west, _north,
                                                        abs(_t.a), abs(_t.e)),
        })
    else:
        _sev_crop, _profile_crop = severity_final, profile

    severity_path = sidecar_dir / "severity_final.tif"
    data_io.write_geotiff(_sev_crop, _profile_crop, severity_path,
                          dtype="uint8", nodata=0)

    # Reference date: current scene (real event time);
    # fallback to now() if not available.
    if scene_date:
        closed_date = str(scene_date)
    else:
        closed_date = datetime.utcnow().isoformat()

    # --- Vectorisation with event metadata ---
    _alert_scene_id = state.get("alert_scene_id", "")

    meta = {
        "event_id": event_id,
        "detection_date": state.get("alert_date", ""),
        "closed_date": closed_date,
        "closure_reason": reason,
        "n_detection_scenes": int(state.get("n_detection_scenes", 0)),
        "tile": state.get("tile", ""),
        "aoi_ref": state.get("aoi", ""),
        "index_mode": config.INDEX_MODE,
        "index_threshold": (
            config.RBR_THRESHOLD if config.INDEX_MODE == "RBR"
            else config.DNBR_THRESHOLD
        ),
    }
    features = postprocess.vectorize_by_severity(
        severity_final, max_dnbr, profile, meta=meta,
    )
    crs_str = str(profile.get("crs", "")) or None
    data_io.write_geopackage(features, gpkg_path, crs=crs_str,
                             layer_name="burnt_final")

    # --- Summary ---
    pixel_res = abs(profile["transform"].a)
    pixel_area_ha = (pixel_res * pixel_res) / 10_000.0
    by_class = {}
    for class_id, info in config.SEVERITY_CLASSES.items():
        n_pixels = int((severity_final == class_id).sum())
        if n_pixels > 0:
            by_class[int(class_id)] = {
                "label": info["label"],
                "area_ha": round(n_pixels * pixel_area_ha, 2),
                "n_pixels": n_pixels,
            }
    total_ha = total_ha_check  # already computed before the false-positive check

    summary = {
        "event_id": event_id,
        "tile": state.get("tile"),
        "aoi": state.get("aoi"),
        "alert_date": state.get("alert_date"),
        "closed_date": closed_date,
        "closure_reason": reason,
        "n_valid_scenes": int(state.get("n_valid_scenes", 0)),
        "n_detection_scenes": int(state.get("n_detection_scenes", 0)),
        "total_burnt_ha": total_ha,
        "by_class": by_class,
        "outputs": {
            "severity_final": str(severity_path),
            "event_gpkg": str(gpkg_path),
        },
    }

    # Burnt pixel mask: derived from severity_final >= 4 (Low Severity and above,
    # post-sieve and post-hole-fill) to ensure geometric consistency between
    # the footprint and the burnt_final.gpkg vector output.
    raw_burnt = (severity_final >= 4)
    # Filter small clusters (scattered false positives on coast, water,
    # isolated cloud shadows). Keeps only contiguous clusters >= configurable
    # threshold, preventing the hull from "swallowing" pixels far from the
    # true fire perimeter.
    if raw_burnt.any() and config.FOOTPRINT_MIN_CLUSTER_HA > 0:
        _struct = np.ones((3, 3), dtype=np.uint8)  # 8-connessi
        _lab, _n = label(raw_burnt, structure=_struct)
        if _n > 0:
            _counts = np.bincount(_lab.ravel())
            _counts[0] = 0  # background
            _min_px = int(np.ceil(config.FOOTPRINT_MIN_CLUSTER_HA / pixel_area_ha))
            _keep_ids = np.where(_counts >= _min_px)[0]
            _keep_mask = np.isin(_lab, _keep_ids)
            n_dropped = int(raw_burnt.sum() - _keep_mask.sum())
            raw_burnt = _keep_mask

    # Generate the perimeter via dissolve + morphological closing + net expansion.
    # Sequence: dissolve(pixels) -> buffer(+(CLOSING+EXPAND)) -> buffer(-CLOSING) -> simplify(SIMPLIFY)
    # Closing rounds the stepped raster edges; EXPAND produces a net outward expansion of the
    # final perimeter. MultiPolygon parts < FOOTPRINT_MIN_CLUSTER_HA are discarded
    # to remove edge artefacts generated by the morphological closing.
    _fp_geom = None
    if raw_burnt.any():
        try:
            _u8 = raw_burnt.astype("uint8")
            _geoms = [shape(g) for g, v in shapes(_u8, mask=_u8,
                                                   transform=profile["transform"]) if v == 1]
            if _geoms:
                _dissolved = unary_union(_geoms)
                _c = config.FOOTPRINT_CLOSING_M
                _e = config.FOOTPRINT_EXPAND_M
                _s = config.FOOTPRINT_SIMPLIFY_M
                _smoothed = (_dissolved.buffer(_c + _e)
                                       .buffer(-_c)
                                       .simplify(_s, preserve_topology=True))
                _min_m2 = config.FOOTPRINT_MIN_CLUSTER_HA * 10_000
                _parts = (list(_smoothed.geoms)
                          if _smoothed.geom_type == "MultiPolygon" else [_smoothed])
                _kept = [p for p in _parts if p.area >= _min_m2]
                if _kept:
                    _fp_geom = _kept[0] if len(_kept) == 1 else MultiPolygon(_kept)
        except Exception as _fp_err:
            logger.warning("Footprint: error generating perimeter: %s", _fp_err)

    # --- Save fire perimeter to the event GPKG ---
    try:
        _crs_str = str(profile.get("crs", "")) or None
        if _fp_geom is not None:
            _fp_features = [{
                "geometry": _fp_geom.__geo_interface__,
                "properties": {
                    "event_id": event_id,
                    "closed_date": closed_date,
                },
            }]
            data_io.write_geopackage(
                _fp_features, gpkg_path,
                crs=_crs_str, layer_name="fire_footprint",
            )
    except Exception as _fp_exc:
        logger.warning("Error saving fire_footprint.gpkg: %s", _fp_exc)

    # --- Update previous_nbr with the NBR of the closing scene ---
    # Valid pixels   -> current post-fire NBR
    # Invalid pixels -> NaN (excluded from dNBR until the next valid pass)
    if current_nbr is not None and previous_nbr_path is not None:
        try:
            _, prev_profile = data_io.read_band(str(previous_nbr_path))
            cur = current_nbr.astype(np.float32)
            if valid_mask is not None:
                consolidated = np.where(valid_mask, cur, np.nan).astype(np.float32)
                n_invalid = int((~valid_mask).sum())
            else:
                consolidated = cur.astype(np.float32)
                n_invalid = 0
            data_io.write_geotiff(
                consolidated, prev_profile,
                str(previous_nbr_path), dtype="float32",
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Error updating previous_nbr for event %s: %s",
                event_id, exc,
            )

    # --- Mark event as closed ---
    events.mark_closed(event_id, output_dir, reason=reason, closed_date=closed_date)

    _class_summary = "  ".join(
        f"{config.SEVERITY_CLASSES[k]['abbr']}={v['area_ha']}ha"
        for k, v in by_class.items()
        if k >= 4  # fire classes only (Low ... High); ER-H/ER-L/Unburned excluded
    )
    logger.info(
        "  Event %s closed (%s) | %.1f ha [%s] | %d detection scenes | output=%s",
        event_id, reason, total_ha, _class_summary, summary["n_detection_scenes"], gpkg_path,
    )
    return summary


def force_close_all_open_events(output_dir, reason="manual_force"):
    """Close all events currently in 'open' state.

    Useful at end of campaign or for emergency manual closures.

    Returns
    -------
    list[dict]
        List of summaries for the closed events.
    """
    event_ids = events.list_active_events(output_dir)
    summaries = []
    for eid in event_ids:
        s = close_event(eid, output_dir, reason=reason)
        if s is not None:
            summaries.append(s)
    logger.info("force_close_all_open_events: closed %d events", len(summaries))
    return summaries
