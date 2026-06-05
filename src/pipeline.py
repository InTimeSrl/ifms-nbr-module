"""
pipeline.py -- Orchestratore del monitoraggio continuo aree bruciate.

Logica per evento:
  - baseline_nbr  : composite pre-campagna (fisso), da baseline.py
  - previous_nbr  : inizia = baseline, evolve a ogni scena valida (anche burnt
                    sui pixel non bruciati)
  - eventi        : gestiti da events.py. Ogni cluster bruciato >= soglia apre
                    un evento o aggiorna un evento attivo (overlap di bbox).
                    Ogni scena valida contribuisce a obs_count degli eventi
                    attivi del tile (e a burnt_count su quelli toccati).
  - chiusura      : end_event.close_event(...) quando un evento raggiunge
                    EVENT_WINDOW_SCENES o EVENT_TIMEOUT_DAYS.
"""

import argparse
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyproj
import rasterio.transform as _rt
import rasterio.windows as _rwin
from rasterio.features import geometry_mask as _geometry_mask
from scipy.ndimage import binary_dilation as _binary_dilation, distance_transform_edt as _distance_transform_edt
from shapely.geometry import mapping as _shapely_mapping
from shapely.ops import transform as _shp_transform

from . import config
from . import data_io
from . import baseline
from . import indices, classify, postprocess
from . import events, end_event
from . import state as pipeline_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtri pre-processing a livello di scena
# ---------------------------------------------------------------------------

def _filter_scenes(scenes, tile_id=""):
    """Filtra scene per qualita': cloud_cover e processing_baseline.

    Parameters
    ----------
    scenes : list[dict]
        Scene candidate (gia' filtrate per data dal chiamante).
    tile_id : str, optional
        Usato solo nel messaggio di log.

    Returns
    -------
    list[dict]
        Scene valide da processare.
    """
    valid = []
    for scene in scenes:
        scene_id = scene["stac_item_id"]

        # Cloud cover troppo alta? (supporta sia "eo_cloud_cover" che "cloud_cover")
        cc = scene.get("eo_cloud_cover") if "eo_cloud_cover" in scene else scene.get("cloud_cover")
        if cc is not None and cc > config.MAX_CLOUD_COVER_PCT:
            logger.debug("Skip %s: cloud_cover=%.1f%%", scene_id, cc)
            continue

        # Processing baseline troppo vecchia? (supporta sia "s2_processing_baseline" che "processing_baseline")
        pb = scene.get("s2_processing_baseline") if "s2_processing_baseline" in scene else scene.get("processing_baseline", "99.99")
        if pb < config.MIN_PROCESSING_BASELINE:
            logger.debug("Skip %s: baseline=%s", scene_id, pb)
            continue

        valid.append(scene)

    logger.info(
        "Tile %s: %d scene candidate, %d valide per qualita'",
        tile_id, len(scenes), len(valid),
    )
    return valid


def _get_tile_id(scene):
    """Estrae il tile MGRS dallo STAC item id (es. S2A_T35SMC_... -> T35SMC)."""
    scene_id = scene.get("stac_item_id", "")
    parts = scene_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

def process_scene(scene, aoi, scene_dir=None, previous_nbr=None):
    """Processa una singola scena per un'AOI.

    Parameters
    ----------
    scene : dict
        Metadati della scena.
    aoi : dict
        AOI dict (da data_io.load_aoi).
    scene_dir : str o Path, optional
        Cartella locale dei TIF. Se None, legge da remoto (COG via VFS).
    previous_nbr : np.ndarray
        NBR dell'ultima scena "pulita" (baseline operativa).

    Returns
    -------
    result : dict o None
        - "nbr": array NBR corrente
        - "valid_mask": maschera pixel validi
        - "profile": profilo rasterio
        - "dnbr": array dNBR (previous - current)
        - "fire_detected": bool
        - "severity": array classificazione (solo se fire_detected)

        None se la scena e' priva di pixel validi (caso degenere).
    """
    scene_id = scene["stac_item_id"]

    result = baseline.compute_nbr_from_scene(scene, aoi, scene_dir)
    if result is None:
        logger.warning("Scena %s: nessun pixel valido, skip", scene_id)
        return None

    nbr, nir, swir, valid_mask, profile = result
    out = {"nbr": nbr, "valid_mask": valid_mask, "profile": profile}

    if config.INDEX_MODE == "RBR":
        delta = indices.compute_rbr(previous_nbr, nbr)
        _index_label = "RBR"
        _threshold = config.RBR_THRESHOLD
    else:
        delta = indices.compute_dnbr(previous_nbr, nbr)
        _index_label = "dNBR"
        _threshold = config.DNBR_THRESHOLD

    out["dnbr"] = delta  # chiave fissa per compatibilita' output; contiene dNBR o RBR

    # Tripla soglia: indice alto + NIR basso + SWIR2 minimo (filtra acqua non coperta da SCL).
    burnt_mask = (delta > _threshold) & (nir < config.NIR_MAX_BURNT) & (swir > config.SWIR2_MIN_BURNT) & valid_mask
    out["burnt_mask"] = burnt_mask

    pixel_area_ha = 0.04  # 20 m × 20 m = 0.04 ha
    burnt_area_ha = burnt_mask.sum() * pixel_area_ha

    _delta_valid = delta[valid_mask]
    max_index_val = float(np.nanmax(_delta_valid)) if np.any(~np.isnan(_delta_valid)) else 0.0
    out["burnt_area_ha"] = burnt_area_ha
    out["max_index_val"] = max_index_val
    out["index_label"] = _index_label
    out["threshold"] = _threshold

    if burnt_area_ha < config.MIN_ALERT_AREA_HA:
        out["fire_detected"] = False
        return out

    out["fire_detected"] = True

    severity = classify.classify_severity(delta, valid_mask)
    severity_clean = postprocess.morphological_filter(severity, valid_mask)

    out["severity"] = severity_clean

    return out


def process_aoi(aoi, scene_dir=None, stac_client=None, output_dir="output",
                data_dir="data", scenes=None):
    """Esegue la pipeline completa per un'AOI: baseline, watermark, loop scene,
    gestione eventi, output raster/vettore per ogni tile MGRS rilevato.

    Parameters
    ----------
    scenes : list[dict], optional
        Scene pre-caricate (es. da query STAC esterna). Se None, vengono
        recuperate tramite get_scenes().
    """
    aoi_name = aoi["name"]
    logger.info("=== Inizio processamento AOI: %s ===", aoi_name)

    # Recupera tutte le scene e determina i tile da processare.
    all_scenes = scenes if scenes is not None else data_io.get_scenes(
        aoi, scene_dir=scene_dir, stac_client=stac_client
    )
    tile_ids = sorted({_get_tile_id(s) for s in all_scenes if _get_tile_id(s) != "unknown"})

    if not tile_ids:
        logger.info("AOI '%s': nessun tile valido trovato nelle scene", aoi_name)
        return {"aoi": aoi_name, "scenes_processed": 0, "alerts": 0, "tiles": []}

    total_scenes_processed = 0
    total_alerts = 0
    total_fp_count = 0
    processed_tiles = []

    for tile_id in tile_ids:
        tile_data_dir = str(Path(data_dir) / tile_id)
        tile_output_dir = str(Path(output_dir) / tile_id)
        logger.info("--- AOI '%s' | Tile %s ---", aoi_name, tile_id)

        # Watermark: datetime ISO 8601 dell'ultima scena processata.
        # Solo scene con datetime > watermark vengono incluse nel loop.
        tile_state = pipeline_state.load_state(tile_data_dir)
        watermark = pipeline_state.get_watermark(tile_state)
        if watermark:
            logger.info("Tile %s: watermark=%s (ultime scene gia' processate)",
                        tile_id, watermark)

        paths = baseline.nbr_paths(aoi, tile_data_dir)
        baseline_nbr, baseline_profile = baseline.load_nbr(paths["baseline"])
        previous_nbr, _ = baseline.load_nbr(paths["previous"])

        # Baseline assente: costruiscila retrospettivamente per tile.
        if baseline_nbr is None:
            def _get_scenes_for_tile(date_from=None, _tile_id=tile_id):
                scenes = data_io.get_scenes(
                    aoi, scene_dir=scene_dir, stac_client=stac_client, date_from=date_from,
                )
                return [s for s in scenes if _get_tile_id(s) == _tile_id]

            baseline_nbr, baseline_profile = baseline.build_baseline(
                aoi, get_scenes_fn=_get_scenes_for_tile,
                scene_dir=scene_dir, data_dir=tile_data_dir,
            )
            previous_nbr = baseline_nbr.copy()

        # Se previous_nbr non esiste (caso anomalo), inizializza da baseline
        if previous_nbr is None:
            previous_nbr = baseline_nbr.copy()
            baseline.save_nbr(previous_nbr, baseline_profile, paths["previous"])

        # Maschera AOI rasterizzata sul grid tile (CRS riproiettato): limita la
        # pipeline ai pixel interni all'AOI per tutta la durata del loop scene.
        _tile_aoi_mask = None
        _aoi_geom_raw = aoi.get("geometry")
        if _aoi_geom_raw is not None and baseline_profile is not None:
            try:
                _aoi_src_crs = pyproj.CRS(aoi.get("crs", "EPSG:4326"))
                _tile_crs = pyproj.CRS(str(baseline_profile["crs"]))
                if not _aoi_src_crs.equals(_tile_crs):
                    _proj_fn = pyproj.Transformer.from_crs(
                        _aoi_src_crs, _tile_crs, always_xy=True
                    ).transform
                    _aoi_proj = _shp_transform(_proj_fn, _aoi_geom_raw)
                else:
                    _aoi_proj = _aoi_geom_raw
                _tile_aoi_mask = _geometry_mask(
                    [_shapely_mapping(_aoi_proj)],
                    transform=baseline_profile["transform"],
                    invert=True,   # True = dentro AOI
                    out_shape=(baseline_profile["height"], baseline_profile["width"]),
                )
                logger.info("Tile %s: maschera AOI applicata (%.1f%% pixel inclusi)",
                            tile_id, 100.0 * _tile_aoi_mask.sum() / _tile_aoi_mask.size)
            except Exception as _e:
                logger.warning("Tile %s: maschera AOI non calcolata: %s", tile_id, _e)

        # Bounding box AOI sul grid tile: ritaglio degli output.
        _aoi_crop_window = None
        if _tile_aoi_mask is not None:
            _rows_in = np.where(_tile_aoi_mask.any(axis=1))[0]
            _cols_in = np.where(_tile_aoi_mask.any(axis=0))[0]
            if len(_rows_in) > 0 and len(_cols_in) > 0:
                _cr0, _cr1 = int(_rows_in[0]), int(_rows_in[-1]) + 1
                _cc0, _cc1 = int(_cols_in[0]), int(_cols_in[-1]) + 1
                _aoi_crop_window = (_cr0, _cc0, _cr1 - _cr0, _cc1 - _cc0)

        # Azzera pixel fuori AOI a NaN nella baseline (in memoria e su disco).
        if _tile_aoi_mask is not None and baseline_profile is not None:
            if baseline_nbr is not None and baseline_nbr.shape == _tile_aoi_mask.shape:
                baseline_nbr = np.where(_tile_aoi_mask, baseline_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(baseline_nbr, baseline_profile, paths["baseline"])
            if previous_nbr is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(previous_nbr, baseline_profile, paths["previous"])

        # Resume dopo crash: riapplica il baseline pre-fire congelato per eventi attivi.
        if previous_nbr is not None:
            _init_active_eids = events.get_active_events(
                tile_output_dir, tile=tile_id, aoi=aoi_name
            )
            for _init_eid in _init_active_eids:
                _frozen_path = Path(tile_data_dir) / f"frozen_nbr_{_init_eid}_temp.tif"
                if _frozen_path.exists():
                    _frozen_arr, _ = data_io.read_band(str(_frozen_path))
                    if _frozen_arr is not None and _frozen_arr.shape == previous_nbr.shape:
                        _fp_mask = events.load_footprint_mask(_init_eid, tile_output_dir)
                        if _fp_mask is not None and _fp_mask.any():
                            _sep = getattr(config, "EVENT_BASELINE_BUFFER_PX", 0) or 0
                            _zone = (
                                _binary_dilation(_fp_mask, iterations=int(_sep))
                                if _sep > 0 else _fp_mask
                            )
                            previous_nbr = np.where(_zone, _frozen_arr, previous_nbr)
                            logger.info(
                                "Tile %s: baseline pre-fire ripristinato per %s (frozen_nbr_temp)",
                                tile_id, _init_eid,
                            )

        # Seleziona scene per tile successive al watermark ("" se scene esterne).
        if scenes is not None:
            date_floor = watermark if watermark is not None else ""
        else:
            date_floor = watermark if watermark is not None else config.CAMPAIGN_START_DATE
        tile_scenes = [
            s for s in all_scenes
            if _get_tile_id(s) == tile_id
            and s.get("datetime", s.get("date", "")) > date_floor
        ]
        tile_scenes = _filter_scenes(tile_scenes, tile_id=tile_id)
        tile_scenes.sort(key=lambda s: s.get("datetime", s.get("date", "")))  # ordine cronologico

        if not tile_scenes:
            logger.info("Nessuna nuova scena per AOI '%s' sul tile %s", aoi_name, tile_id)
            continue

        tile_scenes_processed = 0
        tile_alerts = 0       # eventi aperti in questo run
        tile_fp_count = 0     # eventi chiusi come falso positivo in questo run

        for scene in tile_scenes:
            scene_id = scene["stac_item_id"]
            logger.info("Processo scena: %s", scene_id)

            try:
                result = process_scene(scene, aoi, scene_dir, previous_nbr=previous_nbr)
            except Exception as exc:  # noqa: BLE001  # errori di rete/IO transitori
                logger.warning("Scena %s saltata per errore di rete/IO: %s", scene_id, exc)
                continue
            if result is None:
                continue

            tile_scenes_processed += 1

            # Ricalibra previous_nbr sulla prima scena SCL-ok dopo chiusura evento su SCL-fail.
            if tile_state.get("needs_recalibrate"):
                _rc_nbr = result["nbr"]
                _rc_vm  = result["valid_mask"]
                if _tile_aoi_mask is not None and _tile_aoi_mask.shape == _rc_vm.shape:
                    _rc_vm = _rc_vm & _tile_aoi_mask
                # Rimanda se SCL non sufficiente (scena nuvolosa → FP fenologici).
                _rc_land_px = int(np.isfinite(baseline_nbr).sum()) if baseline_nbr is not None else 0
                _rc_denom   = _rc_land_px if _rc_land_px > 0 else _rc_vm.size
                _rc_valid_pct = 100.0 * float(_rc_vm.sum()) / _rc_denom
                if _rc_valid_pct < config.SCENE_VALID_SCL_PCT:
                    logger.info("  Ricalibrazione rimandata: scena %s SCL %.1f%% < %.0f%%",
                                scene_id, _rc_valid_pct, config.SCENE_VALID_SCL_PCT)
                    pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
                    pipeline_state.save_state(tile_state, tile_data_dir)
                    continue
                previous_nbr = np.where(_rc_vm, _rc_nbr, previous_nbr)
                if _tile_aoi_mask is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                    previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(previous_nbr, result["profile"], paths["previous"])
                del tile_state["needs_recalibrate"]
                pipeline_state.save_state(tile_state, tile_data_dir)
                logger.info("  Ricalibrazione previous_nbr completata (scena %s, SCL %.1f%%)",
                            scene_id, _rc_valid_pct)
                pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
                pipeline_state.save_state(tile_state, tile_data_dir)
                continue
            # ---

            nbr = result["nbr"]
            valid_mask = result["valid_mask"]
            profile = result["profile"]
            dnbr = result["dnbr"]
            burnt_mask = result.get("burnt_mask", np.zeros_like(valid_mask))

            if _tile_aoi_mask is not None:
                _m = _tile_aoi_mask
                if _m.shape != valid_mask.shape:
                    # Shape mismatch anomalo (tile MGRS uniformi non dovrebbero divergere).
                    logger.warning("Tile %s: shape AOI mask %s != valid_mask %s, skip",
                                   tile_id, _m.shape, valid_mask.shape)
                else:
                    valid_mask = valid_mask & _m
                    burnt_mask = burnt_mask & _m
            _threshold = result.get("threshold", config.DNBR_THRESHOLD)
            # Variabili log riepilogo per-scena.
            _log_area_ok     = result.get("fire_detected", False)
            _log_area_ha     = result.get("burnt_area_ha", 0.0)
            _log_idx_max     = result.get("max_index_val", 0.0)
            _log_idx_label   = result.get("index_label", "dNBR")
            _log_cluster_ok  = None   # None=SKIP, True=pass, False=fail
            _log_cluster_ha  = None
            _log_total_ha    = None
            _log_compact_pct = None

            _land_px = int(np.isfinite(baseline_nbr).sum()) if baseline_nbr is not None else 0
            _denom = _land_px if _land_px > 0 else valid_mask.size
            valid_pct = 100.0 * valid_mask.sum() / _denom

            # Lista eventi attivi: caricata prima del check SCL (serve anche in SCL-fail).
            active_eids = events.get_active_events(tile_output_dir, tile=tile_id, aoi=aoi_name)
            scene_date = scene.get("datetime", scene.get("date", ""))

            if valid_pct < config.SCENE_VALID_SCL_PCT:
                logger.info("  SCL:       %.1f%%  FAIL (no-valid)", valid_pct)
                logger.info("  area:      SKIP")
                logger.info("  cluster:   SKIP")
                if active_eids:
                    logger.info("  => EVENTI IN CORSO: %s  (scena non contata - SCL)", active_eids)
                else:
                    logger.info("  => NO ACTIVE EVENT")
                # Timeout su SCL-fail: should_close() controlla solo i giorni, non i dati NBR.
                for _eid in list(active_eids):
                    _close, _reason = events.should_close(_eid, tile_output_dir, current_date=scene_date)
                    if not _close:
                        continue
                    logger.info("  => TIMEOUT su scena non valida: chiudo %s (%s)", _eid, _reason)
                    _summary = end_event.close_event(
                        _eid, tile_output_dir, reason=_reason,
                        scene_date=scene_date,
                    )
                    _is_fp = _summary and _summary.get("closure_reason") == "false_positive"
                    if _is_fp:
                        _gpkg_fp = Path(tile_output_dir) / f"{_eid}.gpkg"
                        if _gpkg_fp.exists():
                            _gpkg_fp.unlink(missing_ok=True)
                        logger.info("  Output rimossi (false_positive): %s", _eid)
                        tile_fp_count += 1
                    _frozen_del = Path(tile_data_dir) / f"frozen_nbr_{_eid}_temp.tif"
                    _frozen_del.unlink(missing_ok=True)
                    tile_state["needs_recalibrate"] = True
                    logger.info("  => needs_recalibrate=True (chiusura %s su SCL-fail)", _eid)
                pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
                pipeline_state.save_state(tile_state, tile_data_dir)
                continue

            pixel_res = abs(profile["transform"].a)
            pixel_area_ha = (pixel_res * pixel_res) / 10_000.0
            _min_compactness = getattr(config, "MIN_CLUSTER_COMPACTNESS", None)
            _log_verdicts = []

            # Pre-close timeout: chiudi eventi scaduti prima della detection.
            if active_eids:
                _timeout_closed = []
                for _eid in list(active_eids):
                    _close, _reason = events.should_close(_eid, tile_output_dir, current_date=scene_date)
                    if not _close:
                        continue
                    logger.info("  => Pre-close timeout: chiudo %s (%s) prima della detection", _eid, _reason)
                    _summary = end_event.close_event(
                        _eid, tile_output_dir, reason=_reason,
                        current_nbr=nbr, valid_mask=valid_mask,
                        previous_nbr_path=paths["previous"],
                        scene_date=scene_date,
                        aoi_mask=_tile_aoi_mask,
                    )
                    _is_fp = _summary and _summary.get("closure_reason") == "false_positive"
                    if _is_fp:
                        _gpkg_fp = Path(tile_output_dir) / f"{_eid}.gpkg"
                        if _gpkg_fp.exists():
                            _gpkg_fp.unlink(missing_ok=True)
                        logger.info("  Output rimossi (false_positive): %s", _eid)
                        tile_fp_count += 1
                    _frozen_del = Path(tile_data_dir) / f"frozen_nbr_{_eid}_temp.tif"
                    _frozen_del.unlink(missing_ok=True)
                    active_eids.remove(_eid)
                    _timeout_closed.append(_eid)
                if _timeout_closed and not active_eids:
                    # Tutti chiusi: ricarica previous_nbr e salta detection.
                    previous_nbr, _ = baseline.load_nbr(paths["previous"])
                    pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
                    pipeline_state.save_state(tile_state, tile_data_dir)
                    continue
            # ---

            if active_eids:
                # EVENTI ATTIVI: nearest-footprint, poi pixel orfani.
                # burnt_mask include filtro NIR: evita accumulo da essiccamento fenologico.
                burnt_for_accum = burnt_mask.copy()
                remaining = burnt_mask.copy()

                # Nearest-footprint (Voronoi): pixel → evento più vicino entro
                # MAX_INITIAL_MERGE_DISTANCE_KM; quelli oltre restano in 'remaining'.
                footprints = {eid: events.load_footprint_mask(eid, tile_output_dir)
                              for eid in active_eids}

                _max_dist_px = (config.MAX_INITIAL_MERGE_DISTANCE_KM * 1000.0) / pixel_res

                if len(active_eids) == 1:
                    fp = footprints[active_eids[0]]
                    _dist_single = (
                        _distance_transform_edt(~fp).astype(np.float64)
                        if (fp is not None and fp.any())
                        else np.full(burnt_for_accum.shape, np.inf, dtype=np.float64)
                    )
                    _near_single = _dist_single <= _max_dist_px
                    assignments = {active_eids[0]: burnt_for_accum & _near_single}
                else:
                    dist_stack = np.stack([
                        _distance_transform_edt(~fp).astype(np.float64)
                        if (fp is not None and fp.any())
                        else np.full(burnt_for_accum.shape, np.inf, dtype=np.float64)
                        for fp in (footprints[e] for e in active_eids)
                    ], axis=0)
                    _min_dist = dist_stack.min(axis=0)
                    _near = _min_dist <= _max_dist_px
                    nearest_idx = np.argmin(dist_stack, axis=0)
                    assignments = {
                        eid: burnt_for_accum & _near & (nearest_idx == i)
                        for i, eid in enumerate(active_eids)
                    }

                for eid in active_eids:
                    assigned = assignments[eid]

                    _ev_n_valid = events.update_event(
                        eid, assigned, dnbr, valid_mask, profile, scene, tile_output_dir,
                    )
                    remaining = remaining & ~assigned

                    if assigned.any():
                        data_io.save_scene_outputs(
                            result, scene, aoi, tile_output_dir, scene_dir,
                            event_ids=eid, tile_id=tile_id,
                            scene_ts=events._format_scene_ts(scene_date),
                            aoi_crop=_aoi_crop_window,
                            aoi_mask=_tile_aoi_mask,
                        )
                    _log_verdicts.append(
                        f"EVENTO IN CORSO: {eid}  (scena {_ev_n_valid}/{config.EVENT_WINDOW_SCENES})"
                    )

                # Verifica pixel orfani (incendio nuovo o frammentazione)
                if remaining.any():
                    _remaining_ha = float(remaining.sum()) * pixel_area_ha
                    _largest_rem = events.largest_cluster_area_ha(remaining, pixel_area_ha)
                    _compact_rem = (_largest_rem / _remaining_ha) if _remaining_ha > 0 else 0.0
                    _compact_ok_rem = not _min_compactness or _compact_rem >= _min_compactness
                    if _remaining_ha >= config.MIN_ALERT_AREA_HA and _compact_ok_rem:
                        new_clusters = events.find_clusters(remaining, pixel_area_ha)
                        for cl in new_clusters:
                            new_eid = events.open_event(
                                cl["mask"], dnbr, valid_mask, profile, scene,
                                tile_output_dir, tile=tile_id, aoi=aoi_name,
                            )
                            active_eids.append(new_eid)
                            tile_alerts += 1
                            data_io.save_scene_outputs(
                                result, scene, aoi, tile_output_dir, scene_dir,
                                event_ids=new_eid, tile_id=tile_id,
                                scene_ts=events._format_scene_ts(scene_date),
                                aoi_crop=_aoi_crop_window,
                                aoi_mask=_tile_aoi_mask,
                            )
                            _log_verdicts.append(
                                f"ALERT: aperto {new_eid} (pixel orfani  "
                                f"{cl['area_ha']:.1f} ha  scena 1/{config.EVENT_WINDOW_SCENES})"
                            )

                _log_cluster_ok  = None

            else:
                # NESSUN EVENTO ATTIVO: applica filtri e apri per cluster.
                largest_cluster_ha = events.largest_cluster_area_ha(burnt_mask, pixel_area_ha)
                _total_burnt_ha = float(burnt_mask.sum()) * pixel_area_ha
                _compactness = (largest_cluster_ha / _total_burnt_ha) if _total_burnt_ha > 0 else 0.0
                _compact_ok = not _min_compactness or _compactness >= _min_compactness

                _log_cluster_ok  = (largest_cluster_ha >= config.MIN_ALERT_AREA_HA) and _compact_ok
                _log_cluster_ha  = largest_cluster_ha
                _log_total_ha    = _total_burnt_ha
                _log_compact_pct = _compactness * 100

                if (
                    result.get("fire_detected")
                    and largest_cluster_ha >= config.MIN_ALERT_AREA_HA
                    and _compact_ok
                ):
                    new_clusters = events.find_clusters(burnt_mask, pixel_area_ha)
                    if not new_clusters:
                        # Fallback: nessun cluster da solidity → apri su intera maschera.
                        new_clusters = [{"mask": burnt_mask, "area_ha": largest_cluster_ha}]
                    # Fonde cluster entro MAX_INITIAL_MERGE_DISTANCE_KM; oltre → eventi separati.
                    _tr = profile["transform"]
                    _merge_thr_m = config.MAX_INITIAL_MERGE_DISTANCE_KM * 1000.0
                    _cl_xy = []
                    for _cln in new_clusters:
                        _ys, _xs = np.where(_cln["mask"])
                        _x, _y = _rt.xy(_tr, float(_ys.mean()), float(_xs.mean()))
                        _cl_xy.append((_x, _y))
                    # Union-Find: raggruppa cluster entro soglia
                    _uf = list(range(len(new_clusters)))
                    def _uf_find(x):
                        while _uf[x] != x:
                            _uf[x] = _uf[_uf[x]]; x = _uf[x]
                        return x
                    for _i in range(len(new_clusters)):
                        for _j in range(_i + 1, len(new_clusters)):
                            _dx = _cl_xy[_i][0] - _cl_xy[_j][0]
                            _dy = _cl_xy[_i][1] - _cl_xy[_j][1]
                            if (_dx**2 + _dy**2)**0.5 <= _merge_thr_m:
                                _ri, _rj = _uf_find(_i), _uf_find(_j)
                                if _ri != _rj:
                                    _uf[_rj] = _ri
                    _groups: dict = {}
                    for _i, _cln in enumerate(new_clusters):
                        _root = _uf_find(_i)
                        if _root in _groups:
                            _groups[_root]["mask"] |= _cln["mask"]
                            _groups[_root]["n"] += 1
                        else:
                            _groups[_root] = {"mask": _cln["mask"].copy(), "n": 1}
                    for _gmeta in _groups.values():
                        _gmask_ha = float(_gmeta["mask"].sum()) * pixel_area_ha
                        new_eid = events.open_event(
                            _gmeta["mask"], dnbr, valid_mask, profile, scene,
                            tile_output_dir, tile=tile_id, aoi=aoi_name,
                        )
                        active_eids.append(new_eid)
                        tile_alerts += 1
                        data_io.save_scene_outputs(
                            result, scene, aoi, tile_output_dir, scene_dir,
                            event_ids=new_eid, tile_id=tile_id,
                            scene_ts=events._format_scene_ts(scene_date),
                            aoi_crop=_aoi_crop_window,
                            aoi_mask=_tile_aoi_mask,
                        )
                        _log_verdicts.append(
                            f"ALERT: aperto {new_eid} (scena 1/{config.EVENT_WINDOW_SCENES}  "
                            f"{_gmask_ha:.1f} ha"
                            + (f"  {_gmeta['n']} cluster fusi" if _gmeta['n'] > 1 else "")
                            + ")"
                        )
                if not _log_verdicts:
                    _log_verdicts.append("NO ACTIVE EVENT")

            # --- Riepilogo per-scena ---
            logger.info("  SCL:       %.1f%%  OK", valid_pct)
            if _log_cluster_ok is None:
                logger.info("  area:      %.2f ha  (%s max=%.3f)",
                            _log_area_ha, _log_idx_label, _log_idx_max)
            else:
                _area_str = (
                    f"{_log_area_ha:.2f} ha  ({_log_idx_label} max={_log_idx_max:.3f})  "
                    + ("OK" if _log_area_ok else f"FAIL (< {config.MIN_ALERT_AREA_HA:.1f} ha)")
                )
                logger.info("  area:      %s", _area_str)
                if not _log_area_ok:
                    logger.info("  cluster:   SKIP (area < soglia)")
                elif _log_cluster_ok:
                    logger.info(
                        "  cluster:   %.1f/%.1f ha = %.1f%%  OK",
                        _log_cluster_ha, _log_total_ha, _log_compact_pct,
                    )
                else:
                    logger.info(
                        "  cluster:   %.1f/%.1f ha = %.1f%%  FAIL (< %.0f%%)",
                        _log_cluster_ha, _log_total_ha, _log_compact_pct,
                        (_min_compactness or 0) * 100,
                    )
            for _verdict in _log_verdicts:
                if _verdict.startswith("ALERT"):
                    logger.warning("  => %s", _verdict)
                else:
                    logger.info("  => %s", _verdict)

            # Senza eventi attivi: aggiorna previous_nbr (anche pixel edge della cicatrice,
            # altrimenti loop infinito). Con eventi attivi: congelato fino a close_event.
            if not active_eids:
                previous_nbr = np.where(valid_mask, nbr, previous_nbr)
                if _tile_aoi_mask is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                    previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(previous_nbr, profile, paths["previous"])

            for eid in list(active_eids):  # copia per poter rimuovere durante iterazione
                close, reason = events.should_close(eid, tile_output_dir, current_date=scene_date)
                if not close:
                    continue
                _summary = end_event.close_event(
                    eid, tile_output_dir, reason=reason,
                    current_nbr=nbr, valid_mask=valid_mask,
                    previous_nbr_path=paths["previous"],
                    scene_date=scene_date,
                    aoi_mask=_tile_aoi_mask,
                )
                _is_fp = _summary and _summary.get("closure_reason") == "false_positive"
                if _is_fp:
                    _gpkg_fp = Path(tile_output_dir) / f"{eid}.gpkg"
                    if _gpkg_fp.exists():
                        _gpkg_fp.unlink(missing_ok=True)
                    logger.info("  Output rimossi (false_positive): %s", eid)
                    tile_fp_count += 1
                active_eids.remove(eid)
                _frozen_del = Path(tile_data_dir) / f"frozen_nbr_{eid}_temp.tif"
                _frozen_del.unlink(missing_ok=True)
                if _is_fp:
                    # FP: previous_nbr era frozen; aggiornalo per evitare loop di FP
                    # causati dall'essiccamento stagionale accumulato nella finestra evento.
                    if not active_eids:  # aggiorna solo se non ci sono altri eventi attivi
                        previous_nbr = np.where(valid_mask, nbr, previous_nbr)
                        if _tile_aoi_mask is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                            previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                        baseline.save_nbr(previous_nbr, profile, paths["previous"])
                    logger.info("  Baseline aggiornata post false_positive (era frozen %d scene)",
                                _summary.get("n_valid_scenes", 0) if _summary else 0)
                else:
                    # Ricarica previous_nbr da disco (close_event ha incorporato la cicatrice)
                    # per evitare ri-detection loop. Poi ripristina pre-fire per eventi ancora attivi.
                    _reloaded, _ = data_io.read_band(str(paths["previous"]))
                    if _reloaded is not None:
                        _new_prev = _reloaded.astype(np.float32)
                        for _still_eid in active_eids:  # eid chiuso gia' rimosso
                            _fp = events.load_footprint_mask(_still_eid, tile_output_dir)
                            if _fp is not None and _fp.any():
                                _sep = getattr(config, "EVENT_BASELINE_BUFFER_PX", 0) or 0
                                _zone = (_binary_dilation(_fp, iterations=int(_sep))
                                         if _sep > 0 else _fp)
                                _new_prev = np.where(_zone, previous_nbr, _new_prev)
                        previous_nbr = _new_prev
                        # Aggiorna frozen_nbr_temp per resume dopo crash.
                        for _still_eid in active_eids:
                            _frozen_path = Path(tile_data_dir) / f"frozen_nbr_{_still_eid}_temp.tif"
                            baseline.save_nbr(previous_nbr, profile, str(_frozen_path))
                    else:
                        previous_nbr = np.where(valid_mask, nbr, previous_nbr)

            pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
            pipeline_state.save_state(tile_state, tile_data_dir)

        _tile_confirmed = tile_alerts - tile_fp_count
        _fp_str = f", {tile_fp_count} falsi positivi" if tile_fp_count else ""
        logger.info(
            "Tile %s completato: %d scene processate, %d alert confermati%s",
            tile_id, tile_scenes_processed, _tile_confirmed, _fp_str,
        )
        total_scenes_processed += tile_scenes_processed
        total_alerts += tile_alerts
        total_fp_count += tile_fp_count
        processed_tiles.append(tile_id)

    _aoi_confirmed = total_alerts - total_fp_count
    _aoi_fp_str = f", {total_fp_count} falsi positivi" if total_fp_count else ""
    # Conta eventi ancora aperti su tutti i tile al termine del run
    _total_open = sum(
        len(events.list_active_events(str(Path(output_dir) / tid), tile=tid, aoi=aoi_name))
        for tid in processed_tiles
    )
    _open_str = f", {_total_open} eventi in corso" if _total_open else ""
    logger.info(
        "=== AOI '%s' completata: %d tile, %d scene processate, %d alert confermati%s%s ===",
        aoi_name, len(processed_tiles), total_scenes_processed, _aoi_confirmed, _aoi_fp_str, _open_str,
    )
    return {
        "aoi": aoi_name,
        "tiles": processed_tiles,
        "scenes_processed": total_scenes_processed,
        "alerts": _aoi_confirmed,
        "open_events": _total_open,
    }


# ---------------------------------------------------------------------------
# Entry point operativo
# ---------------------------------------------------------------------------

def _build_baseline_from_metas(aoi, pre_metas, tile_id, tile_data_dir,
                               campaign_start=None):
    """Wrapper di compatibilità — delegato a baseline.build_baseline_from_metas."""
    return baseline.build_baseline_from_metas(
        aoi, pre_metas, tile_id, tile_data_dir,
        scene_dir=None, campaign_start=campaign_start,
    )

# Formati vettoriali supportati per le AOI, in ordine di priorità.
# GeoParquet richiede GDAL >= 3.5 compilato con il driver Arrow/Parquet.
_AOI_EXTENSIONS = [".geojson", ".gpkg", ".shp", ".kml", ".gml", ".parquet"]


def _scan_aois(aois_root="AOIs"):
    """Scansiona <aois_root>/ e restituisce dict {nome_cartella: path_file}.

    Supporta qualsiasi formato in _AOI_EXTENSIONS (GeoJSON, GeoPackage,
    GeoParquet, Shapefile, KML, GML). Usa il primo file trovato per ciascuna
    cartella, nell'ordine di priorità definito da _AOI_EXTENSIONS.
    """
    root = Path(aois_root)
    found = {}
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        path = None
        for ext in _AOI_EXTENSIONS:
            candidates = list(subfolder.glob(f"*{ext}"))
            if candidates:
                path = str(candidates[0])
                break
        if path:
            found[subfolder.name] = path
        else:
            logger.warning("AOI '%s': nessun file vettoriale trovato, cartella ignorata",
                           subfolder.name)
    return found


def main():
    # Forza UTF-8 sullo stdout per compatibilita' OS-indipendente
    _utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                                   line_buffering=True)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    _console_handler = logging.StreamHandler(_utf8_stdout)
    _console_handler.setFormatter(_fmt)

    # Setup logging console-only in anticipo (prima del parse args)
    logging.basicConfig(level=logging.INFO, handlers=[_console_handler])

    p = argparse.ArgumentParser(
        description="Monitoraggio aree bruciate Sentinel-2 — lancio operativo"
    )
    p.add_argument(
        "--aoi", default=None,
        help="Nome sottocartella in AOIs/ da processare (default: tutte)",
    )
    p.add_argument(
        "--aois-root", default="AOIs",
        help="Path alla cartella radice delle AOI (default: AOIs/)",
    )
    p.add_argument(
        "--output-root", default="output",
        help="Path radice degli output (default: output/)",
    )
    p.add_argument(
        "--stac-url", default="https://earth-search.aws.element84.com/v1",
        help="URL catalogo STAC",
    )
    p.add_argument(
        "--collection", default="sentinel-2-c1-l2a",
        help="Nome collection STAC",
    )
    args = p.parse_args()

    # Timestamp condiviso per i nomi dei log per-AOI
    _run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Date campagna ---
    # CAMPAIGN_START_DATE = None  → campagna parte da oggi (solo baseline)
    # CAMPAIGN_START_DATE = "2024-07-01" → monitoraggio storico da quella data
    today = datetime.utcnow().date()
    if config.CAMPAIGN_START_DATE is None:
        campaign_start = today
    else:
        campaign_start = datetime.fromisoformat(config.CAMPAIGN_START_DATE).date()

    baseline_end = campaign_start
    baseline_start = baseline_end - timedelta(days=config.BASELINE_LOOKBACK_DAYS)
    monitoring_end = today  # sempre "oggi"

    logger.info("=" * 60)
    logger.info("MONITORAGGIO INCENDI -- avvio operativo")
    if config.CAMPAIGN_START_DATE is None:
        logger.info("  Modalita': operativa (campaign_start da file per-AOI o today)")
    else:
        logger.info("  Baseline:    %s -> %s", baseline_start, baseline_end)
        logger.info("  Monitoraggio: %s -> %s", campaign_start, monitoring_end)
    logger.info("=" * 60)

    # --- Scan AOI ---
    aoi_map = _scan_aois(args.aois_root)
    if not aoi_map:
        logger.error("Nessuna AOI trovata in '%s'", args.aois_root)
        sys.exit(1)

    if args.aoi:
        if args.aoi not in aoi_map:
            logger.error("AOI '%s' non trovata in '%s'. Disponibili: %s",
                         args.aoi, args.aois_root, ", ".join(aoi_map))
            sys.exit(1)
        aoi_map = {args.aoi: aoi_map[args.aoi]}

    logger.info("AOI da processare: %s", ", ".join(aoi_map))

    had_failures = False

    for aoi_name, shp_path in aoi_map.items():
        # Log separato per AOI: logs/run_<timestamp>_<aoi_name>.log
        _aoi_log_dir = Path(args.output_root) / "logs"
        _aoi_log_dir.mkdir(parents=True, exist_ok=True)
        _aoi_log_path = _aoi_log_dir / f"run_{_run_ts}_{aoi_name}.log"
        _aoi_fh = logging.FileHandler(str(_aoi_log_path), encoding="utf-8")
        _aoi_fh.setFormatter(_fmt)
        logging.getLogger().addHandler(_aoi_fh)

        logger.info("")
        logger.info(">>> AOI: %s", aoi_name)
        logger.info("    Log AOI: %s", _aoi_log_path)

        aoi = data_io.load_aoi(shp_path)
        aoi["name"] = aoi_name
        bbox_wgs84 = data_io.get_aoi_bbox_wgs84(aoi)
        logger.info("    bbox WGS84: %s", bbox_wgs84)

        aoi_root   = Path(args.output_root) / aoi_name
        data_dir   = str(aoi_root / "data")
        output_dir = str(aoi_root / "products")

        # campaign_start per AOI: legge dallo state se disponibile (evita deriva giornaliera).
        if config.CAMPAIGN_START_DATE is None:
            first_tile_state_dir = next(
                (str(aoi_root / "data" / d.name)
                 for d in (aoi_root / "data").iterdir()
                 if d.is_dir()),
                None,
            ) if (aoi_root / "data").exists() else None
            persisted = None
            if first_tile_state_dir:
                _st = pipeline_state.load_state(first_tile_state_dir)
                persisted = _st.get("baseline", {}).get("campaign_start")
            if persisted:
                aoi_campaign_start = datetime.fromisoformat(persisted).date()
                logger.info("    campaign_start da state: %s", aoi_campaign_start)
            else:
                aoi_campaign_start = campaign_start  # today
        else:
            aoi_campaign_start = campaign_start  # data esplicita da config

        aoi_baseline_end   = aoi_campaign_start
        aoi_baseline_start = aoi_baseline_end - timedelta(days=config.BASELINE_LOOKBACK_DAYS)

        # --- Query STAC baseline ---
        logger.info("    Query STAC baseline: %s -> %s", aoi_baseline_start, aoi_baseline_end)
        try:
            pre_metas = data_io.query_stac(
                bbox_wgs84,
                str(aoi_baseline_start), str(aoi_baseline_end),
                args.stac_url, args.collection, max_items=500,
            )
            pre_metas = _filter_scenes(pre_metas, tile_id=f"{aoi_name}/baseline")
        except Exception as exc:
            logger.error("    Errore query STAC baseline per '%s': %s", aoi_name, exc)
            had_failures = True
            continue

        # --- Costruzione baseline per-tile (salta se già su disco) ---
        tile_ids_pre = sorted({_get_tile_id(m) for m in pre_metas
                                if _get_tile_id(m) != "unknown"})
        if not tile_ids_pre:
            logger.warning("    Nessun tile trovato nella query baseline per '%s', salto",
                           aoi_name)
            continue
        logger.info("    Tile rilevati: %s", ", ".join(tile_ids_pre))
        baseline_ok = True
        for tid in tile_ids_pre:
            tile_data_dir = str(aoi_root / "data" / tid)
            if not _build_baseline_from_metas(aoi, pre_metas, tid, tile_data_dir,
                                               campaign_start=aoi_campaign_start):
                logger.error("    Baseline fallita per tile %s — AOI '%s' saltata",
                             tid, aoi_name)
                baseline_ok = False
        if not baseline_ok:
            had_failures = True
            continue

        # --- Query STAC monitoraggio ---
        if aoi_campaign_start < monitoring_end:
            logger.info("    Query STAC monitoraggio: %s -> %s",
                        aoi_campaign_start, monitoring_end)
            try:
                post_metas = data_io.query_stac(
                    bbox_wgs84,
                    str(aoi_campaign_start), str(monitoring_end),
                    args.stac_url, args.collection, max_items=2000,
                )
                post_metas = _filter_scenes(post_metas, tile_id=f"{aoi_name}/monitoraggio")
            except Exception as exc:
                logger.error("    Errore query STAC monitoraggio per '%s': %s",
                             aoi_name, exc)
                had_failures = True
                continue
        else:
            # aoi_campaign_start == oggi: nessuna scena da monitorare ancora
            post_metas = []
            logger.info("    Nessuna scena di monitoraggio (campagna inizia oggi — "
                        "baseline pronta per il prossimo ciclo schedulato)")

        if not post_metas:
            logger.info("    AOI '%s' completata: baseline OK, nessuna scena da processare",
                        aoi_name)
            continue

        try:
            result = process_aoi(
                aoi,
                scenes=post_metas,
                output_dir=output_dir,
                data_dir=data_dir,
            )
        except Exception as exc:
            logger.error("    Errore pipeline per AOI '%s': %s", aoi_name, exc,
                         exc_info=True)
            had_failures = True

        finally:
            # Chiudi e rimuovi il file handler per-AOI
            logging.getLogger().removeHandler(_aoi_fh)
            _aoi_fh.close()

    if had_failures:
        sys.exit(1)

