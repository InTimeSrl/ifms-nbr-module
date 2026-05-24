"""
pipeline.py -- Orchestratore del monitoraggio continuo aree bruciate.

Logica per evento (post-refactor):
  - baseline_nbr  : composite pre-campagna (fisso), da baseline.py
  - previous_nbr  : inizia = baseline, evolve a ogni scena valida (anche burnt
                    sui pixel non bruciati)
  - eventi        : gestiti da events.py. Ogni cluster bruciato >= soglia apre
                    un evento o aggiorna un evento attivo (overlap di bbox).
                    Ogni scena valida contribuisce a obs_count degli eventi
                    attivi del tile (e a burnt_count su quelli toccati).
  - chiusura      : end_event.close_event(...) quando un evento raggiunge
                    EVENT_WINDOW_SCENES o EVENT_TIMEOUT_DAYS.

TODO: _get_scenes_stac() da completare quando sara' disponibile il
server STAC interno.
"""

import argparse
import io
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyproj
import rasterio.transform as _rt
import rasterio.windows as _rwin
from rasterio.features import geometry_mask as _geometry_mask
from pystac_client import Client
from scipy.ndimage import binary_dilation as _binary_dilation, distance_transform_edt as _distance_transform_edt
from shapely.geometry import mapping as _shapely_mapping
from shapely.ops import transform as _shp_transform

from . import config
from . import data_io
from . import baseline
from . import preprocess
from . import indices, classify, postprocess
from . import events, end_event
from . import state as pipeline_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recupero scene (locale o STAC)
# ---------------------------------------------------------------------------

def _get_scenes_stac(aoi, stac_client, date_from=None):
    """Recupera scene da un catalogo STAC.

    TODO: implementare quando il server STAC interno sara' disponibile.
    """
    raise NotImplementedError("Query STAC non ancora implementata")


def get_scenes(aoi, scene_dir=None, stac_client=None, date_from=None):
    """Recupera le scene disponibili per un'AOI (locale o STAC).

    Una tra scene_dir e stac_client deve essere fornita.
    """
    if scene_dir is not None:
        return data_io.list_scenes(scene_dir)
    elif stac_client is not None:
        return _get_scenes_stac(aoi, stac_client, date_from)
    else:
        raise ValueError("Specificare scene_dir (locale) o stac_client (remoto)")


# ---------------------------------------------------------------------------
# Filtri pre-processing a livello di scena
# ---------------------------------------------------------------------------

def _filter_scenes(scenes, tile_id=""):
    """Filtra scene non valide per qualita' dati.

    Con il modello watermark (scheduling) le scene gia' processate non
    arrivano qui: la query STAC usa come data_from il watermark
    dell'ultima scena processata, quindi questo filtro si occupa
    esclusivamente della qualita':
    - cloud_cover <= MAX_CLOUD_COVER_PCT
    - processing_baseline >= MIN_PROCESSING_BASELINE

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

    # --- Calcolo indice di variazione NBR (dNBR o RBR, da config) ---
    if config.INDEX_MODE == "RBR":
        delta = indices.compute_rbr(previous_nbr, nbr)
        _index_label = "RBR"
        _threshold = config.RBR_THRESHOLD
    else:
        delta = indices.compute_dnbr(previous_nbr, nbr)
        _index_label = "dNBR"
        _threshold = config.DNBR_THRESHOLD

    out["dnbr"] = delta  # chiave fissa per compatibilita' output; contiene dNBR o RBR

    # --- Verifica incendio (pixel-level + soglia area minima) ---
    # Tripla soglia: indice alto + NIR basso + SWIR minimo
    # (esclude nubi misclassificate dalla SCL e corpi idrici con SWIR ~0)
    _swir_min = getattr(config, "SWIR_MIN_BURNT", None)
    burnt_mask = (
        (delta > _threshold)
        & (nir < config.NIR_MAX_BURNT)
        & (swir > _swir_min if _swir_min is not None else True)
        & valid_mask
    )
    out["burnt_mask"] = burnt_mask

    # Calcola area bruciata in ettari (pixel 20m x 20m = 400 m2 = 0.04 ha)
    pixel_area_ha = 0.04
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
    """Esegue la pipeline completa per un'AOI.

    Parameters
    ----------
    scenes : list[dict], optional
        Scene pre-caricate (es. da query STAC esterna). Se None, vengono
        recuperate tramite get_scenes().
    """
    aoi_name = aoi["name"]
    logger.info("=== Inizio processamento AOI: %s ===", aoi_name)

    # Recupera tutte le scene una sola volta e determina i tile da processare
    # Nota: per scheduling continuo con STAC, passare date_from=<watermark>
    # al chiamante prima di invocare process_aoi, oppure interrogare STAC
    # per-tile usando il watermark da pipeline_state.get_watermark(tile_state).
    all_scenes = scenes if scenes is not None else get_scenes(
        aoi, scene_dir=scene_dir, stac_client=stac_client
    )
    tile_ids = sorted({_get_tile_id(s) for s in all_scenes if _get_tile_id(s) != "unknown"})

    if not tile_ids:
        logger.info("AOI '%s': nessun tile valido trovato nelle scene", aoi_name)
        return {"aoi": aoi_name, "scenes_processed": 0, "alerts": 0, "tiles": []}

    total_scenes_processed = 0
    total_alerts = 0
    processed_tiles = []

    for tile_id in tile_ids:
        tile_data_dir = str(Path(data_dir) / tile_id)
        tile_output_dir = str(Path(output_dir) / tile_id)
        logger.info("--- AOI '%s' | Tile %s ---", aoi_name, tile_id)

        # Stato persistente per-tile: watermark = ISO 8601 datetime dell'ultima
        # scena processata. Usato come data_floor per la selezione delle scene:
        # solo scene con datetime > watermark vengono processate. A ogni scena
        # il watermark viene avanzato e salvato su disco (pipeline_state.json
        # in tile_data_dir), cosi' il prossimo scheduling riparte esattamente
        # dall'ultima scena vista, senza tracciare ogni singolo scene_id.
        tile_state = pipeline_state.load_state(tile_data_dir)
        watermark = pipeline_state.get_watermark(tile_state)
        if watermark:
            logger.info("Tile %s: watermark=%s (ultime scene gia' processate)",
                        tile_id, watermark)

        paths = baseline.nbr_paths(aoi, tile_data_dir)
        baseline_nbr, baseline_profile = baseline.load_nbr(paths["baseline"])
        previous_nbr, _ = baseline.load_nbr(paths["previous"])

        # --- Se baseline non esiste, costruiscila retrospettivamente per tile ---
        if baseline_nbr is None:
            def _get_scenes_for_tile(date_from=None, _tile_id=tile_id):
                scenes = get_scenes(
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

        # --- Maschera AOI rasterizzata sul grid della tile (una volta per tile) ---
        # Il poligono AOI viene riproiettato nel CRS della tile e rasterizzato.
        # La maschera viene applicata a valid_mask e burnt_mask ad ogni scena,
        # cosi' tutta la pipeline opera solo sui pixel interni all'AOI:
        # eventi, baseline, footprint e output sono automaticamente ritagliati.
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

        # Finestra di crop AOI (bounding box del poligono sul grid tile)
        # Usata per ritagliare tutti i raster/vettori di output all'AOI.
        _aoi_crop_window = None
        if _tile_aoi_mask is not None:
            _rows_in = np.where(_tile_aoi_mask.any(axis=1))[0]
            _cols_in = np.where(_tile_aoi_mask.any(axis=0))[0]
            if len(_rows_in) > 0 and len(_cols_in) > 0:
                _cr0, _cr1 = int(_rows_in[0]), int(_rows_in[-1]) + 1
                _cc0, _cc1 = int(_cols_in[0]), int(_cols_in[-1]) + 1
                _aoi_crop_window = (_cr0, _cc0, _cr1 - _cr0, _cc1 - _cc0)

        # Applica maschera AOI alla baseline in memoria e su disco.
        # baseline_nbr e previous_nbr coprono l'intera tile; azzerando i pixel
        # fuori AOI a NaN si garantisce che anche i file .tif mostrati in QGIS
        # siano limitati all'AOI e che il dNBR su pixel extra-AOI sia sempre NaN.
        if _tile_aoi_mask is not None and baseline_profile is not None:
            if baseline_nbr is not None and baseline_nbr.shape == _tile_aoi_mask.shape:
                baseline_nbr = np.where(_tile_aoi_mask, baseline_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(baseline_nbr, baseline_profile, paths["baseline"])
            if previous_nbr is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(previous_nbr, baseline_profile, paths["previous"])

        # --- Ripristina protezione frozen_nbr per eventi attivi (resume dopo crash) ---
        # Se un evento precedente si e' chiuso mentre altri erano ancora attivi,
        # il file frozen_nbr_{eid}_temp.tif contiene il baseline pre-fire congelato
        # per quella zona. Lo riapplichiamo in memoria prima di processare nuove scene.
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

        # Recupera scene per tile: solo quelle successive al watermark
        # (o a CAMPAIGN_START_DATE se e' il primo run schedulato senza scene esterne).
        # Se le scene sono passate dall'esterno (scenes!=None), il chiamante ha gia'
        # filtrato per data — usiamo "" come floor per accettarle tutte.
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
        # Ordine cronologico crescente: la pipeline deve vedere le scene dalla
        # piu' vecchia alla piu' recente per aggiornare correttamente previous_nbr.
        tile_scenes.sort(key=lambda s: s.get("datetime", s.get("date", "")))

        if not tile_scenes:
            logger.info("Nessuna nuova scena per AOI '%s' sul tile %s", aoi_name, tile_id)
            continue

        tile_scenes_processed = 0
        tile_alerts = 0

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
            nbr = result["nbr"]
            valid_mask = result["valid_mask"]
            profile = result["profile"]
            dnbr = result["dnbr"]
            burnt_mask = result.get("burnt_mask", np.zeros_like(valid_mask))

            # Applica maschera AOI: limita processing ai pixel interni all'AOI
            if _tile_aoi_mask is not None:
                _m = _tile_aoi_mask
                if _m.shape != valid_mask.shape:
                    # Sicurezza: se il profilo della scena differisce dalla baseline
                    # (non dovrebbe accadere per tile MGRS uniformi), salta la maschera
                    logger.warning("Tile %s: shape AOI mask %s != valid_mask %s, skip",
                                   tile_id, _m.shape, valid_mask.shape)
                else:
                    valid_mask = valid_mask & _m
                    burnt_mask = burnt_mask & _m
            _threshold = result.get("threshold", config.DNBR_THRESHOLD)
            # --- Variabili per il riepilogo per-scena (emesso a fine elaborazione) ---
            _log_area_ok     = result.get("fire_detected", False)
            _log_area_ha     = result.get("burnt_area_ha", 0.0)
            _log_idx_max     = result.get("max_index_val", 0.0)
            _log_idx_label   = result.get("index_label", "dNBR")
            _log_cluster_ok  = None   # None=SKIP, True=pass, False=fail
            _log_cluster_ha  = None
            _log_total_ha    = None
            _log_compact_pct = None
            _log_compact_fallback = False  # True se apertura via controllo integrato
            _log_dnbr_fallback    = 0.0   # mean_dNBR della scena quando fallback attivo
            _log_skip_reason = None   # motivo SKIP del cluster

            _land_px = int(np.isfinite(baseline_nbr).sum()) if baseline_nbr is not None else 0
            _denom = _land_px if _land_px > 0 else valid_mask.size
            valid_pct = 100.0 * valid_mask.sum() / _denom

            # --- Recupera lista eventi attivi per (tile, AOI) ---
            # Caricato PRIMA del check SCL per avere il verdetto corretto anche
            # quando la scena viene scartata per nuvole durante la finestra evento.
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
                # ---
                pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
                pipeline_state.save_state(tile_state, tile_data_dir)
                continue

            pixel_res = abs(profile["transform"].a)
            pixel_area_ha = (pixel_res * pixel_res) / 10_000.0
            _min_compactness = getattr(config, "MIN_CLUSTER_COMPACTNESS", None)
            _min_compact_fb  = getattr(config, "MIN_CLUSTER_COMPACTNESS_FALLBACK", None)
            _min_dnbr_fb     = getattr(config, "MIN_CLUSTER_DNBR_FALLBACK", None)
            _log_verdicts = []

            if active_eids:
                # -------------------------------------------------------
                # EVENTI ATTIVI: assegna pixel per nearest-footprint, poi
                # verifica se i rimanenti aprono nuovi eventi.
                # -------------------------------------------------------
                # Candidati per l'accumulo: usa burnt_mask (include filtro NIR)
                # per evitare che vegetazione in essiccamento fenologico (NIR alto)
                # venga accumulata come fuoco durante la finestra evento.
                burnt_for_accum = burnt_mask.copy()
                remaining = burnt_mask.copy()

                # Assegnazione nearest-footprint (Voronoi): ogni pixel candidato
                # va all'evento il cui footprint e' geograficamente piu' vicino.
                # Con un solo evento attivo non serve calcolare distanze.
                footprints = {eid: events.load_footprint_mask(eid, tile_output_dir)
                              for eid in active_eids}

                if len(active_eids) == 1:
                    assignments = {active_eids[0]: burnt_for_accum}
                else:
                    dist_stack = np.stack([
                        _distance_transform_edt(~fp).astype(np.float64)
                        if (fp is not None and fp.any())
                        else np.full(burnt_for_accum.shape, np.inf, dtype=np.float64)
                        for fp in (footprints[e] for e in active_eids)
                    ], axis=0)
                    nearest_idx = np.argmin(dist_stack, axis=0)
                    assignments = {
                        eid: burnt_for_accum & (nearest_idx == i)
                        for i, eid in enumerate(active_eids)
                    }

                for eid in active_eids:
                    assigned = assignments[eid]

                    _ev_n_valid = events.update_event(
                        eid, assigned, dnbr, valid_mask, profile, scene, tile_output_dir,
                    )
                    remaining = remaining & ~assigned

                    if assigned.any():
                        _save_outputs(
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
                    if not _compact_ok_rem and _min_compact_fb and _min_dnbr_fb:
                        _mean_dnbr_rem = float(dnbr[remaining].mean()) if remaining.any() else 0.0
                        _compact_ok_rem = _compact_rem >= _min_compact_fb and _mean_dnbr_rem >= _min_dnbr_fb
                    if _remaining_ha >= config.MIN_ALERT_AREA_HA and _compact_ok_rem:
                        new_clusters = events.find_clusters(remaining, pixel_area_ha)
                        for cl in new_clusters:
                            new_eid = events.open_event(
                                cl["mask"], dnbr, valid_mask, profile, scene,
                                tile_output_dir, tile=tile_id, aoi=aoi_name,
                            )
                            active_eids.append(new_eid)
                            tile_alerts += 1
                            _save_outputs(
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
                _log_skip_reason = "evento attivo"

            else:
                # -------------------------------------------------------
                # NESSUN EVENTO ATTIVO: applica filtri e apri per cluster
                # -------------------------------------------------------
                largest_cluster_ha = events.largest_cluster_area_ha(burnt_mask, pixel_area_ha)
                _total_burnt_ha = float(burnt_mask.sum()) * pixel_area_ha
                _compactness = (largest_cluster_ha / _total_burnt_ha) if _total_burnt_ha > 0 else 0.0
                _compact_ok = not _min_compactness or _compactness >= _min_compactness
                if not _compact_ok and _min_compact_fb and _min_dnbr_fb:
                    _mean_dnbr_scene = float(dnbr[burnt_mask].mean()) if burnt_mask.any() else 0.0
                    if _compactness >= _min_compact_fb and _mean_dnbr_scene >= _min_dnbr_fb:
                        _compact_ok = True
                        _log_compact_fallback = True
                        _log_dnbr_fallback    = _mean_dnbr_scene

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
                        # find_clusters non ha prodotto nulla (tutti scartati da solidity):
                        # fallback al comportamento precedente (apri su intera maschera)
                        new_clusters = [{"mask": burnt_mask, "area_ha": largest_cluster_ha}]
                    # Cluster della stessa scena iniziale: si fondono se i centroidi
                    # distano <= MAX_INITIAL_MERGE_DISTANCE_KM. Oltre la soglia si
                    # tratta di incendi distinti -> eventi separati.
                    _tr = profile["transform"]
                    _merge_thr_m = config.MAX_INITIAL_MERGE_DISTANCE_KM * 1000.0
                    # Centroidi in coordinate proiettate (stessa unita' del transform)
                    _cl_xy = []
                    for _cln in new_clusters:
                        _ys, _xs = np.where(_cln["mask"])
                        _x, _y = _rt.xy(_tr, float(_ys.mean()), float(_xs.mean()))
                        _cl_xy.append((_x, _y))
                    # Log distanze (utile per calibrare la soglia)
                    if len(new_clusters) > 1:
                        for _ci in range(1, len(new_clusters)):
                            _dx = _cl_xy[0][0] - _cl_xy[_ci][0]
                            _dy = _cl_xy[0][1] - _cl_xy[_ci][1]
                            _d_km = (_dx**2 + _dy**2)**0.5 / 1000
                            _merge = _d_km <= config.MAX_INITIAL_MERGE_DISTANCE_KM
                            logger.info(
                                "  cluster #0 %.1f ha  +  cluster #%d %.1f ha  |  dist=%.1f km  -> %s",
                                new_clusters[0]["area_ha"],
                                _ci, new_clusters[_ci]["area_ha"], _d_km,
                                "UNITI" if _merge else f"SEPARATI (soglia {config.MAX_INITIAL_MERGE_DISTANCE_KM:.0f} km)",
                            )
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
                        _save_outputs(
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
            _area_str = (
                f"{_log_area_ha:.2f} ha  ({_log_idx_label} max={_log_idx_max:.3f})  "
                + ("OK" if _log_area_ok else f"FAIL (< {config.MIN_ALERT_AREA_HA:.1f} ha)")
            )
            logger.info("  SCL:       %.1f%%  OK", valid_pct)
            logger.info("  area:      %s", _area_str)
            if not _log_area_ok:
                logger.info("  cluster:   SKIP (area < soglia)")
            elif _log_cluster_ok is None:
                _skip_str = f" ({_log_skip_reason})" if _log_skip_reason else ""
                logger.info("  cluster:   SKIP%s", _skip_str)
            elif _log_cluster_ok:
                if _log_compact_fallback:
                    logger.info(
                        "  cluster:   %.1f/%.1f ha = %.1f%%  OK (fallback dNBR=%.3f)",
                        _log_cluster_ha, _log_total_ha, _log_compact_pct, _log_dnbr_fallback,
                    )
                else:
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

            # --- Aggiorna previous_nbr e salva ---
            # Con eventi ATTIVI: previous_nbr completamente congelato.
            # Il riferimento pre-fire deve restare stabile per tutta la
            # finestra evento: aggiornarlo produrrebbe una baseline ibrida.
            # La ricalibrazione integrale e' delegata a close_event.
            #
            # Senza eventi: aggiorniamo su TUTTI i pixel validi, inclusi
            # quelli che hanno superato la soglia ma non hanno formato un
            # cluster >= MIN_ALERT_AREA_HA. Senza questo, i pixel "edge"
            # della cicatrice manterrebbero il previous_nbr pre-fire e
            # continuerebbero a scattare ad ogni scena (loop infinito).
            if not active_eids:
                previous_nbr = np.where(valid_mask, nbr, previous_nbr)
                if _tile_aoi_mask is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                    previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                baseline.save_nbr(previous_nbr, profile, paths["previous"])

            # --- Verifica chiusura per ogni evento attivo ---
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
                    logger.info("Output rimossi (false_positive): %s", eid)
                    tile_alerts -= 1
                active_eids.remove(eid)
                # Rimuovi frozen_nbr_temp per l'evento appena chiuso
                _frozen_del = Path(tile_data_dir) / f"frozen_nbr_{eid}_temp.tif"
                _frozen_del.unlink(missing_ok=True)
                if _is_fp:
                    # BUG-FIX: previous_nbr era congelato per tutta la finestra evento.
                    # close_event (false_positive) salta la ricalibrazione della baseline.
                    # Senza questo aggiornamento, la scena successiva calcola dNBR contro
                    # il valore pre-evento: l'essiccamento stagionale accumulato appare
                    # come incendio → loop infinito di false positive.
                    if not active_eids:  # aggiorna solo se non ci sono altri eventi attivi
                        previous_nbr = np.where(valid_mask, nbr, previous_nbr)
                        if _tile_aoi_mask is not None and previous_nbr.shape == _tile_aoi_mask.shape:
                            previous_nbr = np.where(_tile_aoi_mask, previous_nbr, np.nan).astype(np.float32)
                        baseline.save_nbr(previous_nbr, profile, paths["previous"])
                    logger.info("Baseline aggiornata post false_positive (era frozen %d scene)",
                                _summary.get("n_valid_scenes", 0) if _summary else 0)
                else:
                    # Ricarica previous_nbr: close_event ha scritto il baseline
                    # aggiornato su disco (cicatrice dell'evento chiuso incorporata).
                    # Lo carichiamo subito in memoria per evitare la ri-detezione
                    # della cicatrice appena chiusa (loop EVT3 -> EVT5 -> EVT6 ...).
                    #
                    # Per gli eventi ANCORA ATTIVI ripristiniamo il valore
                    # pre-fire nella loro zona (in memoria, non su disco): cosi'
                    # il loro baseline rimane congelato al pre-fire e continuano
                    # a rilevare correttamente i propri pixel bruciati.
                    _reloaded, _ = data_io.read_band(str(paths["previous"]))
                    if _reloaded is not None:
                        _new_prev = _reloaded.astype(np.float32)
                        for _still_eid in active_eids:  # eid chiuso gia' rimosso
                            _fp = events.load_footprint_mask(_still_eid, tile_output_dir)
                            if _fp is not None and _fp.any():
                                _sep = getattr(config, "EVENT_BASELINE_BUFFER_PX", 0) or 0
                                _zone = (_binary_dilation(_fp, iterations=int(_sep))
                                         if _sep > 0 else _fp)
                                # Mantieni previous_nbr pre-fire nella zona attiva
                                _new_prev = np.where(_zone, previous_nbr, _new_prev)
                        previous_nbr = _new_prev
                        # Salva frozen_nbr_temp per ogni evento ancora attivo:
                        # se il pipeline crasha prima che questi eventi si chiudano,
                        # al riavvio il baseline pre-fire viene ripristinato dal file.
                        for _still_eid in active_eids:
                            _frozen_path = Path(tile_data_dir) / f"frozen_nbr_{_still_eid}_temp.tif"
                            baseline.save_nbr(previous_nbr, profile, str(_frozen_path))
                    else:
                        previous_nbr = np.where(valid_mask, nbr, previous_nbr)

            pipeline_state.update_watermark(tile_state, scene.get("datetime", scene.get("date", "")))
            pipeline_state.save_state(tile_state, tile_data_dir)

        logger.info(
            "Tile %s completato: %d scene processate, %d nuovi alert (eventi aperti)",
            tile_id, tile_scenes_processed, tile_alerts,
        )
        total_scenes_processed += tile_scenes_processed
        total_alerts += tile_alerts
        processed_tiles.append(tile_id)

    logger.info(
        "=== AOI '%s' completata: %d tile, %d scene processate, %d alert ===",
        aoi_name, len(processed_tiles), total_scenes_processed, total_alerts,
    )
    return {
        "aoi": aoi_name,
        "tiles": processed_tiles,
        "scenes_processed": total_scenes_processed,
        "alerts": total_alerts,
    }


# ---------------------------------------------------------------------------
# Salvataggio output
# ---------------------------------------------------------------------------

def _save_outputs(result, scene, aoi, output_dir, scene_dir=None,
                  event_ids=None, tile_id=None, scene_ts=None,
                  aoi_crop=None, aoi_mask=None):
    """Salva i prodotti preliminari di una scena con incendio rilevato.

    Output (suffisso ``_prelim`` perche' si tratta di rilevamento di scena, non
    della perimetrazione finale dell'evento -- quella la produce
    ``end_event.close_event``):

    - dNBR GeoTIFF
    - Severity GeoTIFF (uint8, classi 1-7)
    - Poligoni dissolti GeoPackage (CRS nativo)
    - RGB HONC GeoTIFF (opzionale, se config.PRODUCE_RGB e' True)

    Parameters
    ----------
    result : dict
        Output di process_scene (con fire_detected=True).
    scene : dict
        Metadati della scena.
    aoi : dict
        AOI dict.
    output_dir : str
        Cartella radice output.
    scene_dir : str o Path, optional
        Cartella locale dei TIF (necessaria per produrre RGB).
    event_ids : str, optional
        Stringa con gli event_id (separati da virgola) toccati da questa scena,
        salvata come proprieta' nel GeoPackage.
    tile_id : str, optional
        Tile MGRS (es. T35SMC), salvato come proprieta'.
    """
    scene_id = scene["stac_item_id"]
    if event_ids and scene_ts:
        # Sidecar folder condiviso per tutte le scene dell'evento.
        evt_part = event_ids.split("_")[-1] if "_EVT" in event_ids else event_ids
        out_dir = Path(output_dir) / event_ids
        gpkg_path = Path(output_dir) / f"{event_ids}.gpkg"
        raster_pfx = scene_ts
        layer_name_prelim = f"{scene_ts}_burnt_prelim_{evt_part}"
    else:
        out_dir = Path(output_dir) / scene_id
        gpkg_path = out_dir / f"{scene_id}_burnt_prelim.gpkg"
        raster_pfx = scene_id
        layer_name_prelim = "burnt_prelim"
    profile = result["profile"]
    _dnbr = result.get("dnbr")
    _severity = result.get("severity")

    # Azzera pixel fuori AOI (nodata): applica maschera PRIMA del crop
    if aoi_mask is not None:
        if _dnbr is not None and _dnbr.shape == aoi_mask.shape:
            _dnbr = np.where(aoi_mask, _dnbr, np.nan).astype(np.float32)
        if _severity is not None and _severity.shape == aoi_mask.shape:
            _severity = np.where(aoi_mask, _severity, 0).astype(_severity.dtype)

    # Ritaglia al bounding box dell'AOI sul grid tile
    if aoi_crop is not None:
        _r0, _c0, _nrows, _ncols = aoi_crop
        _win = _rwin.Window(_c0, _r0, _ncols, _nrows)
        _crop_tr = _rwin.transform(_win, profile["transform"])
        profile = {**profile, "height": _nrows, "width": _ncols, "transform": _crop_tr}
        if _dnbr is not None:
            _dnbr = _dnbr[_r0:_r0 + _nrows, _c0:_c0 + _ncols]
        if _severity is not None:
            _severity = _severity[_r0:_r0 + _nrows, _c0:_c0 + _ncols]

    # dNBR preliminare
    data_io.write_geotiff(
        _dnbr, profile,
        out_dir / f"{raster_pfx}_dNBR.tif",
        dtype="float32",
    )

    # Severity raster preliminare
    if _severity is not None:
        data_io.write_geotiff(
            _severity, profile,
            out_dir / f"{raster_pfx}_severity_prelim.tif",
            dtype="uint8",
            nodata=0,
        )

    # Poligoni GeoPackage preliminari (CRS nativo)
    if _severity is not None and _dnbr is not None:
        meta = {
            "event_id": event_ids or "",
            "detection_date": scene.get("datetime", scene.get("date", "")),
            "satellite": scene_id.split("_")[0] if "_" in scene_id else "",
            "processing_mode": "scene_prelim",
            "aoi_ref": aoi.get("name", ""),
            "tile": tile_id or "",
            "cloud_cover_pct": scene.get("eo_cloud_cover"),
            "index_mode": config.INDEX_MODE,
            "index_threshold": config.RBR_THRESHOLD if config.INDEX_MODE == "RBR" else config.DNBR_THRESHOLD,
        }
        features = postprocess.vectorize_by_severity(
            _severity, _dnbr, profile, meta=meta,
        )
        crs_str = str(profile.get("crs", "")) or None
        data_io.write_geopackage(
            features,
            gpkg_path,
            crs=crs_str,
            layer_name=layer_name_prelim,
        )

    logger.info("Output preliminari salvati in %s", out_dir)

    # RGB composito Highlight Optimized Natural Color (opzionale)
    if config.PRODUCE_RGB:
        _save_rgb_composite(scene, aoi, scene_dir, out_dir)

    return out_dir


def _save_rgb_composite(scene, aoi, scene_dir, out_dir):
    """Produce e salva composito RGB Highlight Optimized Natural Color (HONC).

    Carica le bande B4, B3, B2 a 10 m nativi, applica la formula HONC
    di Marko Repse (cbrt(0.6 * reflectance)) e salva un GeoTIFF uint8
    a 3 bande.

    Parameters
    ----------
    scene : dict
        Metadati della scena.
    aoi : dict
        AOI dict.
    scene_dir : str o Path
        Cartella locale dei TIF (o None per lettura remota COG).
    out_dir : Path
        Cartella di output della scena.
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

        # RGB true-color: stretch percentile globale (p2-p98) + gamma 0.8
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
        honc_uint8[:, ~valid_mask] = 0             # solo pixel nodata a zero

        out_path = Path(out_dir) / f"{scene_id}_RGB_HONC.tif"
        data_io.write_rgb_geotiff(honc_uint8, profile_rgb, out_path)
        logger.info("RGB HONC salvato: %s", out_path)
    except Exception:
        logger.warning("RGB HONC non prodotto per %s (bande RGB non disponibili)", scene_id)


# ---------------------------------------------------------------------------
# Entry point operativo
# ---------------------------------------------------------------------------

def _build_baseline_from_metas(aoi, pre_metas, tile_id, tile_data_dir,
                               campaign_start=None):
    """Costruisce la baseline NBR da una lista di metadati pre-fetched.

    Se la baseline è già su disco per questo tile, la salta.

    Returns
    -------
    bool
        True se baseline disponibile (già esistente o appena costruita).
    """
    paths = baseline.nbr_paths(aoi, tile_data_dir)
    existing, _ = baseline.load_nbr(paths["baseline"])
    if existing is not None:
        logger.info("Baseline tile %s gia' presente, salto costruzione", tile_id)
        return True

    tile_metas = [m for m in pre_metas if _get_tile_id(m) == tile_id]
    if not tile_metas:
        logger.warning("Nessuna scena pre-campagna per tile %s", tile_id)
        return False

    logger.info("Costruzione baseline tile %s: %d scene candidate", tile_id, len(tile_metas))

    stack_list = []
    last_profile = None
    for meta in tile_metas:
        result = baseline.compute_nbr_from_scene(meta, aoi, scene_dir=None)
        if result is None:
            continue
        nbr, _nir, _swir, valid_mask, prof = result
        layer = np.full_like(nbr, np.nan)
        layer[valid_mask] = nbr[valid_mask]
        stack_list.append(layer)
        last_profile = prof

    if len(stack_list) < config.BASELINE_MIN_SCENES:
        logger.error(
            "Tile %s: solo %d scene utilizzabili (servono %d) — baseline non costruita",
            tile_id, len(stack_list), config.BASELINE_MIN_SCENES,
        )
        return False

    stack = np.array(stack_list, dtype="float32")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        pixel_median = np.nanmedian(stack, axis=0)
        pixel_mad = np.nanmedian(np.abs(stack - pixel_median[np.newaxis, :, :]), axis=0)
    pixel_mad = np.maximum(pixel_mad, config.BASELINE_MAD_FLOOR)
    threshold = pixel_median - config.BASELINE_MAD_K * pixel_mad
    anomaly_mask = stack < threshold[np.newaxis, :, :]
    n_anom = int(np.count_nonzero(anomaly_mask))
    if n_anom > 0:
        stack[anomaly_mask] = np.nan
        logger.info("Filtro MAD tile %s: rimossi %d pixel/scena (%.2f%%)",
                    tile_id, n_anom, 100.0 * n_anom / stack.size)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        baseline_nbr = np.nanmedian(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0).astype("float32")
    coverage = float((count > 0).mean())

    Path(tile_data_dir).mkdir(parents=True, exist_ok=True)
    baseline.save_nbr(baseline_nbr, last_profile, paths["baseline"])
    baseline.save_nbr(baseline_nbr.copy(), last_profile, paths["previous"])
    baseline.save_nbr(np.zeros_like(baseline_nbr), last_profile, paths["max_dnbr"])

    tile_state = pipeline_state.load_state(tile_data_dir)
    pipeline_state.mark_baseline_built(
        tile_state, len(stack_list), coverage * 100,
        [m["stac_item_id"] for m in tile_metas],
    )
    if campaign_start is not None:
        tile_state["baseline"]["campaign_start"] = str(campaign_start)
    pipeline_state.save_state(tile_state, tile_data_dir)

    logger.info("Baseline tile %s: %d scene, copertura %.1f%%",
                tile_id, len(stack_list), coverage * 100)
    return True

def _stac_item_to_meta(item):
    """Converte un STAC Item in dict metadati compatibile con la pipeline."""
    props = item.properties
    assets = {}
    for stac_key, band_name in config.BANDS.items():
        if stac_key in item.assets:
            asset = item.assets[stac_key]
            extra = {}
            if hasattr(asset, "extra_fields"):
                rb_list = asset.extra_fields.get("raster:bands", [])
                if rb_list:
                    extra["scale"] = rb_list[0].get("scale")
                    extra["offset"] = rb_list[0].get("offset")
            assets[stac_key] = {"href": asset.href, "band_name": band_name, **extra}
    return {
        "stac_item_id": item.id,
        "datetime": props.get("datetime", ""),
        "platform": props.get("platform", ""),
        "proj_code": "EPSG:%s" % props.get("proj:epsg", ""),
        "eo_cloud_cover": props.get("eo:cloud_cover"),
        "s2_processing_baseline": props.get("s2:processing_baseline", "99.99"),
        "bbox": list(item.bbox) if item.bbox else [],
        "assets": assets,
    }


def _query_stac(bbox_wgs84, date_from, date_to, stac_url, collection,
                max_items=2000):
    """Esegue una query STAC e restituisce la lista di metadati scena."""
    catalog = Client.open(stac_url)
    date_range = "%s/%s" % (date_from, date_to)
    logger.info("Query STAC: bbox=%s  date=%s", bbox_wgs84, date_range)
    results = catalog.search(
        collections=[collection],
        bbox=list(bbox_wgs84),
        datetime=date_range,
        max_items=max_items,
    )
    items = list(results.items())
    logger.info("STAC: trovate %d scene", len(items))
    return [_stac_item_to_meta(it) for it in items]



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

    # File handler: output/logs/run_<timestamp>.log (UTF-8, un file per lancio)
    _log_dir = Path("output") / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _file_handler = logging.FileHandler(str(_log_path), encoding="utf-8")
    _file_handler.setFormatter(_fmt)

    logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
    logger.info("Log salvato in: %s", _log_path)

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
    logger.info("MONITORAGGIO INCENDI — avvio operativo")
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
        logger.info("")
        logger.info(">>> AOI: %s", aoi_name)

        aoi = data_io.load_aoi(shp_path)
        aoi["name"] = aoi_name
        bbox_wgs84 = data_io.get_aoi_bbox_wgs84(aoi)
        logger.info("    bbox WGS84: %s", bbox_wgs84)

        aoi_root   = Path(args.output_root) / aoi_name
        data_dir   = str(aoi_root / "data")
        output_dir = str(aoi_root / "products")

        # --- Risoluzione campaign_start per AOI ---
        # Se CAMPAIGN_START_DATE=None (modalita' operativa), al primo run
        # campaign_start = today e viene salvato nello state JSON della baseline.
        # Ai run successivi (baseline gia' su disco) lo leggiamo dallo state,
        # evitando la deriva giornaliera senza creare file extra.
        if config.CAMPAIGN_START_DATE is None:
            # Prova a leggere da uno state tile gia' esistente
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
            pre_metas = _query_stac(
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
                post_metas = _query_stac(
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
            logger.info(
                "    Completato: %d scene processate, %d alert",
                result.get("scenes_processed", 0),
                result.get("alerts", 0),
            )
        except Exception as exc:
            logger.error("    Errore pipeline per AOI '%s': %s", aoi_name, exc,
                         exc_info=True)
            had_failures = True

    if had_failures:
        sys.exit(1)

