"""
baseline.py -- Baseline NBR retrospettiva (pre-campagna).

One-shot: recupera scene dalla finestra di lookback, filtra, calcola NBR,
costruisce median composite con filtro anomalie MAD, salva su disco
(baseline_nbr, previous_nbr, max_dnbr).
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config
from . import data_io
from . import preprocess, indices

logger = logging.getLogger(__name__)


def _nanmedian_no_allnan_warning(arr, axis=0):
    """Compute nanmedian silencing expected all-NaN slice warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


# ---------------------------------------------------------------------------
# Path dei file NBR persistenti per un'AOI
# ---------------------------------------------------------------------------

def nbr_paths(aoi, data_dir="data"):
    """Restituisce i path dei file NBR persistenti per un'AOI.

    Parameters
    ----------
    aoi : dict
        AOI dict (da data_io.load_aoi).
    data_dir : str o Path
        Cartella radice dati persistenti.

    Returns
    -------
    dict
        {"baseline": Path, "previous": Path, "max_dnbr": Path}
    """
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "baseline": d / "baseline_nbr.tif",
        "previous": d / "previous_nbr.tif",
        "max_dnbr": d / "max_dnbr.tif",
    }


def load_nbr(path):
    """Carica un raster NBR da disco. Restituisce (array, profile) o (None, None)."""
    if Path(path).exists():
        return data_io.read_band(str(path))
    return None, None


def save_nbr(data, profile, path):
    """Salva un raster NBR su disco."""
    data_io.write_geotiff(data, profile, path, dtype="float32")


# ---------------------------------------------------------------------------
# Filtri e calcolo NBR da scena singola
# ---------------------------------------------------------------------------

def _filter_baseline_scenes(scenes):
    """Filtra scene per la costruzione della baseline (solo qualita').

    Usa gli stessi criteri di qualita' di pipeline._filter_scenes, ma senza
    controllare lo stato di processamento (le scene pre-campagna
    non entrano nel loop di monitoraggio).
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
    """Carica una scena e restituisce (nbr, nir, valid_mask, profile) o None.

    Parameters
    ----------
    scene : dict
        Metadati della scena.
    aoi : dict
        AOI dict.
    scene_dir : str o Path
        Cartella locale dei TIF.

    Returns
    -------
    tuple (nbr, nir, valid_mask, profile) o None
        None solo se la scena e' completamente priva di pixel validi
        (caso degenere). Il filtro di qualita' operativo e'
        ``SCENE_VALID_SCL_PCT`` applicato a valle nel monitoring.
    """
    # Rileva CRS reale dalla prima banda raster (non dai metadati)
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
# Costruzione baseline retrospettiva
# ---------------------------------------------------------------------------

def build_baseline(aoi, get_scenes_fn, scene_dir=None, data_dir="data"):
    """Costruisce la baseline NBR retrospettiva per un'AOI.

    Recupera scene dalla finestra PRE-campagna, costruisce un median
    composite con filtro anomalie MAD, e salva baseline_nbr.tif +
    baseline_count.tif rimosso. Inizializza previous_nbr e max_dnbr.

    Parameters
    ----------
    aoi : dict
        AOI dict (da data_io.load_aoi).
    get_scenes_fn : callable
        Funzione per recuperare le scene: get_scenes_fn(date_from) -> list[dict].
        Deve restituire scene per l'AOI a partire dalla data indicata.
    scene_dir : str o Path, optional
        Cartella locale con scene di test.
    data_dir : str, optional
        Cartella dati persistenti (default: "data").

    Returns
    -------
    baseline_nbr : np.ndarray
        Median composite (con filtro anomalie).
    profile : dict
        Profilo rasterio del composite.

    Raises
    ------
    RuntimeError
        Se non ci sono abbastanza scene cloud-free nella finestra.
    """
    aoi_name = aoi["name"]
    paths = nbr_paths(aoi, data_dir)
    logger.info("Costruzione baseline retrospettiva per AOI '%s'", aoi_name)

    # --- Calcola finestra di lookback ---
    # CAMPAIGN_START_DATE = None → usa oggi come inizio campagna
    if config.CAMPAIGN_START_DATE is None:
        campaign_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        campaign_start = datetime.fromisoformat(config.CAMPAIGN_START_DATE)
    lookback_start = campaign_start - timedelta(days=config.BASELINE_LOOKBACK_DAYS)

    all_scenes = get_scenes_fn(date_from=lookback_start.isoformat())

    # Tieni solo scene nella finestra pre-campagna
    campaign_str = config.CAMPAIGN_START_DATE
    pre_scenes = [
        s for s in all_scenes
        if s.get("datetime", s.get("date", "")) < campaign_str
    ]
    pre_scenes = _filter_baseline_scenes(pre_scenes)

    logger.info(
        "Baseline: %d scene pre-campagna trovate (finestra %s -> %s)",
        len(pre_scenes), lookback_start.date(), campaign_start.date(),
    )

    if len(pre_scenes) < config.BASELINE_MIN_SCENES:
        raise RuntimeError(
            f"AOI '{aoi_name}': solo {len(pre_scenes)} scene pre-campagna, "
            f"servono almeno {config.BASELINE_MIN_SCENES}. "
            f"Ampliare BASELINE_LOOKBACK_DAYS o verificare i dati."
        )

    # --- Costruisci stack NBR ---
    stack_list = []
    last_profile = None
    for i, scene in enumerate(pre_scenes, 1):
        logger.info(
            "  [%d/%d] download scena baseline: %s",
            i, len(pre_scenes), scene["stac_item_id"],
        )
        result = compute_nbr_from_scene(scene, aoi, scene_dir)
        if result is None:
            logger.info("  [%d/%d] skip %s: nessun pixel valido",
                        i, len(pre_scenes), scene["stac_item_id"])
            continue
        nbr, _nir, _swir, valid_mask, profile = result
        layer = np.full_like(nbr, np.nan)
        layer[valid_mask] = nbr[valid_mask]
        stack_list.append(layer)
        last_profile = profile

    if len(stack_list) < config.BASELINE_MIN_SCENES:
        raise RuntimeError(
            f"AOI '{aoi_name}': solo {len(stack_list)} scene utilizzabili "
            f"(dopo filtro qualita'), servono almeno {config.BASELINE_MIN_SCENES}."
        )

    stack = np.array(stack_list, dtype="float32")  # (N, H, W)

    # --- Filtro anomalie MAD: scarta osservazioni anomalmente basse ---
    # Per ogni pixel, calcola mediana e MAD (Median Absolute Deviation)
    # lungo l'asse temporale. Il MAD e' robusto agli outlier (breakdown
    # point 50%), a differenza della deviazione standard (breakdown 0%).
    # Rif: Leys et al. 2013, Rousseeuw & Croux 1993.
    pixel_median = _nanmedian_no_allnan_warning(stack, axis=0)
    pixel_mad = _nanmedian_no_allnan_warning(
        np.abs(stack - pixel_median[np.newaxis, :, :]), axis=0,
    )
    # Pavimento: se MAD < eps, le osservazioni sono quasi identiche -> non filtrare
    pixel_mad = np.maximum(pixel_mad, config.BASELINE_MAD_FLOOR)
    k = config.BASELINE_MAD_K
    threshold = pixel_median - k * pixel_mad
    anomaly_mask = stack < threshold[np.newaxis, :, :]
    n_anomalies = np.count_nonzero(anomaly_mask)
    if n_anomalies > 0:
        stack[anomaly_mask] = np.nan
        logger.info(
            "Filtro anomalie MAD: rimossi %d pixel/scena (%.2f%% dello stack)",
            n_anomalies, 100.0 * n_anomalies / stack.size,
        )

    # --- Median composite ---
    baseline_nbr = _nanmedian_no_allnan_warning(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0)

    # --- Salva su disco ---
    paths["baseline"].parent.mkdir(parents=True, exist_ok=True)
    save_nbr(baseline_nbr, last_profile, paths["baseline"])

    # previous_nbr = baseline_nbr (FOTO_VERDE)
    save_nbr(baseline_nbr.copy(), last_profile, paths["previous"])

    # max_dnbr = 0 (nessun danno registrato)
    save_nbr(np.zeros_like(baseline_nbr), last_profile, paths["max_dnbr"])

    logger.info(
        "Baseline completata: %d scene, mediana osservazioni/pixel: %.0f",
        len(stack_list),
        np.median(count[count > 0]) if (count > 0).any() else 0,
    )
    return baseline_nbr, last_profile
