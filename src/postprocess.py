"""
postprocess.py -- Filtri morfologici e vettorializzazione poligoni bruciati.

Operazioni:
  1. Sieve: rimozione patch isolate < MIN_PATCH_PIXELS (default 4 px = 0.16 ha)
  2. Fill holes: riempimento buchi < HOLE_FILL_PIXELS nei poligoni
  3. Vectorize: conversione raster severita' -> poligoni GeoJSON con attributi
     (classe, label, area_ha)
  4. RGB HONC: composito Highlight Optimized Natural Color per visualizzazione

Usato da pipeline.py (process_scene) quando viene rilevato un incendio.

Ref: progetto tecnico, fase post-processing.
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
    """Applica filtri morfologici alla mappa di severita'.

    Passi:
    1. Sieve: rimuove patch bruciati (severita' >= 4) con meno di
       MIN_PATCH_PIXELS pixel contigui.
    2. Fill holes: riempie buchi interni alle aree bruciate con meno
       di HOLE_FILL_PIXELS pixel.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Mappa di severita' (0=nodata, 1-7=classi USGS).
    valid_mask : np.ndarray (bool)
        Maschera pixel validi.

    Returns
    -------
    np.ndarray (uint8)
        Mappa di severita' filtrata.
    """
    out = severity.copy()

    # --- Sieve: rimuovi patch bruciati troppo piccoli ---
    # Lavora sulla maschera binaria burnt (severita' >= 4) per il sieve,
    # poi riporta le classi originali solo dove sopravvive
    burnt = (out >= 4).astype("uint8")
    if burnt.any():
        burnt_sieved = sieve(burnt, size=config.MIN_PATCH_PIXELS, connectivity=8)
        # Azzera classi nei pixel rimossi dal sieve
        removed = (burnt == 1) & (burnt_sieved == 0)
        out[removed] = 3  # Unburned

    # --- Fill holes: riempi piccoli buchi interni alle aree bruciate ---
    burnt = (out >= 4)
    if burnt.any():
        filled = _fill_small_holes(burnt, max_pixels=config.HOLE_FILL_PIXELS)
        # Pixel aggiunti dal fill: assegna la classe del vicino piu' comune
        new_pixels = filled & ~burnt & valid_mask
        if new_pixels.any():
            out[new_pixels] = _assign_neighbor_class(out, new_pixels)

    return out


def vectorize(severity, profile):
    """Converte la mappa di severita' in poligoni GeoJSON.

    Genera un poligono per ogni regione contigua con severita' >= 4
    (Low Severity e superiore). Ogni feature ha attributi:
    - class_id: codice severita' (4-7)
    - class_label: nome della classe USGS
    - area_ha: area del poligono in ettari

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Mappa di severita' filtrata (da morphological_filter).
    profile : dict
        Profilo rasterio (contiene transform e CRS).

    Returns
    -------
    list[dict]
        Lista di feature GeoJSON (geometry + properties).
    """
    transform = profile["transform"]
    # Risoluzione pixel in metri (per calcolo area)
    pixel_res = abs(transform.a)  # dimensione pixel in m
    pixel_area_ha = (pixel_res * pixel_res) / 10000.0

    features = []
    # Estrai solo pixel bruciati (class >= 4)
    burnt_mask = (severity >= 4).astype("uint8")
    if not burnt_mask.any():
        return features

    for geom_dict, class_val in shapes(severity, mask=burnt_mask, transform=transform):
        class_id = int(class_val)
        if class_id < 4:
            continue

        geom = shape(geom_dict)
        class_info = config.SEVERITY_CLASSES.get(class_id, {})

        # Area: conta pixel nel poligono (approssimazione dal raster)
        n_pixels = geom.area / (pixel_res * pixel_res)
        area_ha = n_pixels * pixel_area_ha

        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "class_id": class_id,
                "class_label": class_info.get("label", ""),
                "area_ha": round(area_ha, 2),
            },
        })

    return features


def vectorize_by_severity(severity, dnbr, profile, meta=None):
    """Dissolve per classe di severita' -> 1 feature MultiPolygon per classe.

    Geometrie nel CRS nativo del raster (nessuna riproiezione).
    Include classi 3-7 (Unburned + tutte le severita' di bruciatura).
    Classi 1-2 (Enhanced Regrowth) escluse: la ricrescita vegetazionale
    e' un fenomeno a lungo termine, non rilevante nel monitoraggio NRT.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Mappa di severita' filtrata (classi 3-7).
    dnbr : np.ndarray (float32)
        Mappa dNBR (o RBR) corrispondente.
    profile : dict
        Profilo rasterio (contiene transform e CRS).
    meta : dict, optional
        Attributi globali da aggiungere a ogni feature:
        - event_id, detection_date, processing_mode, aoi_ref,
          cloud_cover_pct, index_mode, index_threshold, satellite

    Returns
    -------
    list[dict]
        Lista di feature GeoJSON (una per classe di severita' presente).
    """
    if meta is None:
        meta = {}

    transform = profile["transform"]
    pixel_res = abs(transform.a)
    pixel_area_ha = (pixel_res * pixel_res) / 10000.0

    # Verifica che ci siano pixel bruciati (>= 4) prima di procedere
    burnt_mask = (severity >= 4).astype("uint8")
    if not burnt_mask.any():
        return []

    features = []
    for class_id in sorted(config.SEVERITY_CLASSES.keys()):
        if class_id < 4:  # escludi ER-H, ER-L, Unburned
            continue
        class_mask = (severity == class_id)
        if not class_mask.any():
            continue

        # Dissolve: raccogli tutte le geometrie di questa classe e unisci
        class_u8 = class_mask.astype("uint8")
        geoms = []
        for geom_dict, val in shapes(class_u8, mask=class_u8,
                                     transform=transform):
            if val == 1:
                geoms.append(shape(geom_dict))
        if not geoms:
            continue
        merged = unary_union(geoms)

        # Statistiche dNBR per questa classe
        dnbr_vals = dnbr[class_mask & np.isfinite(dnbr)]
        n_pixels = int(class_mask.sum())

        class_info = config.SEVERITY_CLASSES.get(class_id, {})
        props = {
            "event_id": meta.get("event_id", ""),
            "detection_date": meta.get("detection_date", ""),
            "closed_date": meta.get("closed_date", ""),
            "closure_reason": meta.get("closure_reason", ""),
            "satellite": meta.get("satellite", ""),
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
# Funzioni interne
# ---------------------------------------------------------------------------

def filter_small_clusters(severity, pixel_res_m, min_area_ha):
    """Removes burnt clusters smaller than min_area_ha from a severity map.

    All connected components (8-connected) with area >= min_area_ha are kept,
    regardless of their position. Isolated pixels and small clusters are zeroed.
    This is applied to severity_final before vectorisation and footprint generation
    so that all output products (burnt_final, fire_footprint, severity_final.tif)
    are consistent and noise-free.

    Parameters
    ----------
    severity : np.ndarray (uint8, 2D)
        Severity map (0 = nodata, 1-7 = USGS classes).
    pixel_res_m : float
        Pixel resolution in metres (typically 20.0 for Sentinel-2).
    min_area_ha : float
        Minimum cluster area in hectares to keep.

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
    """Riempi buchi interni a una maschera binaria se < max_pixels.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Maschera binaria (True = area bruciata).
    max_pixels : int
        Dimensione massima buchi da riempire.

    Returns
    -------
    np.ndarray (bool)
        Maschera con buchi piccoli riempiti.
    """
    filled = binary_fill_holes(mask)
    if max_pixels is None:
        return filled

    # Identifica buchi riempiti e controlla dimensione
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
    """Assegna ai pixel nuovi la classe del vicino bruciato piu' comune.

    Parameters
    ----------
    severity : np.ndarray (uint8)
        Mappa severita' corrente.
    new_pixels : np.ndarray (bool)
        Pixel appena aggiunti dal fill.

    Returns
    -------
    np.ndarray (uint8)
        Classi da assegnare ai new_pixels (stesso shape di new_pixels sommato).
    """
    # Dilata le classi bruciate e prendi il valore piu' alto nel vicinato 3x3
    burnt_classes = severity.copy()
    burnt_classes[burnt_classes < 4] = 0
    neighbor_class = maximum_filter(burnt_classes, size=3)
    return neighbor_class[new_pixels]


# ---------------------------------------------------------------------------
# RGB HONC composito (output visualizzazione, opzionale)
# ---------------------------------------------------------------------------

def save_rgb_composite(scene, aoi, scene_dir, out_dir):
    """Produce e salva composito RGB Highlight Optimized Natural Color (HONC).

    Carica le bande B4, B3, B2 a 10 m nativi, applica stretch percentile
    (p2-p98) + gamma 0.8 e salva un GeoTIFF uint8 a 3 bande.

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
