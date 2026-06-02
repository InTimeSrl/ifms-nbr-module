"""
events.py -- Gestione eventi di incendio.

Supporta multi-event per tile/AOI:
- Ogni cluster bruciato spazialmente separato apre il proprio evento.
- Nella finestra evento, i pixel burnt vengono assegnati all'evento il cui
  footprint e' piu' prossimo (dilation-based); i pixel orfani aprono nuovi
  eventi se passano i filtri area+solidity.
- Ogni evento ha il proprio contatore n_valid_scenes indipendente.

Persistenza:
    <output_dir>/events_index.json
    <output_dir>/<event_id>/
        burnt_count.tif (uint8)   -- somma scene in cui ogni pixel e' bruciato
        obs_count.tif   (uint8)   -- numero di scene valide su ogni pixel
        max_dnbr.tif    (float32) -- dNBR massimo per pixel sull'evento
        (+ raster per-scena: *_dNBR.tif, *_severity_prelim.tif, severity_final.tif)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_opening, label
from scipy.spatial import ConvexHull, cKDTree

from . import config
from . import data_io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_scene_ts(dt_str):
    """Converte datetime ISO in formato compatto YYYYMMDDTHHMMSS."""
    if not dt_str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    s = dt_str.replace("Z", "")
    if "+" in s:
        s = s[:s.index("+")]
    s = s.replace("-", "").replace(":", "").replace(" ", "T")
    if "." in s:
        s = s.split(".")[0]
    if "T" not in s:
        s = s + "T000000"
    return s  # es. "20250814T091700"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def event_dir(event_id, output_dir):
    """Sidecar folder dell'evento (accumulatori + raster per-scena)."""
    p = Path(output_dir) / event_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def event_paths(event_id, output_dir):
    d = event_dir(event_id, output_dir)
    return {
        "burnt_count": d / "burnt_count.tif",
        "obs_count":   d / "obs_count.tif",
        "max_dnbr":    d / "max_dnbr.tif",
    }


def index_path(output_dir):
    return Path(output_dir) / "events_index.json"


# ---------------------------------------------------------------------------
# Indice eventi (atomic JSON I/O)
# ---------------------------------------------------------------------------

def load_index(output_dir):
    p = index_path(output_dir)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Indice eventi corrotto (%s), ricreo vuoto", exc)
        return {}


def save_index(index, output_dir):
    p = index_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Lookup evento attivo
# ---------------------------------------------------------------------------

def get_active_events(output_dir, tile, aoi):
    """Restituisce la lista degli event_id aperti per (tile, aoi), ordinata per data."""
    idx = load_index(output_dir)
    candidates = [
        (eid, meta) for eid, meta in idx.items()
        if meta.get("status") == "open"
        and meta.get("tile") == tile
        and meta.get("aoi") == aoi
    ]
    candidates.sort(key=lambda kv: kv[1].get("alert_date", ""))
    return [eid for eid, _ in candidates]


def list_active_events(output_dir, tile=None, aoi=None):
    idx = load_index(output_dir)
    return [
        eid for eid, meta in idx.items()
        if meta.get("status") == "open"
        and (tile is None or meta.get("tile") == tile)
        and (aoi is None or meta.get("aoi") == aoi)
    ]


# ---------------------------------------------------------------------------
# I/O accumulator raster (1 read + 1 write per scena)
# ---------------------------------------------------------------------------

def _read_accumulator(path, shape, dtype):
    if Path(path).exists():
        with rasterio.open(path) as src:
            return src.read(1).astype(dtype, copy=False)
    return np.zeros(shape, dtype=dtype)


def load_accumulators(event_id, output_dir, shape):
    """Carica i 3 raster accumulator in RAM.

    Returns
    -------
    dict
        {"burnt_count": uint8, "obs_count": uint8, "max_dnbr": float32}
    """
    p = event_paths(event_id, output_dir)
    return {
        "burnt_count": _read_accumulator(p["burnt_count"], shape, np.uint8),
        "obs_count": _read_accumulator(p["obs_count"], shape, np.uint8),
        "max_dnbr": _read_accumulator(p["max_dnbr"], shape, np.float32),
    }


def save_accumulators(event_id, output_dir, accs, profile):
    """Scrive i 3 raster accumulator su disco (1 sola volta per scena)."""
    p = event_paths(event_id, output_dir)
    data_io.write_geotiff(accs["burnt_count"], profile, p["burnt_count"], dtype="uint8",   nodata=0)
    data_io.write_geotiff(accs["obs_count"],   profile, p["obs_count"],   dtype="uint8",   nodata=0)
    data_io.write_geotiff(accs["max_dnbr"],    profile, p["max_dnbr"],    dtype="float32", nodata=0)


# ---------------------------------------------------------------------------
# Cluster detection: find_clusters() + load_footprint_mask()
# ---------------------------------------------------------------------------

def find_clusters(burnt_mask, pixel_area_ha,
                  min_area_ha=None, min_solidity=None, separation_px=None,
                  opening_radius_px=None):
    """Individua cluster contigui in burnt_mask che passano i filtri.

    Parametri (se None: usa valori da config):
        min_area_ha      : area minima per cluster (ha)
        min_solidity     : rapporto area/convex_hull minimo
        separation_px    : distanza (px) sotto cui due cluster vengono fusi prima
                           dell'analisi (gestisce frammentazione da nuvole)
        opening_radius_px: raggio opening morfologico pre-labeling (rompe bridge di rumore).
                           0 = disabilitato. None = legge da config.

    Returns
    -------
    list[dict]
        Ogni elemento: {"mask": np.ndarray bool, "area_ha": float, "label_id": int}
        Ordinato per area decrescente.
    """
    if min_area_ha is None:
        min_area_ha = getattr(config, "MIN_ALERT_AREA_HA", 5.0)
    if min_solidity is None:
        min_solidity = getattr(config, "MIN_CLUSTER_SOLIDITY", None)
    if separation_px is None:
        separation_px = getattr(config, "MIN_CLUSTER_SEPARATION_PX", None)
    if opening_radius_px is None:
        opening_radius_px = int(getattr(config, "CLUSTER_OPENING_RADIUS_PX", 0))

    if not burnt_mask.any():
        return []

    # --- Opening morfologico pre-labeling ---
    # Rimuove pixel isolati e rompe bridge sottili di rumore tra aree bruciate
    # distinte. La labeling avviene sulla maschera "pulita"; i pixel rimossi
    # dall'erosione vengono recuperati via dilatazione prima di assegnare
    # il cluster all'evento (per non perdere pixel reali di bordo).
    if opening_radius_px > 0:
        r = int(opening_radius_px)
        kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        seed_mask = binary_opening(burnt_mask, structure=kernel)
    else:
        seed_mask = burnt_mask

    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connessi
    labeled, n = label(seed_mask, structure=structure)
    if n == 0:
        return []

    # --- Fusione cluster vicini (separation_px) ---
    if separation_px and separation_px > 0 and n > 1:
        # Centroide di ogni cluster (coordinate riga, colonna)
        centroids = {}
        for lbl in range(1, n + 1):
            ys, xs = np.where(labeled == lbl)
            if len(ys):
                centroids[lbl] = (float(ys.mean()), float(xs.mean()))

        lbl_ids = list(centroids.keys())
        coords = np.array([centroids[l] for l in lbl_ids])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=separation_px)

        # Union-Find per fondere cluster vicini
        parent = {l: l for l in lbl_ids}
        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in pairs:
            ri, rj = _find(lbl_ids[i]), _find(lbl_ids[j])
            if ri != rj:
                parent[rj] = ri

        # Remap etichette: ogni cluster ottiene l'etichetta del suo root
        remap = {l: _find(l) for l in lbl_ids}
        new_labeled = np.zeros_like(labeled)
        for old_lbl, new_lbl in remap.items():
            new_labeled[labeled == old_lbl] = new_lbl
        labeled = new_labeled
        unique_lbls = [l for l in np.unique(labeled) if l != 0]
    else:
        unique_lbls = list(range(1, n + 1))

    # --- Filtri area e solidity per ogni cluster ---
    # Alloca un array di priorità per il recovery: pixel di bordo vengono
    # assegnati al cluster più grande che li riclama (nessun duplicato).
    assigned = np.zeros(burnt_mask.shape, dtype=np.int32)  # 0 = non assegnato

    result = []
    for lbl in unique_lbls:
        seed_cl = (labeled == lbl)
        area_ha = float(seed_cl.sum()) * pixel_area_ha
        if area_ha < min_area_ha:
            continue

        # Solidity: calcolata sul seed (forma pulita dopo opening)
        if min_solidity and min_solidity > 0:
            try:
                ys, xs = np.where(seed_cl)
                pts = np.column_stack([ys, xs])
                if len(pts) >= 4:
                    hull = ConvexHull(pts)
                    hull_area_px = hull.volume  # in 2D volume = area
                    solidity = seed_cl.sum() / hull_area_px if hull_area_px > 0 else 1.0
                else:
                    solidity = 1.0
            except Exception:
                solidity = 1.0
            if solidity < min_solidity:
                logger.debug(
                    "Cluster %d scartato: solidity=%.2f < %.2f (area=%.1f ha)",
                    lbl, solidity, min_solidity, area_ha,
                )
                continue

        # Recovery pixel di bordo: dilata il seed e intereca con burnt originale
        if opening_radius_px > 0:
            expanded = binary_dilation(seed_cl, structure=kernel)
            event_mask = burnt_mask & expanded & (assigned == 0)
            event_mask |= seed_cl  # i seed sono sempre inclusi
        else:
            event_mask = seed_cl

        # Segna pixel assegnati per evitare duplicati con altri cluster
        assigned[event_mask] = lbl
        event_area_ha = float(event_mask.sum()) * pixel_area_ha
        result.append({"mask": event_mask, "area_ha": event_area_ha, "label_id": int(lbl)})

    result.sort(key=lambda d: d["area_ha"], reverse=True)
    return result


def load_footprint_mask(event_id, output_dir):
    """Carica la maschera burnt_count > 0 (footprint accumulato) per un evento.

    Returns np.ndarray bool o None se il file non esiste.
    """
    p = event_paths(event_id, output_dir)["burnt_count"]
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        arr = src.read(1)
    return arr > 0


# ---------------------------------------------------------------------------
# API alto livello: open / update / close
# ---------------------------------------------------------------------------

def open_event(burnt_mask, dnbr, valid_mask, profile, scene_meta,
               output_dir, tile, aoi):
    """Apre un nuovo evento dalla prima scena con burnt sopra soglia.

    Returns
    -------
    event_id : str
    """
    # event_id: {scene_ts}_EVT{N} -- N sequenziale per tile
    alert_date = scene_meta.get("datetime", scene_meta.get("date", ""))
    _ts = _format_scene_ts(alert_date)
    idx = load_index(output_dir)
    _n = sum(
        1 for m in idx.values()
        if m.get("tile") == tile and m.get("status") != "false_positive"
    ) + 1
    event_id = f"{_ts}_EVT{_n}"

    accs = {
        "burnt_count": burnt_mask.astype(np.uint8),
        "obs_count": valid_mask.astype(np.uint8),
        "max_dnbr": np.where(burnt_mask, dnbr.astype(np.float32), 0.0).astype(np.float32),
    }
    save_accumulators(event_id, output_dir, accs, profile)

    idx[event_id] = {
        "status": "open",
        "tile": tile,
        "aoi": aoi,
        "alert_date": alert_date,
        "alert_scene_id": scene_meta.get("stac_item_id"),
        "n_valid_scenes": 1,
        "n_detection_scenes": 1,
        "scenes": [{
            "scene_id": scene_meta.get("stac_item_id"),
            "datetime": alert_date,
            "is_detection": True,
            "cloud_cover": scene_meta.get("eo_cloud_cover"),
        }],
        "shape": list(burnt_mask.shape),
    }
    save_index(idx, output_dir)

    return event_id


def update_event(event_id, burnt_mask, dnbr, valid_mask, profile, scene_meta,
                 output_dir):
    """Aggiorna l'evento con una nuova scena valida.

    - Incrementa obs_count su tutti i pixel validi.
    - Incrementa burnt_count e aggiorna max_dnbr sui pixel bruciati.
    - n_valid_scenes += 1 (UNA volta per scena, non per cluster).
    """
    idx = load_index(output_dir)
    state = idx.get(event_id) or {}
    shape = tuple(state.get("shape") or burnt_mask.shape)

    # Idempotency: se questa scena e' gia' stata registrata (crash/interrupt
    # dopo save_index ma prima di update_watermark) non ri-incrementare.
    _scene_id = scene_meta.get("stac_item_id")
    if _scene_id and any(s.get("scene_id") == _scene_id for s in state.get("scenes", [])):
        logger.debug(
            "update_event %s: scena %s gia' registrata, skip (idempotency)",
            event_id, _scene_id,
        )
        return int(state.get("n_valid_scenes", 0))

    accs = load_accumulators(event_id, output_dir, shape)

    # obs_count: +1 sui pixel SCL validi (saturazione a 255)
    accs["obs_count"] = np.minimum(
        accs["obs_count"].astype(np.int16) + valid_mask.astype(np.int16), 255
    ).astype(np.uint8)

    # burnt_count: +1 sui pixel bruciati validi
    burnt_valid = burnt_mask & valid_mask
    accs["burnt_count"] = np.minimum(
        accs["burnt_count"].astype(np.int16) + burnt_valid.astype(np.int16), 255
    ).astype(np.uint8)

    # max_dnbr: aggiorna solo sui pixel bruciati validi se nuovo > vecchio
    upd = burnt_valid & (dnbr.astype(np.float32) > accs["max_dnbr"])
    accs["max_dnbr"] = np.where(upd, dnbr.astype(np.float32), accs["max_dnbr"])

    save_accumulators(event_id, output_dir, accs, profile)

    is_detection = bool(burnt_valid.any())
    state["n_valid_scenes"] = int(state.get("n_valid_scenes", 0)) + 1
    if is_detection:
        state["n_detection_scenes"] = int(state.get("n_detection_scenes", 0)) + 1
    state.setdefault("scenes", []).append({
        "scene_id": scene_meta.get("stac_item_id"),
        "datetime": scene_meta.get("datetime", scene_meta.get("date", "")),
        "is_detection": is_detection,
        "cloud_cover": scene_meta.get("eo_cloud_cover"),
    })
    idx[event_id] = state
    save_index(idx, output_dir)

    logger.debug(
        "Aggiornato evento %s con scena %s (n_valid=%d, det=%s, +%d px burnt)",
        event_id, scene_meta.get("stac_item_id"), state["n_valid_scenes"],
        is_detection, int(burnt_valid.sum()),
    )
    return state["n_valid_scenes"]


def should_close(event_id, output_dir, current_date=None):
    """Restituisce (close: bool, reason: str|None)."""
    state = load_index(output_dir).get(event_id)
    if state is None:
        return False, None

    if int(state.get("n_valid_scenes", 0)) >= config.EVENT_WINDOW_SCENES:
        return True, "window_scenes_reached"

    alert_str = state.get("alert_date", "")
    if alert_str:
        try:
            alert_dt = datetime.fromisoformat(alert_str.replace("Z", "+00:00"))
            ref_dt = (
                datetime.fromisoformat(current_date.replace("Z", "+00:00"))
                if current_date else datetime.utcnow()
            )
            if alert_dt.tzinfo is not None:
                alert_dt = alert_dt.replace(tzinfo=None)
            if ref_dt.tzinfo is not None:
                ref_dt = ref_dt.replace(tzinfo=None)
            elapsed = (ref_dt - alert_dt).days
            n_valid = int(state.get("n_valid_scenes", 0))
            # Soft timeout: si applica solo se abbiamo gia' abbastanza scene valide
            # per poter raggiungere EVENT_MIN_DETECTIONS per i pixel bruciati.
            # Se non bastano, la finestra si allunga fino all'hard cap.
            if elapsed >= config.EVENT_TIMEOUT_DAYS and n_valid >= config.EVENT_MIN_DETECTIONS:
                return True, "timeout_days_reached"
            # Hard cap assoluto: chiude sempre indipendentemente dalle scene valide.
            if elapsed >= config.EVENT_MAX_TIMEOUT_DAYS:
                return True, "timeout_days_reached"
        except ValueError:
            pass

    return False, None


def mark_closed(event_id, output_dir, reason, closed_date=None):
    idx = load_index(output_dir)
    if event_id in idx:
        idx[event_id]["status"] = "closed"
        idx[event_id]["closure_reason"] = reason
        idx[event_id]["closed_date"] = closed_date or datetime.utcnow().isoformat()
        save_index(idx, output_dir)


def purge_event(event_id, output_dir):
    """Elimina la sidecar folder di un falso positivo.

    La voce nell'indice viene mantenuta con status='false_positive'
    per preservare la sequenza EVT{N} ed evitare riassegnazioni di
    numeri a eventi ancora aperti.
    """
    sidecar = event_dir(event_id, output_dir)
    if sidecar.exists():
        shutil.rmtree(sidecar)

    idx = load_index(output_dir)
    if event_id in idx:
        idx[event_id]["status"] = "false_positive"
        save_index(idx, output_dir)

    logger.info("Evento %s eliminato (falso positivo)", event_id)


# ---------------------------------------------------------------------------
# Cluster contiguo: filtro apertura eventi
# ---------------------------------------------------------------------------

def largest_cluster_area_ha(burnt_mask, pixel_area_ha):
    """Area (ha) del piu' grande cluster contiguo (8-connessi) in burnt_mask.

    Serve a discriminare un vero incendio (cluster compatto) dal rumore
    post-fire (pixel sparsi). Ritorna 0.0 se la maschera e' vuota.
    """
    if not burnt_mask.any():
        return 0.0
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connessi
    labeled, n = label(burnt_mask, structure=structure)
    if n == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    return float(counts.max() * pixel_area_ha)
