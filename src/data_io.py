"""
data_io.py -- I/O scene Sentinel-2, AOI, output raster/vector.
Usa rasterio (GDAL) per raster, fiona per shapefile, JSON per metadati.

Cartella di output:
    Il percorso radice degli output si imposta tramite l'argomento --output-root
    di run.py (default: "output/", relativa alla CWD del progetto).
    Per cambiare la cartella di default in modo permanente, modificare in
    src/pipeline.py la riga:
        "--output-root", default="output",
    La struttura creata automaticamente e':
        <output-root>/<nome_aoi>/data/      <- baseline_nbr, previous_nbr, pipeline_state.json
        <output-root>/<nome_aoi>/products/  <- GeoPackage, severity_final.tif, ...

Parallelismo AOI (opzione consigliata):
    Le AOI sono completamente indipendenti su disco (data_dir e output_dir
    separate per nome AOI). Per processarle in parallelo senza modifiche al
    codice e' sufficiente lanciare processi OS separati usando il flag --aoi:

        # Linux: avvia N AOI in parallelo, ognuna con il proprio log
        python run.py --aoi AOI_A > output_AOI_A.log 2>&1 &
        python run.py --aoi AOI_B > output_AOI_B.log 2>&1 &

    Nessun lock necessario: pipeline_state.json, events_index.json e i raster
    NBR sono tutti per-tile/per-AOI.
    N.B.: Possibile collo di bottiglia sulla
    banda di rete verso il provider STAC/COG: piu' di 3-4 AOI
    simultanee potrebbero portare a saturazione la connessione o incorrere in rate limiting.
"""

import json
import logging
import os
from pathlib import Path

import fiona
import numpy as np
import rasterio
import rasterio.windows as _rwin
from pystac_client import Client
from pyproj import Transformer
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione GDAL Virtual File Systems per lettura COG remoti
# Ref: https://gdal.org/en/stable/user/virtual_file_systems.html
#
# /vsicurl/  -> lettura HTTP/HTTPS (AWS Element84, CDSE)
# /vsis3/    -> lettura S3 diretta (STAC interno, MinIO)
#
# rasterio.open() attiva automaticamente il VFS appropriato in base
# al protocollo dell'URL (https:// -> vsicurl, s3:// -> vsis3).
# Le opzioni sotto sono ottimizzate per COG con range requests.
# ---------------------------------------------------------------------------

def configure_gdal_vfs():
    """Imposta le variabili d'ambiente GDAL per lettura di COG remoti."""
    gdal_opts = {
        # Disabilita la scansione della directory remota all'apertura
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # Restringe vsicurl alle sole estensioni .tif/.tiff
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        # Unisce range request HTTP adiacenti in un'unica richiesta
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_CACHE_SIZE": "67108864",  # 64 MB cache
        # Accesso anonimo per bucket S3 pubblici (Element84).
        # TODO (STAC interno / MinIO): sostituire con:
        #   "AWS_S3_ENDPOINT":     "<host:porta>",   # es. "minio.internal:9000"
        #   "AWS_VIRTUAL_HOSTING": "FALSE",           # MinIO usa path-style URL (non virtual-hosted)
        #   "AWS_HTTPS":           "NO",              # solo se il MinIO non usa TLS
        # Se invece ha TLS con certificato self-signed, lasciare AWS_HTTPS=YES e aggiungere:
        #   "GDAL_HTTP_UNSAFESSL": "YES"              # disabilita la verifica del certificato TLS:
        #                                             # accetta cert. self-signed/scaduti/CN errato
        # Rimuovere la riga:
        #   "AWS_NO_SIGN_REQUEST" (bucket con richieste firmate).
        # Le credenziali (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) NON vanno nel codice:
        # GDAL/vsis3 le legge automaticamente dall'ambiente del processo.
        # Gli altri quattro parametri (GDAL_DISABLE_READDIR_ON_OPEN, ecc.) dovrebbero rimanere invariati.
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    for key, val in gdal_opts.items():
        os.environ.setdefault(key, val)


configure_gdal_vfs()


# ---------------------------------------------------------------------------
# Lettura metadati
# ---------------------------------------------------------------------------

def load_metadata(json_path):
    """Carica il JSON di metadati di una scena.

    Parameters
    ----------
    json_path : str o Path
        Percorso al file *_metadata.json.

    Returns
    -------
    dict
        Contenuto del JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_scenes(tile_dir):
    """Elenca tutte le scene (JSON) in una cartella tile.

    Parameters
    ----------
    tile_dir : str o Path
        Cartella contenente i file di una o piu' scene.

    Returns
    -------
    list[dict]
        Lista di metadati, ordinata per datetime.
    """
    tile_dir = Path(tile_dir)
    scenes = []
    for jp in tile_dir.glob("*_metadata.json"):
        scenes.append(load_metadata(jp))
    scenes.sort(key=lambda s: s["datetime"])
    return scenes


# ---------------------------------------------------------------------------
# Lettura bande raster
# ---------------------------------------------------------------------------

def _resolve_href(meta, asset_key):
    """Restituisce il percorso/URL della banda da aprire con rasterio.

    Se l'href e' un URL remoto (http/s3), lo restituisce cosi' com'e'.
    Se e' un path relativo o locale, lo lascia invariato (rasterio lo gestisce).
    """
    asset = meta["assets"][asset_key]
    return asset["href"]


def _local_path(meta, asset_key, scene_dir):
    """Costruisce il path locale del TIF a partire da metadati e cartella.

    Usato come fallback quando l'href punta a un URL remoto ma il file
    e' gia' presente su disco (es. dati di test scaricati).
    """
    item_id = meta["stac_item_id"]
    band_name = meta["assets"][asset_key]["band_name"]
    return Path(scene_dir) / f"{item_id}_{band_name}.tif"


def _resolve_source(meta, asset_key, scene_dir):
    """Restituisce il path effettivo di una banda.

    Logica di risoluzione:
    1. Se scene_dir e' fornito e il file locale esiste -> path locale
    2. Altrimenti -> href dal JSON STAC (URL HTTP o s3://)
       rasterio/GDAL lo apre via /vsicurl/ o /vsis3/ automaticamente.
    """
    if scene_dir is not None:
        local = _local_path(meta, asset_key, scene_dir)
        if local.exists():
            return str(local)
    href = _resolve_href(meta, asset_key)
    return href


def get_scene_crs(meta, scene_dir=None):
    """Restituisce il CRS di una scena leggendo l'header della prima banda raster.

    Parameters
    ----------
    meta : dict
        Metadati della scena.
    scene_dir : str o Path, optional
        Cartella locale dei TIF. Se None, legge da remoto (COG via VFS).

    Returns
    -------
    str
        CRS come stringa (es. "EPSG:32634").
    """
    source = _resolve_source(meta, "nir08", scene_dir)
    with rasterio.open(source) as src:
        return str(src.crs)


def read_band(source, bbox=None):
    """Legge una singola banda raster.

    Parameters
    ----------
    source : str
        Path locale, URL HTTP o S3 del file raster.
    bbox : tuple, optional
        (xmin, ymin, xmax, ymax) per ritaglio spaziale. Deve essere
        nello stesso CRS del raster. Se None, legge l'intero tile.

    Returns
    -------
    data : np.ndarray
        Array 2D (height, width), dtype originale (uint16 o uint8).
    profile : dict
        Profilo rasterio (CRS, transform, dtype, ecc.).
    """
    with rasterio.open(source) as src:
        if bbox is not None:
            window = from_bounds(*bbox, transform=src.transform)
            # Clamp alla griglia pixel intera e ai limiti del raster.
            # from_bounds può produrre offset negativi o fuori extent
            # se il bbox è più grande del tile.
            int_window = window.round_offsets().round_lengths()
            int_window = int_window.intersection(
                Window(0, 0, src.width, src.height)
            )
            data = src.read(1, window=int_window)
            transform = src.window_transform(int_window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=transform,
            )
        else:
            data = src.read(1)
            profile = src.profile.copy()
    return data, profile


def load_scene_bands(meta, scene_dir=None, asset_keys=None, bbox=None):
    """Carica tutte le bande di una scena come dizionario di array.

    Supporta lettura locale (scene_dir) o remota (COG via GDAL VFS).
    Se scene_dir e' None, i raster vengono letti direttamente dall'href
    STAC tramite /vsicurl/ (HTTP) o /vsis3/ (S3).

    Parameters
    ----------
    meta : dict
        Metadati della scena (da load_metadata).
    scene_dir : str o Path, optional
        Cartella locale dei TIF. Se None, legge da remoto.
    asset_keys : list[str], optional
        Chiavi asset da caricare (es. ["nir08", "swir22", "scl"]).
        Se None, carica tutte le bande in config.BANDS.
    bbox : tuple, optional
        Ritaglio spaziale (xmin, ymin, xmax, ymax) nel CRS del raster.

    Returns
    -------
    bands : dict[str, np.ndarray]
        Dizionario {asset_key: array 2D}.
    profile : dict
        Profilo rasterio (dalla prima banda letta).
    """
    if asset_keys is None:
        asset_keys = list(config.BANDS.keys())

    bands = {}
    ref_profile = None

    for key in asset_keys:
        if key not in meta["assets"]:
            continue

        source = _resolve_source(meta, key, scene_dir)

        data, profile = read_band(source, bbox=bbox)
        bands[key] = data

        if ref_profile is None:
            ref_profile = profile

    return bands, ref_profile


# ---------------------------------------------------------------------------
# Scrittura output
# ---------------------------------------------------------------------------

def _write_tiff(out_path, profile, write_fn):
    """Primitiva interna: scrive su file temporaneo e sostituisce out_path (pattern tmp->replace).

    Parameters
    ----------
    out_path : Path
        Percorso di destinazione (la directory deve esistere).
    profile : dict
        Profilo rasterio per il file di output.
    write_fn : callable
        ``write_fn(dst)`` scrive i dati nel dataset rasterio aperto.
    """
    tmp_path = out_path.with_suffix(".tmp.tif")
    tmp_path.unlink(missing_ok=True)
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            write_fn(dst)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp_path, out_path)


def write_geotiff(data, profile, out_path, dtype=None, nodata=None):
    """Salva un array 2D come GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        Array 2D da salvare.
    profile : dict
        Profilo rasterio (CRS, transform, ecc.).
    out_path : str o Path
        Percorso di output.
    dtype : str, optional
        Tipo dato output (es. "float32", "uint8"). Se None, usa quello di data.
    nodata : number, optional
        Valore nodata. Se None, usa config.NODATA_OUTPUT.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is None:
        dtype = data.dtype
    if nodata is None:
        nodata = config.NODATA_OUTPUT

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
    )

    _write_tiff(out_path, out_profile, lambda dst: dst.write(data.astype(dtype), 1))


def write_rgb_geotiff(data, profile, out_path, nodata=None):
    """Salva un array (3, H, W) come GeoTIFF RGB (uint8).

    Parameters
    ----------
    data : np.ndarray
        Array 3D (3, height, width) con bande R, G, B (uint8, 0-255).
    profile : dict
        Profilo rasterio (CRS, transform, ecc.).
    out_path : str o Path
        Percorso di output.
    nodata : int, optional
        Valore nodata. Se None, non imposta nodata sul GeoTIFF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.pop("nodata", None)
    out_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=3,
        compress="deflate",
    )
    if nodata is not None:
        out_profile["nodata"] = nodata

    def _write_rgb(dst):
        for i in range(3):
            dst.write(data[i], i + 1)

    _write_tiff(out_path, out_profile, _write_rgb)


def write_geopackage(features, out_path, crs=None, layer_name="burnt"):
    """Salva una lista di feature in un GeoPackage (.gpkg).

    Tipo geometria fissato a MultiPolygon

    Parameters
    ----------
    features : list[dict]
        Lista di feature in formato GeoJSON-like (geometry + properties).
    out_path : str o Path
        Percorso del file .gpkg di output (sovrascritto se esistente).
    crs : str, optional
        Codice EPSG (es. "EPSG:32635"). Se None, nessun CRS scritto.
    layer_name : str
        Nome del layer all'interno del GeoPackage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not features:
        return

    # Costruisce schema dalle proprieta' della prima feature.
    # I tipi fiona supportati: int, float, str, bool, date, datetime.
    sample_props = features[0].get("properties", {}) or {}
    prop_schema = {}
    for key, val in sample_props.items():
        if isinstance(val, bool):
            prop_schema[key] = "bool"
        elif isinstance(val, int):
            prop_schema[key] = "int"
        elif isinstance(val, float):
            prop_schema[key] = "float"
        else:
            prop_schema[key] = "str"

    schema = {"geometry": "MultiPolygon", "properties": prop_schema}

    # fiona in modalita' "w" su GPKG esistente appende invece di sovrascrivere:
    # su run ripetuti (es. restart dopo interruzione) il layer si duplicherebbe.
    # Rimuove solo il layer target; gli altri layer del file restano intatti.
    if out_path.exists():
        try:
            if layer_name in fiona.listlayers(str(out_path)):
                fiona.remove(str(out_path), layer=layer_name)
        except Exception:
            pass

    with fiona.open(
        out_path, "w",
        driver="GPKG",
        schema=schema,
        crs=crs,
        layer=layer_name,
    ) as dst:
        for feat in features:
            geom = feat.get("geometry")
            if geom is None:
                continue
            # Promuovi Polygon -> MultiPolygon per uniformita' di schema
            if geom.get("type") == "Polygon":
                geom = {"type": "MultiPolygon", "coordinates": [geom["coordinates"]]}
            # Coerce property values al tipo dichiarato (None resta None).
            props = {}
            for key, decl in prop_schema.items():
                v = feat.get("properties", {}).get(key)
                if v is None:
                    props[key] = None
                elif decl == "str":
                    props[key] = str(v)
                else:
                    props[key] = v
            dst.write({"geometry": geom, "properties": props})


# ---------------------------------------------------------------------------
# Lettura AOI
# ---------------------------------------------------------------------------
# Le AOI vengono scoperte automaticamente da pipeline.py (_scan_aois) che
# scansiona la cartella AOIs/ (o quella passata con --aois-root).
# Struttura attesa:
#
#   AOIs/
#     <nome_aoi>/
#       <file_vettoriale>.<ext>    <- .geojson | .gpkg | .shp | .kml | .gml
#
# Per aggiungere una nuova AOI: creare una sottocartella in AOIs/ e
# inserirvi un file vettoriale in uno dei formati supportati.
# Il nome della sottocartella diventa l'identificatore AOI usato nei log,
# nei path di output e nel flag --aoi.
#
# Per cambiare la cartella di default: usare --aois-root <path> a riga di
# comando, oppure modificare il default in pipeline.py:
#   p.add_argument("--aois-root", default="AOIs", ...)
#
# Il file vettoriale può contenere qualsiasi geometria (Polygon,
# MultiPolygon); viene letta solo la prima feature. Il CRS può essere
# qualsiasi proiezione leggibile da Fiona — la pipeline riproietta
# internamente in WGS84 per le query STAC.
# ---------------------------------------------------------------------------

def load_aoi(path):
    """Carica un'AOI da un file vettoriale e restituisce geometria + bbox.

    Supporta qualsiasi formato leggibile da Fiona (Shapefile, GeoJSON,
    GeoPackage, KML, ecc.). Per i formati multi-layer (es. GeoPackage)
    viene aperto il primo layer.

    Parameters
    ----------
    path : str o Path
        Percorso al file vettoriale dell'AOI.

    Returns
    -------
    dict con chiavi:
        name : str -- nome AOI (dal campo 'Name' o dal nome cartella)
        geometry : shapely.geometry -- geometria nel CRS nativo del file
        bbox : tuple -- (xmin, ymin, xmax, ymax) nel CRS nativo
        crs : str -- CRS del file (es. EPSG:4326 o EPSG:32634)
    """
    path = Path(path)
    with fiona.open(path) as src:
        crs = str(src.crs)
        feat = next(iter(src))
        geom = shape(feat["geometry"])
        name = feat["properties"].get("Name") or path.parent.name

    return {
        "name": name,
        "geometry": geom,
        "bbox": geom.bounds,  # (minx, miny, maxx, maxy)
        "crs": crs,
    }


# ---------------------------------------------------------------------------
# Gestione CRS / riproiezione
# ---------------------------------------------------------------------------

def reproject_bbox(bbox, src_crs, dst_crs):
    """Riproietta un bounding box da src_crs a dst_crs.

    Parameters
    ----------
    bbox : tuple
        (xmin, ymin, xmax, ymax) nel CRS sorgente.
    src_crs : str
        CRS sorgente.
    dst_crs : str
        CRS destinazione.

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) nel CRS destinazione.
    """
    if _crs_equal(src_crs, dst_crs):
        return bbox
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def get_aoi_bbox_wgs84(aoi):
    """Restituisce il bbox dell'AOI in WGS 84, per query STAC.

    Parameters
    ----------
    aoi : dict
        AOI dict (da load_aoi).

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    return reproject_bbox(aoi["bbox"], aoi["crs"], "EPSG:4326")


def get_aoi_bbox_raster(aoi, raster_crs):
    """Restituisce il bbox dell'AOI nel CRS del raster, per ritaglio.

    Parameters
    ----------
    aoi : dict
        AOI dict (da load_aoi).
    raster_crs : str
        CRS del raster Sentinel-2 (es. "EPSG:32634").

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) nel CRS del raster.
    """
    return reproject_bbox(aoi["bbox"], aoi["crs"], raster_crs)


def _crs_equal(crs_a, crs_b):
    """Confronto normalizzato tra stringhe CRS (case-insensitive, senza spazi)."""
    return crs_a.upper().replace(" ", "") == crs_b.upper().replace(" ", "")


# ---------------------------------------------------------------------------
# STAC: query catalogo e conversione metadati
# ---------------------------------------------------------------------------

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


def query_stac(bbox_wgs84, date_from, date_to, stac_url, collection,
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
        return list_scenes(scene_dir)
    elif stac_client is not None:
        return _get_scenes_stac(aoi, stac_client, date_from)
    else:
        raise ValueError("Specificare scene_dir (locale) o stac_client (remoto)")


# ---------------------------------------------------------------------------
# Salvataggio output scena
# ---------------------------------------------------------------------------

def save_scene_outputs(result, scene, aoi, output_dir, scene_dir=None,
                       event_ids=None, tile_id=None, scene_ts=None,
                       aoi_crop=None, aoi_mask=None):
    """Salva i prodotti preliminari di una scena: dNBR, severity, GeoPackage e
    RGB opzionale (suffisso ``_prelim``; perimetrazione finale da ``end_event``).

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
        Event ID toccato dalla scena; usato come nome cartella e layer GPKG.
    tile_id : str, optional
        Tile MGRS (es. T35SMC), salvato come proprieta' nel GeoPackage.
    scene_ts : str, optional
        Timestamp della scena formattato (da events._format_scene_ts); usato
        come prefisso file e nome layer quando event_ids e' fornito.
    aoi_crop : tuple, optional
        Finestra di ritaglio (row_off, col_off, nrows, ncols) sul grid tile.
        Se fornita, i raster di output vengono ritagliati al bounding box AOI.
    aoi_mask : np.ndarray (bool), optional
        Maschera pixel interni all'AOI. I pixel esterni vengono azzerati
        prima del salvataggio.
    """
    from . import postprocess  # local import: postprocess importa data_io

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
    write_geotiff(
        _dnbr, profile,
        out_dir / f"{raster_pfx}_dNBR.tif",
        dtype="float32",
    )

    # Severity raster preliminare
    if _severity is not None:
        write_geotiff(
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
        write_geopackage(
            features,
            gpkg_path,
            crs=crs_str,
            layer_name=layer_name_prelim,
        )

    # RGB composito Highlight Optimized Natural Color (opzionale)
    if config.PRODUCE_RGB:
        postprocess.save_rgb_composite(scene, aoi, scene_dir, out_dir)

    return out_dir
