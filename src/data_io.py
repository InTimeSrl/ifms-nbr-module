"""
data_io.py -- I/O scene Sentinel-2, AOI, output raster/vector.
Usa rasterio (GDAL) per raster, fiona per shapefile, JSON per metadati.
"""

import json
import os
from pathlib import Path

import fiona
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from . import config


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

def reproject_geometry(geometry, src_crs, dst_crs):
    """Riproietta una geometria Shapely da src_crs a dst_crs.

    Parameters
    ----------
    geometry : shapely.geometry
        Geometria sorgente.
    src_crs : str
        CRS sorgente (es. "EPSG:4326", "EPSG:32634").
    dst_crs : str
        CRS destinazione.

    Returns
    -------
    shapely.geometry
        Geometria riproiettata.
    """
    if _crs_equal(src_crs, dst_crs):
        return geometry
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


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
