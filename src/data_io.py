"""
data_io.py -- Sentinel-2 scene I/O, AOI loading, raster/vector output.
Uses rasterio (GDAL) for rasters, fiona for shapefiles, JSON for metadata.

Output folder:
    The root output path is set via the --output-root argument of run.py
    (default: "output/", relative to the project CWD).
    To change the default folder permanently, edit in src/pipeline.py:
        "--output-root", default="output",
    The directory structure created automatically is:
        <output-root>/<aoi_name>/data/      <- baseline_nbr, previous_nbr, pipeline_state.json
        <output-root>/<aoi_name>/products/  <- GeoPackage, severity_final.tif, ...

AOI parallelism (recommended approach):
    AOIs are fully independent on disk (separate data_dir and output_dir
    per AOI name). To process them in parallel, launch separate OS processes
    using the --aoi flag:

        # Linux: start N AOIs in parallel, each with its own log
        python run.py --aoi AOI_A > output_AOI_A.log 2>&1 &
        python run.py --aoi AOI_B > output_AOI_B.log 2>&1 &

    No locking needed: pipeline_state.json, events_index.json and NBR rasters
    are all per-tile/per-AOI.
    Note: network bandwidth to the STAC/COG provider may be a bottleneck;
    more than 3-4 simultaneous AOIs may saturate the connection or hit rate limits.
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
# GDAL Virtual File System configuration for remote COG reading
# Ref: https://gdal.org/en/stable/user/virtual_file_systems.html
#
# /vsicurl/  -> HTTP/HTTPS reading (AWS Element84, CDSE)
# /vsis3/    -> direct S3 reading (internal STAC, MinIO)
#
# rasterio.open() activates the appropriate VFS automatically based on
# the URL scheme (https:// -> vsicurl, s3:// -> vsis3).
# The options below are tuned for COG range requests.
# ---------------------------------------------------------------------------

def configure_gdal_vfs():
    """Set GDAL environment variables for remote COG reading."""
    gdal_opts = {
        # Disable remote directory scan on open
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # Restrict vsicurl to .tif/.tiff extensions only
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        # Merge adjacent HTTP range requests into a single request
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_CACHE_SIZE": "67108864",  # 64 MB cache
        # Anonymous access for public S3 buckets (Element84).
        # TODO (internal STAC / MinIO): replace with:
        #   "AWS_S3_ENDPOINT":     "<host:port>",    # e.g. "minio.internal:9000"
        #   "AWS_VIRTUAL_HOSTING": "FALSE",           # MinIO uses path-style URLs
        #   "AWS_HTTPS":           "NO",              # only if MinIO does not use TLS
        # If MinIO uses TLS with a self-signed certificate, keep AWS_HTTPS=YES and add:
        #   "GDAL_HTTP_UNSAFESSL": "YES"              # skip TLS certificate verification
        # Remove the line:
        #   "AWS_NO_SIGN_REQUEST" (for buckets requiring signed requests).
        # Credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) must NOT be hardcoded:
        # GDAL/vsis3 reads them automatically from the process environment.
        # The other four parameters (GDAL_DISABLE_READDIR_ON_OPEN, etc.) should remain unchanged.
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    for key, val in gdal_opts.items():
        os.environ.setdefault(key, val)


configure_gdal_vfs()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(json_path):
    """Load the JSON metadata file for a scene.

    Parameters
    ----------
    json_path : str or Path
        Path to the *_metadata.json file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_scenes(tile_dir):
    """List all scenes (JSON metadata files) in a tile folder.

    Parameters
    ----------
    tile_dir : str or Path
        Folder containing one or more scene files.

    Returns
    -------
    list[dict]
        List of metadata dicts, sorted by datetime.
    """
    tile_dir = Path(tile_dir)
    scenes = []
    for jp in tile_dir.glob("*_metadata.json"):
        scenes.append(load_metadata(jp))
    scenes.sort(key=lambda s: s["datetime"])
    return scenes


# ---------------------------------------------------------------------------
# Raster band reading
# ---------------------------------------------------------------------------

def _resolve_href(meta, asset_key):
    """Return the path or URL of a band asset to open with rasterio.

    Remote URLs (http/s3) are returned as-is.
    Local or relative paths are returned unchanged (rasterio handles them).
    """
    asset = meta["assets"][asset_key]
    return asset["href"]


def _local_path(meta, asset_key, scene_dir):
    """Build the local TIF path from metadata and scene folder.

    Used as a fallback when the href points to a remote URL but the file
    already exists on disk (e.g. downloaded test data).
    """
    item_id = meta["stac_item_id"]
    band_name = meta["assets"][asset_key]["band_name"]
    return Path(scene_dir) / f"{item_id}_{band_name}.tif"


def _resolve_source(meta, asset_key, scene_dir):
    """Return the effective path for a band asset.

    Resolution logic:
    1. If scene_dir is provided and the local file exists -> local path
    2. Otherwise -> href from the STAC JSON (HTTP URL or s3://)
       rasterio/GDAL opens it via /vsicurl/ or /vsis3/ automatically.
    """
    if scene_dir is not None:
        local = _local_path(meta, asset_key, scene_dir)
        if local.exists():
            return str(local)
    href = _resolve_href(meta, asset_key)
    return href


def get_scene_crs(meta, scene_dir=None):
    """Return the CRS of a scene by reading the header of the first raster band.

    Parameters
    ----------
    meta : dict
        Scene metadata.
    scene_dir : str or Path, optional
        Local TIF folder. If None, reads remotely (COG via VFS).

    Returns
    -------
    str
        CRS as a string (e.g. "EPSG:32634").
    """
    source = _resolve_source(meta, "nir08", scene_dir)
    with rasterio.open(source) as src:
        return str(src.crs)


def read_band(source, bbox=None):
    """Read a single raster band.

    Parameters
    ----------
    source : str
        Local path, HTTP URL, or S3 URL of the raster file.
    bbox : tuple, optional
        (xmin, ymin, xmax, ymax) for spatial clipping. Must be in the
        same CRS as the raster. If None, reads the entire tile.

    Returns
    -------
    data : np.ndarray
        2D array (height, width), original dtype (uint16 or uint8).
    profile : dict
        Rasterio profile (CRS, transform, dtype, etc.).
    """
    with rasterio.open(source) as src:
        if bbox is not None:
            window = from_bounds(*bbox, transform=src.transform)
            # Clamp to integer pixel grid and raster extent.
            # from_bounds may produce negative offsets or out-of-extent windows
            # when the bbox is larger than the tile.
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
    """Load all bands for a scene as a dictionary of arrays.

    Supports local reading (scene_dir) or remote reading (COG via GDAL VFS).
    If scene_dir is None, rasters are read directly from the STAC href
    via /vsicurl/ (HTTP) or /vsis3/ (S3).

    Parameters
    ----------
    meta : dict
        Scene metadata (from load_metadata).
    scene_dir : str or Path, optional
        Local TIF folder. If None, reads remotely.
    asset_keys : list[str], optional
        Asset keys to load (e.g. ["nir08", "swir22", "scl"]).
        If None, loads all bands defined in config.BANDS.
    bbox : tuple, optional
        Spatial clip (xmin, ymin, xmax, ymax) in the raster CRS.

    Returns
    -------
    bands : dict[str, np.ndarray]
        Dictionary {asset_key: 2D array}.
    profile : dict
        Rasterio profile (from the first band read).
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
# Output writing
# ---------------------------------------------------------------------------

def _write_tiff(out_path, profile, write_fn):
    """Internal primitive: write to a temp file then replace out_path (tmp->replace pattern).

    Parameters
    ----------
    out_path : Path
        Destination path (directory must exist).
    profile : dict
        Rasterio profile for the output file.
    write_fn : callable
        ``write_fn(dst)`` writes data into the open rasterio dataset.
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
    """Save a 2D array as a GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        2D array to save.
    profile : dict
        Rasterio profile (CRS, transform, etc.).
    out_path : str or Path
        Output path.
    dtype : str, optional
        Output data type (e.g. "float32", "uint8"). If None, uses data.dtype.
    nodata : number, optional
        Nodata value. If None, uses config.NODATA_OUTPUT.
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
    """Save a (3, H, W) array as an RGB GeoTIFF (uint8).

    Parameters
    ----------
    data : np.ndarray
        3D array (3, height, width) with R, G, B bands (uint8, 0-255).
    profile : dict
        Rasterio profile (CRS, transform, etc.).
    out_path : str or Path
        Output path.
    nodata : int, optional
        Nodata value. If None, no nodata is set on the GeoTIFF.
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
    """Save a list of features to a GeoPackage (.gpkg).

    Geometry type is fixed to MultiPolygon.

    Parameters
    ----------
    features : list[dict]
        Features in GeoJSON-like format (geometry + properties).
    out_path : str or Path
        Output .gpkg path (overwritten if it exists).
    crs : str, optional
        EPSG code (e.g. "EPSG:32635"). If None, no CRS is written.
    layer_name : str
        Layer name inside the GeoPackage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not features:
        return

    # Build schema from the properties of the first feature.
    # Supported fiona types: int, float, str, bool, date, datetime.
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

    # fiona in "w" mode on an existing GPKG appends instead of overwriting:
    # on repeated runs (e.g. restart after interruption) the layer would be duplicated.
    # Remove only the target layer; other layers in the file are left intact.
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
            # Promote Polygon -> MultiPolygon for schema consistency
            if geom.get("type") == "Polygon":
                geom = {"type": "MultiPolygon", "coordinates": [geom["coordinates"]]}
            # Coerce property values to declared type (None stays None).
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
# AOI loading
# ---------------------------------------------------------------------------
# AOIs are discovered automatically by pipeline.py (_scan_aois), which
# scans the AOIs/ folder (or the path passed via --aois-root).
# Expected structure:
#
#   AOIs/
#     <aoi_name>/
#       <vector_file>.<ext>    <- .geojson | .gpkg | .shp | .kml | .gml
#
# To add a new AOI: create a subfolder under AOIs/ and place a vector
# file in one of the supported formats inside it.
# The subfolder name becomes the AOI identifier used in logs, output
# paths, and the --aoi flag.
#
# To change the default AOI folder: use --aois-root <path> on the
# command line, or change the default in pipeline.py:
#   p.add_argument("--aois-root", default="AOIs", ...)
#
# The vector file can contain any geometry (Polygon, MultiPolygon);
# only the first feature is read. The CRS can be any projection
# readable by Fiona -- the pipeline reprojects internally to WGS84
# for STAC queries.
# ---------------------------------------------------------------------------

def load_aoi(path):
    """Load an AOI from a vector file and return geometry + bbox.

    Supports any format readable by Fiona (Shapefile, GeoJSON,
    GeoPackage, KML, etc.). For multi-layer formats (e.g. GeoPackage)
    the first layer is opened.

    Parameters
    ----------
    path : str or Path
        Path to the AOI vector file.

    Returns
    -------
    dict with keys:
        name : str -- AOI name (from 'Name' field or parent folder name)
        geometry : shapely.geometry -- geometry in the file's native CRS
        bbox : tuple -- (xmin, ymin, xmax, ymax) in the native CRS
        crs : str -- file CRS (e.g. EPSG:4326 or EPSG:32634)
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
# CRS handling / reprojection
# ---------------------------------------------------------------------------

def reproject_bbox(bbox, src_crs, dst_crs):
    """Reproject a bounding box from src_crs to dst_crs.

    Parameters
    ----------
    bbox : tuple
        (xmin, ymin, xmax, ymax) in the source CRS.
    src_crs : str
        Source CRS.
    dst_crs : str
        Destination CRS.

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) in the destination CRS.
    """
    if _crs_equal(src_crs, dst_crs):
        return bbox
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def get_aoi_bbox_wgs84(aoi):
    """Return the AOI bbox in WGS 84 for STAC queries.

    Parameters
    ----------
    aoi : dict
        AOI dict (from load_aoi).

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    return reproject_bbox(aoi["bbox"], aoi["crs"], "EPSG:4326")


def get_aoi_bbox_raster(aoi, raster_crs):
    """Return the AOI bbox in the raster CRS for spatial clipping.

    Parameters
    ----------
    aoi : dict
        AOI dict (from load_aoi).
    raster_crs : str
        Sentinel-2 raster CRS (e.g. "EPSG:32634").

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) in the raster CRS.
    """
    return reproject_bbox(aoi["bbox"], aoi["crs"], raster_crs)


def _crs_equal(crs_a, crs_b):
    """Normalised CRS string comparison (case-insensitive, no spaces)."""
    return crs_a.upper().replace(" ", "") == crs_b.upper().replace(" ", "")


# ---------------------------------------------------------------------------
# STAC: catalogue query and metadata conversion
# ---------------------------------------------------------------------------

def _stac_item_to_meta(item):
    """Convert a STAC Item into a pipeline-compatible metadata dict."""
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
    """Run a STAC query and return a list of scene metadata dicts."""
    catalog = Client.open(stac_url)
    date_range = "%s/%s" % (date_from, date_to)
    logger.info("STAC query: bbox=%s  date=%s", bbox_wgs84, date_range)
    results = catalog.search(
        collections=[collection],
        bbox=list(bbox_wgs84),
        datetime=date_range,
        max_items=max_items,
    )
    items = list(results.items())
    logger.info("STAC: found %d scenes", len(items))
    return [_stac_item_to_meta(it) for it in items]


def _get_scenes_stac(aoi, stac_client, date_from=None):
    """Retrieve scenes from a STAC catalogue.

    TODO: implement when the internal STAC server is available.
    """
    raise NotImplementedError("STAC query not yet implemented")


def get_scenes(aoi, scene_dir=None, stac_client=None, date_from=None):
    """Retrieve available scenes for an AOI (local or STAC).

    Either scene_dir or stac_client must be provided.
    """
    if scene_dir is not None:
        return list_scenes(scene_dir)
    elif stac_client is not None:
        return _get_scenes_stac(aoi, stac_client, date_from)
    else:
        raise ValueError("Provide either scene_dir (local) or stac_client (remote)")


# ---------------------------------------------------------------------------
# Scene output saving
# ---------------------------------------------------------------------------

def save_scene_outputs(result, scene, aoi, output_dir, scene_dir=None,
                       event_ids=None, tile_id=None, scene_ts=None,
                       aoi_crop=None, aoi_mask=None):
    """Save preliminary scene products: dNBR, severity, GeoPackage and
    optional RGB (suffix ``_prelim``; final perimeter produced by ``end_event``).

    Parameters
    ----------
    result : dict
        Output from process_scene (with fire_detected=True).
    scene : dict
        Scene metadata.
    aoi : dict
        AOI dict.
    output_dir : str
        Root output folder.
    scene_dir : str or Path, optional
        Local TIF folder (required to produce RGB).
    event_ids : str, optional
        Event ID touched by the scene; used as the folder name and GPKG layer name.
    tile_id : str, optional
        MGRS tile (e.g. T35SMC), saved as a property in the GeoPackage.
    scene_ts : str, optional
        Formatted scene timestamp (from events._format_scene_ts); used as
        file prefix and layer name when event_ids is provided.
    aoi_crop : tuple, optional
        Crop window (row_off, col_off, nrows, ncols) on the tile grid.
        If provided, output rasters are clipped to the AOI bounding box.
    aoi_mask : np.ndarray (bool), optional
        Mask of pixels inside the AOI. Pixels outside are zeroed before saving.
    """
    from . import postprocess  # local import: postprocess importa data_io

    scene_id = scene["stac_item_id"]
    if event_ids and scene_ts:
        # Shared sidecar folder for all scenes of the event.
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

    # Zero pixels outside the AOI (nodata): apply mask BEFORE crop
    if aoi_mask is not None:
        if _dnbr is not None and _dnbr.shape == aoi_mask.shape:
            _dnbr = np.where(aoi_mask, _dnbr, np.nan).astype(np.float32)
        if _severity is not None and _severity.shape == aoi_mask.shape:
            _severity = np.where(aoi_mask, _severity, 0).astype(_severity.dtype)

    # Clip to AOI bounding box on the tile grid
    if aoi_crop is not None:
        _r0, _c0, _nrows, _ncols = aoi_crop
        _win = _rwin.Window(_c0, _r0, _ncols, _nrows)
        _crop_tr = _rwin.transform(_win, profile["transform"])
        profile = {**profile, "height": _nrows, "width": _ncols, "transform": _crop_tr}
        if _dnbr is not None:
            _dnbr = _dnbr[_r0:_r0 + _nrows, _c0:_c0 + _ncols]
        if _severity is not None:
            _severity = _severity[_r0:_r0 + _nrows, _c0:_c0 + _ncols]

    # Preliminary dNBR raster
    write_geotiff(
        _dnbr, profile,
        out_dir / f"{raster_pfx}_dNBR.tif",
        dtype="float32",
    )

    # Preliminary severity raster
    if _severity is not None:
        write_geotiff(
            _severity, profile,
            out_dir / f"{raster_pfx}_severity_prelim.tif",
            dtype="uint8",
            nodata=0,
        )

    # Preliminary GeoPackage polygons (native CRS)
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

    # Optional RGB Highlight Optimized Natural Color composite
    if config.PRODUCE_RGB:
        postprocess.save_rgb_composite(scene, aoi, scene_dir, out_dir)

    return out_dir
