"""
preprocess.py -- Correzione radiometrica (DN -> riflettanza) e cloud masking (SCL).

Usato da:
  - baseline.py        (costruzione baseline pre-campagna)
  - pipeline.py        (monitoraggio continuo operativo)
  - output RGB         (composito true-color per visualizzazione)

Sentinel-2 Collection 1 L2A (Processing Baseline >= 05.00):
    riflettanza = DN * 0.0001 - 0.1

La banda SCL (Scene Classification Layer) e' una mappa di classificazione
(0-11, uint8) e non richiede correzione radiometrica.

Bande NBR (B8A, B12, SCL): tutte a 20 m, nessun resampling necessario.
Bande RGB (B4, B3, B2): 10 m nativi; il composito e' prodotto a 10 m,
  la maschera SCL (20 m) viene ricampionata a 10 m con nearest-neighbour.

Ref: progetto tecnico, fasi 4 (Radiometric Correction) e 6 (Cloud Masking).
"""

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# Fallback radiometrici Sentinel-2 Collection 1 L2A
# Usati SOLO se scale/offset non sono presenti nel JSON STAC della scena.
# ---------------------------------------------------------------------------
_DEFAULT_SCALE = 0.0001
_DEFAULT_OFFSET = -0.1


# ---------------------------------------------------------------------------
# Funzioni pubbliche
# ---------------------------------------------------------------------------

def get_band_calibration(meta, asset_key):
    """Estrae scale e offset dal JSON STAC per una banda.

    Parameters
    ----------
    meta : dict
        Metadati della scena (da data_io.load_metadata).
    asset_key : str
        Chiave STAC dell'asset (es. "nir08", "swir22", "red").

    Returns
    -------
    scale : float
    offset : float
    """
    asset = meta.get("assets", {}).get(asset_key, {})
    scale = asset.get("scale", _DEFAULT_SCALE)
    offset = asset.get("offset", _DEFAULT_OFFSET)
    return scale, offset


def dn_to_reflectance(dn, scale=_DEFAULT_SCALE, offset=_DEFAULT_OFFSET):
    """Converte un array di DN in riflettanza di superficie.

    Formula: rho = DN * scale + offset
    I valori di scale/offset vanno letti dal JSON STAC della scena
    tramite get_band_calibration().

    Parameters
    ----------
    dn : np.ndarray
        Array 2D di Digital Number (uint16).
    scale : float
        Fattore moltiplicativo (default: 0.0001).
    offset : float
        Offset additivo (default: -0.1).

    Returns
    -------
    np.ndarray
        Riflettanza di superficie (float32).
    """
    return dn.astype("float32") * scale + offset


def scl_mask(scl, mask_classes=None):
    """Genera una maschera booleana dai valori SCL.

    Parameters
    ----------
    scl : np.ndarray
        Array 2D Scene Classification Layer (uint8, valori 0-11).
    mask_classes : list[int], optional
        Classi SCL da mascherare (pixel non validi).
        Se None, usa config.SCL_MASK_CLASSES.

    Returns
    -------
    np.ndarray (bool)
        True = pixel valido (non mascherato), False = pixel da scartare.
    """
    if mask_classes is None:
        mask_classes = config.SCL_MASK_CLASSES
    return ~np.isin(scl, mask_classes)


def prepare_bands(nir_raw, swir_raw, scl, meta=None):
    """Preprocessing completo: DN -> riflettanza + maschera SCL + nodata.

    Passi:
    1. Maschera pixel nodata (DN == 0) su NIR e SWIR
    2. Maschera pixel non validi da SCL (nuvole, ombre, acqua, neve)
    3. Conversione DN -> riflettanza (scale/offset letti dal JSON STAC)

    Parameters
    ----------
    nir_raw : np.ndarray
        Banda B8A (NIR, 20 m) come DN uint16.
    swir_raw : np.ndarray
        Banda B12 (SWIR, 20 m) come DN uint16.
    scl : np.ndarray
        Banda SCL (Scene Classification Layer, 20 m) uint8.
    meta : dict, optional
        Metadati STAC della scena. Se fornito, scale/offset vengono
        letti da meta["assets"][<band>]. Se None, usa i default C1.

    Returns
    -------
    nir : np.ndarray (float32)
        Riflettanza NIR.
    swir : np.ndarray (float32)
        Riflettanza SWIR.
    valid_mask : np.ndarray (bool)
        True = pixel valido per il calcolo NBR.
    """
    nodata = config.NODATA_DN
    mask_nodata = (nir_raw == nodata) | (swir_raw == nodata)
    mask_scl = scl_mask(scl)
    valid_mask = mask_scl & ~mask_nodata

    nir_s, nir_o = get_band_calibration(meta, "nir08") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    swir_s, swir_o = get_band_calibration(meta, "swir22") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)

    nir = dn_to_reflectance(nir_raw, nir_s, nir_o)
    swir = dn_to_reflectance(swir_raw, swir_s, swir_o)

    return nir, swir, valid_mask


def prepare_rgb(red_raw, green_raw, blue_raw, scl=None, meta=None):
    """Preprocessing bande RGB: DN -> riflettanza + maschera opzionale.

    Le bande RGB (B4, B3, B2) sono a 10 m nativi; la SCL e' a 20 m.
    Se scl e' fornita e ha dimensioni diverse (20 m), viene ricampionata
    a 10 m con nearest-neighbour per applicare la maschera.

    Parameters
    ----------
    red_raw : np.ndarray
        Banda B4 (Red, 10 m) come DN uint16.
    green_raw : np.ndarray
        Banda B3 (Green, 10 m) come DN uint16.
    blue_raw : np.ndarray
        Banda B2 (Blue, 10 m) come DN uint16.
    scl : np.ndarray, optional
        Banda SCL (20 m) uint8. Se fornita, i pixel non validi vengono
        mascherati (impostati a 0 nel composito).
    meta : dict, optional
        Metadati STAC della scena (per scale/offset per banda).

    Returns
    -------
    rgb : np.ndarray (float32)
        Array (3, H, W) con riflettanza [R, G, B], clippata a [0, 1].
    valid_mask : np.ndarray (bool)
        Maschera pixel validi alla risoluzione RGB (H, W).
        Se scl e' None, tutti i pixel non-nodata sono validi.
    """
    nodata = config.NODATA_DN
    # AND: un pixel è nodata solo se TUTTE le bande sono zero (bordi tile).
    # Con OR, bande singole a DN basso (es. B2 su vegetazione scura) bucano
    # pixel validi in tutto il raster.
    mask_nodata = (red_raw == nodata) & (green_raw == nodata) & (blue_raw == nodata)

    r_s, r_o = get_band_calibration(meta, "red") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    g_s, g_o = get_band_calibration(meta, "green") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)
    b_s, b_o = get_band_calibration(meta, "blue") if meta else (_DEFAULT_SCALE, _DEFAULT_OFFSET)

    red = dn_to_reflectance(red_raw, r_s, r_o)
    green = dn_to_reflectance(green_raw, g_s, g_o)
    blue = dn_to_reflectance(blue_raw, b_s, b_o)

    # Clip a [0, 1] per visualizzazione
    rgb = np.clip(np.array([red, green, blue]), 0.0, 1.0)

    valid_mask = ~mask_nodata

    if scl is not None:
        mask_scl = scl_mask(scl)
        # Se SCL e' a 20 m e RGB a 10 m, ricampiona nearest-neighbour
        if mask_scl.shape != valid_mask.shape:
            mask_scl = _upsample_nearest(mask_scl, valid_mask.shape)
        valid_mask = valid_mask & mask_scl

    return rgb, valid_mask


def _upsample_nearest(arr, target_shape):
    """Ricampiona un array 2D a target_shape con nearest-neighbour.

    Parameters
    ----------
    arr : np.ndarray
        Array 2D sorgente (es. 20 m).
    target_shape : tuple
        (height, width) target (es. 10 m).

    Returns
    -------
    np.ndarray
        Array ricampionato.
    """
    row_idx = np.linspace(0, arr.shape[0] - 1, target_shape[0]).round().astype(int)
    col_idx = np.linspace(0, arr.shape[1] - 1, target_shape[1]).round().astype(int)
    return arr[np.ix_(row_idx, col_idx)]
