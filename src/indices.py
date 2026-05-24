"""
indices.py -- Calcolo indici spettrali per il rilevamento aree bruciate.

Indici implementati:
  - NBR  = (NIR - SWIR) / (NIR + SWIR)
  - dNBR = NBR_pre - NBR_post
  - RBR  = dNBR / (NBR_pre + 1.001)

Usato da:
  - baseline.py   (compute_nbr per la costruzione del composite)
  - pipeline.py   (compute_dnbr / compute_rbr per il monitoraggio continuo)

La scelta tra dNBR e RBR e' configurata in config.INDEX_MODE.

Ref: Key & Benson 2006 (dNBR), Parks et al. 2014 (RBR).
"""

import numpy as np


def compute_nbr(nir, swir):
    """Normalized Burn Ratio: NBR = (NIR - SWIR) / (NIR + SWIR).

    Parameters
    ----------
    nir : np.ndarray (float32)
        Riflettanza banda B8A (da preprocess.prepare_bands).
    swir : np.ndarray (float32)
        Riflettanza banda B12 (da preprocess.prepare_bands).

    Returns
    -------
    np.ndarray (float32)
        Valori NBR, tipicamente in [-1, 1]. Divisione per zero -> 0.
    """
    denom = nir + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = np.where(np.abs(denom) > 1e-6, (nir - swir) / denom, 0.0)
    return np.clip(nbr, -1.0, 1.0).astype("float32")


def compute_dnbr(nbr_pre, nbr_post):
    """Differenced NBR: dNBR = NBR_pre - NBR_post.

    Valori positivi indicano perdita di vegetazione (potenziale incendio).

    Parameters
    ----------
    nbr_pre : np.ndarray (float32)
        NBR pre-evento (baseline o previous).
    nbr_post : np.ndarray (float32)
        NBR post-evento (scena corrente).

    Returns
    -------
    np.ndarray (float32)
    """
    return (nbr_pre - nbr_post).astype("float32")


def compute_rbr(nbr_pre, nbr_post):
    """Relativized Burn Ratio: RBR = dNBR / (NBR_pre + 1.001).

    Normalizza il dNBR rispetto al valore pre-evento, riducendo la
    dipendenza dalla copertura vegetale iniziale. Utile in aree
    eterogenee (mix vegetazione/suolo nudo).

    L'offset 1.001 evita la divisione per zero (NBR_pre = -1 teorico).

    Parameters
    ----------
    nbr_pre : np.ndarray (float32)
        NBR pre-evento.
    nbr_post : np.ndarray (float32)
        NBR post-evento.

    Returns
    -------
    np.ndarray (float32)

    Ref: Parks, Dillon, Miller (2014), Int. J. Wildland Fire.
    """
    dnbr = nbr_pre - nbr_post
    return (dnbr / (nbr_pre + 1.001)).astype("float32")
