"""
state.py -- Tracciamento stato pipeline per singola AOI/tile.

Persiste un file JSON in <data_dir>/pipeline_state.json con:
  - baseline          : flag di costruzione + metadati (n. scene, copertura, ...)
  - last_processed_dt : ISO 8601 datetime dell'ultima scena processata
                        (watermark per la query STAC: si recuperano solo scene
                        con datetime > watermark, senza tracciare ogni scene_id)

Il file viene creato al primo run e aggiornato incrementalmente a ogni
esecuzione; in modalita' continua il sistema parte sempre dall'ultima scena
vista e processa solo le nuove.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"

# ---------------------------------------------------------------------------
# I/O base
# ---------------------------------------------------------------------------

def state_path(data_dir):
    """Restituisce il Path del file JSON di stato per un data_dir."""
    return Path(data_dir) / STATE_FILENAME


def load_state(data_dir):
    """Carica lo stato dal JSON.

    Se config.FORCE_REPROCESS e' True restituisce sempre uno stato vuoto,
    forzando il ricalcolo completo (baseline + tutte le scene).
    Se il file non esiste restituisce uno stato vuoto con struttura valida.
    Non solleva eccezioni: un JSON corrotto viene loggato e ignorato
    (restituisce stato vuoto, costringendo il pipeline a ripartire da zero).
    """
    if config.FORCE_REPROCESS:
        logger.info("FORCE_REPROCESS=True: stato JSON ignorato, riprocessing completo")
        return {"baseline": {"built": False}}
    p = state_path(data_dir)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                state = json.load(f)
            # Assicura le chiavi obbligatorie anche su file vecchi
            state.setdefault("baseline", {"built": False})
            # Migrazione da vecchio formato: processed_scenes -> last_processed_dt
            if "processed_scenes" in state and "last_processed_dt" not in state:
                scene_dates = [v.get("date", "") for v in state["processed_scenes"].values()
                               if v.get("date")]
                if scene_dates:
                    state["last_processed_dt"] = max(scene_dates)
                    logger.info("Migrazione stato: watermark derivato da processed_scenes: %s",
                                state["last_processed_dt"])
                del state["processed_scenes"]
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Stato JSON non leggibile (%s), ripartenza da zero: %s", p, exc)
    return {"baseline": {"built": False}}


def save_state(state, data_dir):
    """Salva lo stato su JSON."""
    p = state_path(data_dir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def is_baseline_built(state, required_paths):
    """True se la baseline risulta costruita E tutti i file raster esistono.

    Parameters
    ----------
    state : dict
        Stato caricato con load_state().
    required_paths : dict
        Dict {nome: Path} restituito da baseline.nbr_paths().
        Controlla almeno 'baseline', 'previous'.
    """
    if not state.get("baseline", {}).get("built", False):
        return False
    # Verifica fisicamente i file critici della baseline (non previous_nbr:
    # e' lo stato corrente, non parte della baseline, viene ricostruito
    # copiando baseline_nbr.tif se mancante)
    for key in ("baseline",):
        if key in required_paths and not Path(required_paths[key]).exists():
            logger.warning("File baseline mancante: %s -- ricostruzione necessaria",
                           required_paths[key])
            return False
    return True


def mark_baseline_built(state, n_scenes, coverage_pct, scene_ids):
    """Registra la costruzione della baseline nello stato (in-place).

    Parameters
    ----------
    n_scenes : int
    coverage_pct : float   Copertura 0-100.
    scene_ids : list[str]  IDs delle scene usate.
    """
    state["baseline"] = {
        "built": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_scenes": n_scenes,
        "coverage_pct": round(coverage_pct, 1),
        "scene_ids": list(scene_ids),
    }


# ---------------------------------------------------------------------------
# Watermark monitoraggio
# ---------------------------------------------------------------------------

def get_watermark(state):
    """ISO 8601 datetime dell'ultima scena processata, o None se prima esecuzione.

    Usato come limite inferiore (esclusivo) per la query STAC: si recuperano
    solo scene con datetime > watermark.
    """
    return state.get("last_processed_dt")


def update_watermark(state, scene_datetime):
    """Aggiorna il watermark se scene_datetime e' piu' recente (in-place).

    Parameters
    ----------
    scene_datetime : str
        ISO 8601 datetime della scena appena processata
        (es. "2025-08-14T09:09:52.139000Z").
    """
    current = state.get("last_processed_dt")
    if scene_datetime and (current is None or scene_datetime > current):
        state["last_processed_dt"] = scene_datetime
