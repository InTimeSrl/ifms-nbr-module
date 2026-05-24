"""
run.py -- Entry point operativo del sistema di monitoraggio incendi.

Scansiona tutte le AOI in AOIs/, calcola le date di campagna da
config.CAMPAIGN_START_DATE e lancia la pipeline completa.

Uso:
    python run.py                          # tutte le AOI
    python run.py --aoi EMSR743-EMSR744_CentralGreece  # singola AOI
    python run.py --aois-root /path/to/AOIs --output-root /path/to/output

Parametri chiave (src/config.py):
    CAMPAIGN_START_DATE = None            # campagna parte da oggi (solo baseline)
    CAMPAIGN_START_DATE = "2024-07-01"    # monitoraggio storico da quella data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import main

if __name__ == "__main__":
    main()
