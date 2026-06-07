"""
run.py -- Operational entry point for the wildfire monitoring system.

Scans all AOIs in AOIs/, computes campaign dates from
config.CAMPAIGN_START_DATE and runs the full pipeline.

Usage:
    python run.py                          # all AOIs
    python run.py --aoi EMSR743-EMSR744_CentralGreece  # single AOI
    python run.py --aoi Chios Kos          # multiple AOIs
    python run.py --aois-root /path/to/AOIs --output-root /path/to/output

Key parameters (src_v8_ENG_version_standby/config.py):
    CAMPAIGN_START_DATE = None            # campaign starts today (baseline only)
    CAMPAIGN_START_DATE = "2024-07-01"    # historical monitoring from that date
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src_v8_ENG_version_standby.pipeline import main

if __name__ == "__main__":
    main()
