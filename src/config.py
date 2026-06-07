"""
config.py -- Parameters and thresholds for the FireDetection algorithm.
Initial values from literature; calibrated on EMSR Copernicus & EFFIS data.

Sections follow the pipeline execution order:
  1. INPUT            -- required bands, nodata, product version
  2. SCL MASKING      -- invalid pixels (clouds, water, shadows), cloud cover filters
  3. PIXEL DETECTION  -- spectral indices (dNBR/RBR), NIR thresholds, severity classification
  4. CLUSTER ANALYSIS -- morphology, merging, minimum area, solidity (find_clusters)
  5. EVENT OPENING    -- scene compactness, initial merge distance, minimum event area
  6. EVENT LIFECYCLE  -- confirmation window, timeout, pixel confirmation thresholds
  7. FOOTPRINT OUTPUT -- final geometry, vector post-processing filters
  8. BASELINE         -- pre-campaign median composite
  9. SYSTEM           -- control flags, optional RGB output
"""

# ===========================================================================
# 1. INPUT -- Sentinel-2 bands, nodata, product version
# ===========================================================================

BANDS = {
    # Required -- NBR calculation + cloud masking (20 m)
    "nir08":  "B8A",
    "swir22": "B12",
    "scl":    "SCL",
    # Optional -- true-colour RGB composite (10 m)
    "red":    "B4",
    "green":  "B3",
    "blue":   "B2",
}

NODATA_DN = 0          # nodata value in original DNs (Sentinel-2 L2A)
NODATA_OUTPUT = -9999  # nodata value in output rasters (float32)

# Minimum processing baseline required for Collection 1
MIN_PROCESSING_BASELINE = "05.00"

# ===========================================================================
# 2. SCL MASKING / CLOUD FILTERING
#    Applied at the start of each scene, before any calculation.
# ===========================================================================

# SCL classes to mask (invalid pixels for analysis)
SCL_MASK_CLASSES = [
    0,   # NO_DATA
    1,   # SATURATED_OR_DEFECTIVE
    3,   # CLOUD_SHADOWS
    6,   # WATER
    8,   # CLOUD_MEDIUM_PROBABILITY
    9,   # CLOUD_HIGH_PROBABILITY
    10,  # THIN_CIRRUS
    11,  # SNOW_ICE
]

# SCL classes considered valid for analysis:
# 2 = DARK_AREA_PIXELS (included: burnt areas are dark and SCL misclassifies them;
#     topographic shadows are stable between baseline and current scene,
#     so dNBR stays low and false positives are unlikely)
# 4 = VEGETATION, 5 = NOT_VEGETATED, 7 = UNCLASSIFIED

MAX_CLOUD_COVER_PCT = 70.0  # pre-screening from metadata: skip scenes with cloud cover > 70%

SCENE_VALID_SCL_PCT = 70.0  # minimum % of valid SCL pixels over the AOI to consider a scene usable.
                             # Below threshold: the scene does not update active events and does not
                             # count towards the observation window.

# ===========================================================================
# 3. BURNT PIXEL DETECTION (spectral)
#    Applied pixel by pixel after SCL masking.
# ===========================================================================

# Index to use: "dNBR" (default) or "RBR" (heterogeneous vegetation)
# Note: in RBR mode the severity classification (SEVERITY_CLASSES) still uses
# the thresholds calibrated on dNBR (Key & Benson 2006). On sparse vegetation
# (NBR_pre ~= 0) RBR ~= dNBR (denominator ~= 1); on dense vegetation (NBR_pre ~= 0.5-0.7)
# RBR < dNBR (denominator > 1): severity tends to be underestimated relative to dNBR.
INDEX_MODE = "dNBR"

# Detection thresholds -- used according to INDEX_MODE
# dNBR = NBR_pre - NBR_post;  RBR = dNBR / (NBR_pre + 1.001)
# Ref: Key & Benson (2006), Parks et al. (2014), HAZA02 Copernicus
DNBR_THRESHOLD = 0.10  # lower bound for Low Severity (Key & Benson 2006: 0.099)
RBR_THRESHOLD  = 0.10  # consistent with DNBR_THRESHOLD; used when INDEX_MODE='RBR'

# NIR filter: burnt pixels are dark (charcoal, 0.05-0.15); excludes residual clouds (NIR > 0.30).
NIR_MAX_BURNT = 0.25

# SWIR2 filter: excludes water (SWIR2 near zero) not covered by SCL class 6.
SWIR2_MIN_BURNT = 0.03

# USGS burn severity classification (7 classes) -- thresholds on dNBR
# Ref: USGS, Key and Benson (2006)
SEVERITY_CLASSES = {
    1: {"label": "Enhanced Regrowth, High",   "abbr": "ER-H",   "min": None,   "max": -0.251},
    2: {"label": "Enhanced Regrowth, Low",    "abbr": "ER-L",   "min": -0.251, "max": -0.101},
    3: {"label": "Unburned",                  "abbr": "Unb",    "min": -0.101, "max":  0.099},
    4: {"label": "Low Severity",              "abbr": "Low",    "min":  0.099, "max":  0.269},
    5: {"label": "Moderate-Low Severity",     "abbr": "Mod-L",  "min":  0.269, "max":  0.439},
    6: {"label": "Moderate-High Severity",    "abbr": "Mod-H",  "min":  0.439, "max":  0.659},
    7: {"label": "High Severity",             "abbr": "High",   "min":  0.659, "max":  None},
}

# ===========================================================================
# 4. CLUSTER ANALYSIS (find_clusters -- applied to each scene)
#    Groups burnt pixels into distinct geometric clusters by applying in sequence:
#    morphological cleaning, labeling, nearby-cluster merging,
#    minimum area filter, and shape filter (solidity).
# ===========================================================================

CLUSTER_OPENING_RADIUS_PX = 2    # radius (px) of morphological opening on burnt_mask before
                                 # label analysis: removes isolated pixels and breaks thin bridges
                                 # between distinct areas. 0 = disabled. At 20 m/px: 2 px = 40 m.

MIN_CLUSTER_SEPARATION_PX = 100  # maximum centroid distance (px) to merge clusters from the same
                                 # fire fragmented by clouds.
                                 # None / 0 = disabled. At 20 m/px: 100 px = 2 km.

MIN_ALERT_AREA_HA = 5.0          # minimum area (ha) to return a cluster from find_clusters.
                                 # This is a notification threshold, not a detection threshold
                                 # (the algorithm detects everything).

MIN_CLUSTER_SOLIDITY = 0.40      # minimum ratio (cluster_area / convex_hull_area) per cluster.
                                 # Filters coastal strips and elongated shapes (low solidity).
                                 # Applied to each individual cluster after the area filter.
                                 # None = disabled.

# ===========================================================================
# 5. EVENT OPENING (pipeline -- first scene, no active events)
#    Additional checks run in pipeline.py AFTER find_clusters,
#    before deciding whether to open one or more new events.
# ===========================================================================

MIN_CLUSTER_COMPACTNESS = 0.40   # minimum ratio (largest_cluster_ha / total_burnt_ha) at scene level.
                                 # Low values indicate scattered pixels (shadows, clouds, noise) -> scene skipped.
                                 # None = disabled.

MAX_INITIAL_MERGE_DISTANCE_KM = 15.0  # maximum centroid distance (km) between clusters in the same
                                       # opening scene to merge them into a single event.
                                       # More distant clusters open separate events.
                                       # Used only at first opening (no active events on the tile).

EVENT_MIN_CLUSTER_HA = 5.0       # minimum cluster area (ha) to open a new event.

# ===========================================================================
# 6. EVENT LIFECYCLE (tracking, confirmation, closure)
#    An event closes after EVENT_WINDOW_SCENES valid scenes, or after
#    EVENT_TIMEOUT_DAYS days if already sufficiently observed, or when
#    the hard cap EVENT_MAX_TIMEOUT_DAYS is reached.
#    A pixel is confirmed burnt if detected in at least EVENT_MIN_DETECTIONS
#    scenes out of all valid observations in the window.
# ===========================================================================

EVENT_WINDOW_SCENES    = 8       # number of valid scenes for automatic closure
EVENT_TIMEOUT_DAYS     = 30      # days from alert for closure (soft: only if n_valid >= EVENT_MIN_DETECTIONS)
EVENT_MAX_TIMEOUT_DAYS = 60      # absolute hard cap: closes regardless after N days
EVENT_MIN_DETECTIONS   = 4       # minimum number of detections (burnt_count) to confirm a pixel

EVENT_BASELINE_BUFFER_PX = 0     # safety margin (px) around the footprint of a still-open event
                                 # when a concurrent event closes on the same tile.
                                 # Prevents the post-closure baseline update from overwriting the
                                 # pre-fire baseline of the ongoing event in border pixels.
                                 # 0 = no margin. At 20 m/px: 5 px = 100 m.

# ===========================================================================
# 7. FOOTPRINT OUTPUT (final geometry + vector post-processing)
#    Applied at event closure to produce the final perimeter.
# ===========================================================================

OUTPUT_NOISE_FILTER_MIN_AREA_HA = 5.0  # min cluster area (ha) kept in severity_final before vectorisation;
                                       # zeroes scattered noise pixels; 0 = disabled
FOOTPRINT_MIN_CLUSTER_HA = 5.0   # clusters below threshold excluded from footprint (removes scattered FPs)
FOOTPRINT_CLOSING_M  = 80        # morphological closing radius for the footprint perimeter (metres);
                                 # larger radius produces smoother curves
FOOTPRINT_EXPAND_M   = 30        # net outward expansion of the footprint perimeter after closing (metres)
FOOTPRINT_SIMPLIFY_M =  1        # Douglas-Peucker tolerance for perimeter simplification (metres)

MIN_PATCH_PIXELS = 25            # remove isolated patches < 25 pixels (< 1 ha at 20 m) from final raster
HOLE_FILL_PIXELS = 500           # fill holes < 500 pixels (~20 ha) in final polygons

# ===========================================================================
# 8. BASELINE -- pre-campaign median composite
# ===========================================================================

CAMPAIGN_START_DATE = "2026-06-01"  # None = campaign starts today; "YYYY-MM-DD" = explicit date.
                                    # Note: the baseline is built once and saved to disk
                                    # (<output>/<aoi>/<tile>/baseline_nbr.tif). On subsequent runs
                                    # it is reloaded from the existing file, regardless of this parameter.
                                    # To force a rebuild: delete the file or set FORCE_REPROCESS = True.
BASELINE_LOOKBACK_DAYS = 35         # pre-campaign look-back window (days)
BASELINE_MIN_SCENES = 3             # minimum cloud-free scenes for the composite
BASELINE_MAD_K = 6                  # MAD filter: discard pixels below median - k*MAD
BASELINE_MAD_FLOOR = 0.005          # MAD floor: if MAD < eps, skip filtering

# ===========================================================================
# 9. SYSTEM -- control flags and optional output
# ===========================================================================

# If True, ignores the JSON state and reprocesses everything from scratch (baseline + scenes).
# Use after changing algorithm parameters.
#
# To reprocess from a specific date (e.g. after scenes skipped due to a network error)
# without starting from scratch, manually edit the "last_processed_dt" field in
# <output>/<aoi>/<tile>/pipeline_state.json to the ISO 8601 date just before the scene
# to recover. The pipeline will resume from that date on the next run.
FORCE_REPROCESS = False

# RGB composite output -- Highlight Optimized Natural Color (HONC)
# If True, produces a 10 m RGB composite for each scene with detected fire.
# L2A formula: cbrt(0.6 * reflectance) for B4, B3, B2
# Ref: Marko Repse, Sentinel Hub Custom Scripts
#      https://sentinel-hub.github.io/custom-scripts/sentinel-2/highlight_optimized_natural_color/
PRODUCE_RGB = False
