"""
config.py -- Parametri e soglie dell'algoritmo FireDetection.
Valori iniziali da letteratura; calibrazione su dati EMSR Copernicus & EFFIS.

Organizzazione per area di azione (ordine di esecuzione nella pipeline):
  1. INPUT              -- bande richieste, nodata, versione prodotto
  2. MASCHERATURA SCL   -- pixel non validi (nuvole, acqua, ombre), filtri cloud cover
  3. RILEVAMENTO PIXEL  -- indici spettrali (dNBR/RBR), soglie NIR, classificazione severity
  4. ANALISI CLUSTER    -- morfologia, fusione, area minima, solidity (find_clusters)
  5. APERTURA EVENTO    -- compactness scena, distanza max fusione iniziale, area min evento
  6. LIFECYCLE EVENTO   -- finestra conferma, timeout, soglie conferma pixel
  7. FOOTPRINT OUTPUT   -- geometria finale, filtri post-processing vettoriali
  8. BASELINE           -- composite pre-campagna
  9. SISTEMA            -- flag di controllo, output RGB
"""

# ===========================================================================
# 1. INPUT -- bande Sentinel-2, nodata, versione prodotto
# ===========================================================================

BANDS = {
    # Obbligatorie -- calcolo NBR + cloud masking (20 m)
    "nir08":  "B8A",
    "swir22": "B12",
    "scl":    "SCL",
    # Opzionali -- composito RGB true color (10 m)
    "red":    "B4",
    "green":  "B3",
    "blue":   "B2",
}

NODATA_DN = 0        # valore nodata nei DN originali (Sentinel-2 L2A)
NODATA_OUTPUT = -9999  # valore nodata nei raster di output (float32)

# Processing baseline minima per Collection 1
MIN_PROCESSING_BASELINE = "05.00"

# ===========================================================================
# 2. MASCHERATURA SCL / CLOUD FILTERING
#    Applicata all'inizio di ogni scena, prima di qualsiasi calcolo.
# ===========================================================================

# Classi SCL da mascherare (pixel non validi per l'analisi)
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

# Classi SCL considerate valide per l'analisi:
# 2 = DARK_AREA_PIXELS (inclusa: le aree bruciate sono scure e la SCL
#     le misclassifica; le ombre topografiche sono stabili tra baseline
#     e corrente, quindi il dNBR resta basso, poco probabili FP)
# 4 = VEGETATION, 5 = NOT_VEGETATED, 7 = UNCLASSIFIED

MAX_CLOUD_COVER_PCT = 70.0  # pre-screening da metadati: scarta scene con copertura nuvolosa > 70%

SCENE_VALID_SCL_PCT = 70.0  # % minima pixel SCL validi sull'AOI per considerare la scena utile.
                             # Sotto soglia: la scena non aggiorna gli eventi attivi e non
                             # contribuisce al conteggio delle osservazioni.

# ===========================================================================
# 3. RILEVAMENTO PIXEL BRUCIATI (spettrale)
#    Applicato pixel per pixel dopo la mascheratura SCL.
# ===========================================================================

# Indice da usare: "dNBR" (default) oppure "RBR" (aree eterogenee)
# Nota: in modalita' RBR la classificazione severity (SEVERITY_CLASSES) usa comunque
# le stesse soglie calibrate su dNBR (Key & Benson 2006). Su vegetazione sparsa
# (NBR_pre ≈ 0) RBR ≈ dNBR (denominatore ≈ 1); su vegetazione densa (NBR_pre ≈ 0.5-0.7)
# RBR < dNBR (denominatore > 1): la severity tende a essere sottostimata rispetto a dNBR.
INDEX_MODE = "dNBR"

# Soglie di rilevamento -- usate in base a INDEX_MODE
# dNBR = NBR_pre - NBR_post; RBR = dNBR / (NBR_pre + 1.001)
# Ref: Key & Benson (2006), Parks et al. (2014), HAZA02 Copernicus
DNBR_THRESHOLD = 0.10  # limite inferiore Low Severity (Key & Benson 2006: 0.099)
RBR_THRESHOLD  = 0.10  # coerente con DNBR_THRESHOLD; usato se INDEX_MODE='RBR'

# Filtro NIR: pixel bruciati sono scuri (carbone, 0.05-0.15); esclude nubi residue (NIR > 0.30).
NIR_MAX_BURNT = 0.25

# Filtro SWIR2: esclude acqua (SWIR2 vicino allo zero) non coperta dalla SCL class 6.
SWIR2_MIN_BURNT = 0.03

# Classificazione severita USGS (7 classi) -- soglie su dNBR
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
# 4. ANALISI CLUSTER (find_clusters -- applicato a ogni scena)
#    Raggruppa i pixel bruciati in cluster geometrici distinti, applicando
#    in sequenza: pulizia morfologica, labeling, fusione cluster vicini,
#    filtro per area minima e filtro per forma (solidity).
# ===========================================================================

CLUSTER_OPENING_RADIUS_PX = 2    # raggio (px) dell'opening morfologico sulla burnt_mask prima della
                                 # label analysis: rimuove pixel isolati e rompe bridge sottili tra
                                 # aree distinte. 0 = disabilitato. A 20 m/px: 2 px = 40 m.

MIN_CLUSTER_SEPARATION_PX = 100  # distanza massima (px) tra centroidi per fondere cluster
                                 # dello stesso incendio frammentati da nuvole.
                                 # None / 0 = disabilitato. A 20 m/px: 100 px = 2 km.

MIN_ALERT_AREA_HA = 5.0          # area minima (ha) per restituire un cluster da find_clusters.
                                 # Soglia di notifica, non di rilevamento (l'algoritmo rileva tutto).

MIN_CLUSTER_SOLIDITY = 0.40      # rapporto minimo (area_cluster / area_convex_hull) per ciascun cluster.
                                 # Filtra strisce costiere e forme allungate (solidity bassa).
                                 # Applicato per ogni cluster individuale dopo il filtro area.
                                 # None = disabilitato.

# ===========================================================================
# 5. APERTURA EVENTO (pipeline -- scena iniziale, nessun evento attivo)
#    Controlli aggiuntivi eseguiti in pipeline.py DOPO find_clusters,
#    prima di decidere se aprire uno o piu' nuovi eventi.
# ===========================================================================

MIN_CLUSTER_COMPACTNESS = 0.40   # rapporto minimo (largest_cluster_ha / total_burnt_ha) a livello scena.
                                 # Valori bassi indicano pixel sparsi (ombre, nuvole, rumore) -> scena ignorata.
                                 # None = disabilitato.

MAX_INITIAL_MERGE_DISTANCE_KM = 15.0  # distanza massima (km) tra centroidi di cluster della stessa scena
                                       # iniziale per fonderli in un unico evento.
                                       # Cluster piu' lontani aprono eventi separati.
                                       # Usato solo alla prima apertura (nessun evento attivo sulla tile).

EVENT_MIN_CLUSTER_HA = 5.0       # area minima (ha) perche' un cluster apra un nuovo evento.

# ===========================================================================
# 6. LIFECYCLE EVENTO (tracking, conferma, chiusura)
#    Un evento si chiude dopo EVENT_WINDOW_SCENES scene valide, oppure dopo
#    EVENT_TIMEOUT_DAYS giorni se gia' sufficientemente osservato, oppure
#    al raggiungimento del hard cap EVENT_MAX_TIMEOUT_DAYS.
#    Un pixel e' confermato bruciato se rilevato in almeno EVENT_MIN_DETECTIONS
#    scene su tutte le osservazioni valide nella finestra.
# ===========================================================================

EVENT_WINDOW_SCENES    = 8       # n. scene valide per chiusura automatica
EVENT_TIMEOUT_DAYS     = 30      # giorni dall'alert per chiusura (soft: solo se n_valid >= EVENT_MIN_DETECTIONS)
EVENT_MAX_TIMEOUT_DAYS = 60      # hard cap assoluto: chiude comunque dopo N giorni
EVENT_MIN_DETECTIONS   = 4       # n. minimo di rilevamenti (burnt_count) per confermare un pixel

EVENT_BASELINE_BUFFER_PX = 0     # margine di sicurezza (px) attorno al footprint di un evento ancora
                                 # aperto quando un evento concomitante si chiude sulla stessa tile.
                                 # Evita che l'aggiornamento post-chiusura sovrascriva il baseline
                                 # pre-fire dell'evento ancora in corso nei pixel di bordo.
                                 # 0 = nessun margine. A 20 m/px: 5 px = 100 m.

# ===========================================================================
# 7. FOOTPRINT OUTPUT (geometria finale + post-processing vettoriale)
#    Applicato alla chiusura dell'evento per produrre il perimetro finale.
# ===========================================================================

OUTPUT_NOISE_FILTER_MIN_AREA_HA = 5.0  # min cluster area (ha) kept in severity_final before vectorisation;
                                       # zeroes scattered noise pixels; 0 = disabled
FOOTPRINT_MIN_CLUSTER_HA = 5.0   # cluster < soglia esclusi dalla footprint (rimuove falsi positivi sparsi)
FOOTPRINT_CLOSING_M  = 80        # raggio del closing morfologico per il perimetro footprint (metri);
                                 # un raggio maggiore produce curve piu' tondeggianti
FOOTPRINT_EXPAND_M   = 30        # espansione netta del perimetro footprint dopo il closing (metri)
FOOTPRINT_SIMPLIFY_M =  1        # tolleranza Douglas-Peucker per la semplificazione del perimetro (metri)

MIN_PATCH_PIXELS = 25            # rimuovi patch isolate < 25 pixel (~10 ha)
HOLE_FILL_PIXELS = 500           # riempi buchi < 500 pixel (~20 ha) nei poligoni finali

# ===========================================================================
# 8. BASELINE -- median composite pre-campagna
# ===========================================================================

CAMPAIGN_START_DATE = "2024-06-01"  # None = campagna parte da oggi; "YYYY-MM-DD" = data esplicita
                                    # Nota: la baseline viene costruita una sola volta e salvata su disco
                                    # (<output>/<aoi>/<tile>/baseline_nbr.tif). Ai run successivi viene
                                    # ricaricata dal file esistente, indipendentemente da questo parametro.
                                    # Per forzare la ricostruzione: cancellare il file o usare FORCE_REPROCESS.
BASELINE_LOOKBACK_DAYS = 35         # finestra retrospettiva pre-campagna (giorni)
BASELINE_MIN_SCENES = 3             # minimo scene cloud-free per il composite
BASELINE_MAD_K = 6                  # filtro MAD: scarta pixel < mediana - k*MAD
BASELINE_MAD_FLOOR = 0.005          # pavimento MAD: se MAD < eps, non filtrare

# ===========================================================================
# 9. SISTEMA -- flag di controllo e output opzionale
# ===========================================================================

# Se True, ignora lo stato JSON e riprocessa tutto da zero (baseline + scene).
# Usare dopo cambio parametri algoritmici.
#
# Per riprocessare da una data specifica (es. dopo scene saltate per errore di rete)
# senza ripartire da zero, modificare manualmente il campo "last_processed_dt" nel file
# <output>/<aoi>/<tile>/pipeline_state.json impostandolo alla data ISO 8601 precedente
# alla scena da recuperare. Il pipeline ripartira' da quella data nel workflow.
FORCE_REPROCESS = False

# Output RGB composito -- Highlight Optimized Natural Color (HONC)
# Se True, produce un composito RGB a 10 m per ogni scena con incendio rilevato.
# Formula L2A: cbrt(0.6 * riflettanza) per B4, B3, B2
# Ref: Marko Repse, Sentinel Hub Custom Scripts
#      https://sentinel-hub.github.io/custom-scripts/sentinel-2/highlight_optimized_natural_color/
PRODUCE_RGB = False
