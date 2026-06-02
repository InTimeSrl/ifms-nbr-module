"""
config.py -- Parametri e soglie dell'algoritmo NBR.
Valori iniziali da letteratura; calibrazione su dati Chios/Arta vs EFFIS.

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
# Ref: Sentinel-2 L2A Scene Classification Layer
SCL_MASK_CLASSES = [
    0,   # NO_DATA
    1,   # SATURATED_OR_DEFECTIVE
    3,   # CLOUD_SHADOWS
    6,   # WATER (NBR basso simile a burnt -> falsi positivi)
    8,   # CLOUD_MEDIUM_PROBABILITY
    9,   # CLOUD_HIGH_PROBABILITY
    10,  # THIN_CIRRUS
    11,  # SNOW_ICE
]

# Classi SCL considerate valide per l'analisi:
# 2 = DARK_AREA_PIXELS (inclusa: le aree bruciate sono scure e la SCL
#     le misclassifica; le ombre topografiche sono stabili tra baseline
#     e corrente, quindi il dNBR resta basso -> nessun falso positivo)
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
# Ref: progetto tecnico par. 6.1 -- "start with dNBR, switch to RBR if needed"
INDEX_MODE = "dNBR"

# Soglie di rilevamento area bruciata -- usata in base a INDEX_MODE
#
# DNBR_THRESHOLD: soglia su dNBR (NBR_pre - NBR_post)
#   0.27 = confine Low / Moderate-Low Severity (Key & Benson 2006)
#   Conservativo: meno falsi positivi, possibili falsi negativi su phrygana
#   Ref: EFFIS/JRC fire-severity, UN-SPIDER NBR recommended practice
#
# RBR_THRESHOLD: soglia su RBR = dNBR / (NBR_pre + 1.001)
#   0.27 indicato dal tutorial HAZA02 (Copernicus) per Sentinel-2
#   Normalizza rispetto alla densita' vegetale pre-fuoco: piu' robusto
#   su aree eterogenee (mix foresta/macchia/phrygana)
#   Ref: Parks, Dillon & Miller (2014), Int. J. Wildland Fire; HAZA02
DNBR_THRESHOLD = 0.10  # abbassato al limite inferiore Low Severity (Key & Benson 2006: 0.099)
RBR_THRESHOLD  = 0.10  # coerente con DNBR_THRESHOLD; usato se INDEX_MODE='RBR'

# Soglia massima riflettanza NIR post-fire per conferma bruciatura.
# Un pixel bruciato e' scuro nel NIR (carbone assorbe) -> riflettanza < 0.20
# Nubi non mascherate dalla SCL hanno NIR alto (> 0.30) -> escluse
# Ref: carbone ha riflettanza NIR tipica 0.05-0.15
NIR_MAX_BURNT = 0.25

# Soglia minima riflettanza SWIR2 (B12) post-fire per conferma bruciatura.
# L'acqua assorbe quasi tutto il SWIR2 -> riflettanza tipica 0.005-0.02.
# Char/suolo bruciato -> SWIR2 tipicamente 0.03-0.10.
# Filtra corpi idrici che SCL class 6 non copre sempre (bacini stagionali,
# zone temporaneamente allagate, misclassificazione SCL).
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
#    Trasforma la burnt_mask grezza in un insieme di cluster geometrici.
#    Ordine di applicazione:
#      a) opening morfologico  (CLUSTER_OPENING_RADIUS_PX)
#      b) label connected components
#      c) fusione cluster vicini  (MIN_CLUSTER_SEPARATION_PX)
#      d) filtro area per cluster (MIN_ALERT_AREA_HA)
#      e) filtro forma per cluster (MIN_CLUSTER_SOLIDITY)
# ===========================================================================

CLUSTER_OPENING_RADIUS_PX = 2   # raggio (px) dell'opening morfologico applicato alla burnt_mask prima della
                                 # label analysis. L'opening (erosione + dilatazione) rimuove pixel isolati di
                                 # rumore e rompe i "bridge" sottili che connettono aree bruciate distinte,
                                 # permettendo di separarle in cluster distinti.
                                 # I pixel rimossi dall'erosione vengono recuperati via dilatazione prima di
                                 # assegnarli all'evento. 0 = disabilitato.
                                 # A 20 m/px: 2 px = 40 m di erosione minima.

MIN_CLUSTER_SEPARATION_PX = 100  # find_clusters (ogni scena): fonde cluster il cui centroide
                                 # dista < soglia (stesso incendio frammentato da nuvole).
                                 # None / 0 = ogni cluster e' indipendente.
                                 # A 20 m/px: 100 px = 2 km.

MIN_ALERT_AREA_HA = 5.0          # area minima (ha) perche' un cluster sia restituito da find_clusters.
                                 # Non e' soglia di rilevamento (l'algoritmo rileva tutto), ma soglia di notifica.
                                 # Ref: progetto tecnico par. 6.3.2 -- "configurable thresholds (e.g. 5 ha)"

MIN_CLUSTER_SOLIDITY = 0.40      # rapporto minimo (area_cluster / area_convex_hull) per ciascun cluster.
                                 # Filtra strisce costiere e forme allungate (solidity bassa).
                                 # Applicato per ogni cluster individuale dopo il filtro area.
                                 # None = disabilitato.

# ===========================================================================
# 5. APERTURA EVENTO (pipeline -- scena iniziale, nessun evento attivo)
#    Controlli aggiuntivi eseguiti in pipeline.py DOPO find_clusters,
#    prima di decidere se aprire uno o piu' nuovi eventi.
# ===========================================================================

MIN_CLUSTER_COMPACTNESS = 0.40   # rapporto minimo (largest_cluster_ha / total_burnt_ha) a livello di scena.
                                 # Un incendio reale e' compatto (>50%). Valori bassi indicano pixel
                                 # sparsi da ombre nuvole o rumore -> scena ignorata.
                                 # Gate rapido: evita di chiamare find_clusters su scene di puro rumore.
                                 # None = disabilitato.

MAX_INITIAL_MERGE_DISTANCE_KM = 15.0  # distanza massima (km) tra centroidi di cluster della stessa scena
                                       # iniziale per fonderli in un unico evento.
                                       # Cluster piu' lontani aprono eventi separati.
                                       # Usato solo alla prima apertura (nessun evento attivo sulla tile).

EVENT_MIN_CLUSTER_HA = 5.0       # area minima (ha) perche' un cluster apra un nuovo evento.
                                 # Coincide con MIN_ALERT_AREA_HA ma e' controllato separatamente
                                 # in pipeline per la decisione di apertura.

# ===========================================================================
# 6. LIFECYCLE EVENTO (tracking, conferma, chiusura)
#    Parametri della finestra temporale e dei criteri di conferma.
#
#    Quando una scena rileva un cluster >= EVENT_MIN_CLUSTER_HA non associabile
#    a eventi attivi, si apre un nuovo evento. Le scene valide successive
#    (SCENE_VALID_SCL_PCT >= soglia) aggiornano:
#      - obs_count   : n. osservazioni valide per pixel
#      - burnt_count : n. rilevamenti bruciato per pixel
#      - max_dnbr    : dNBR massimo nella finestra
#
#    Chiusura: al raggiungimento di EVENT_WINDOW_SCENES scene valide
#    OPPURE EVENT_TIMEOUT_DAYS giorni (soft, solo se n_valid >= EVENT_MIN_DETECTIONS).
#    Hard cap: EVENT_MAX_TIMEOUT_DAYS (chiude comunque).
#    Alla chiusura:
#      confirmed_mask = (burnt_count / obs_count >= EVENT_CONFIRM_RATIO)
#                       & (burnt_count >= EVENT_MIN_DETECTIONS)
# ===========================================================================

EVENT_WINDOW_SCENES    = 8       # n. scene valide per chiusura automatica
EVENT_TIMEOUT_DAYS     = 30      # giorni dall'alert per chiusura (soft: solo se n_valid >= EVENT_MIN_DETECTIONS)
EVENT_MAX_TIMEOUT_DAYS = 60      # hard cap assoluto: chiude comunque dopo N giorni
EVENT_MIN_DETECTIONS   = 4       # n. minimo di rilevamenti (burnt_count) per confermare un pixel

EVENT_BASELINE_BUFFER_PX = 0    # dilation (px) del footprint di un evento ancora attivo quando un altro
                                 # evento si chiude sulla stessa tile e aggiorna previous_nbr.
                                 # Protegge una zona di sicurezza intorno all'evento aperto: i pixel
                                 # del buffer mantengono il baseline pre-fire, non quello post-chiusura.
                                 # 0 = protegge solo i pixel esatti del footprint (nessun buffer).
                                 # A 20 m/px: 100 px = 2 km.

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

MIN_PATCH_PIXELS = 25            # rimuovi patch isolate < 25 pixel (< 1 ha a 20 m) dal raster finale
HOLE_FILL_PIXELS = 500           # riempi buchi < 500 pixel (~20 ha) nei poligoni finali

# ===========================================================================
# 8. BASELINE -- median composite pre-campagna
# ===========================================================================

CAMPAIGN_START_DATE = "2024-06-01"  # None = campagna parte da oggi; "YYYY-MM-DD" = data esplicita
BASELINE_LOOKBACK_DAYS = 35         # finestra retrospettiva pre-campagna (giorni)
BASELINE_MIN_SCENES = 3             # minimo scene cloud-free per il composite
BASELINE_MAD_K = 6                  # filtro MAD: scarta pixel < mediana - k*MAD
BASELINE_MAD_FLOOR = 0.005          # pavimento MAD: se MAD < eps, non filtrare

# ===========================================================================
# 9. SISTEMA -- flag di controllo e output opzionale
# ===========================================================================

# Se True, ignora lo stato JSON esistente e riprocessa tutto da zero
# (baseline + tutte le scene), sovrascrivendo il JSON al termine.
# Utile per forzare ricalcolo dopo cambio parametri algoritmici.
FORCE_REPROCESS = False

# Output RGB composito -- Highlight Optimized Natural Color (HONC)
# Se True, produce un composito RGB a 10 m per ogni scena con incendio rilevato.
# Formula L2A: cbrt(0.6 * riflettanza) per B4, B3, B2
# Ref: Marko Repse, Sentinel Hub Custom Scripts
#      https://sentinel-hub.github.io/custom-scripts/sentinel-2/highlight_optimized_natural_color/
PRODUCE_RGB = False
