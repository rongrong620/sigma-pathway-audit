# mediation_paths.py
"""
Central registry of mediation path sets for the ISMB2026 pipeline.

Stage04 compatibility note:
- Current Stage04 implementation supports:
  1) simple mediation: X -> M1 -> Y
  2) serial mediation (2 mediators): X -> M1 -> M2 -> Y
- Any paths requiring 3 mediators (M3) are stored separately and MUST NOT
  be passed into Stage04 unless you extend the runner.

Recommended usage in run_pipeline.py:
    from mediation_paths import PATH_SETS
    paths = PATH_SETS["part2"]  # or "part1", "part3"
"""

# ============================================================
# Part 1 paths (T01–T19): core set used in Part1
# ============================================================
PATHS_PART1 = [
    {"path_id":"T01", "type":"serial", "X":"Sleep quality",           "M1":"Vascular health", "M2":"Body fat composition", "Y":"Carotid IMT"},
    {"path_id":"T02", "type":"serial", "X":"Sleep heart rate / HRV",  "M1":"Vascular health", "M2":"Body fat composition", "Y":"Carotid IMT"},
    {"path_id":"T03", "type":"serial", "X":"Sleep OSA",               "M1":"Vascular health", "M2":"Body fat composition", "Y":"Carotid IMT"},

    {"path_id":"T04", "type":"simple", "X":"Sleep quality",           "M1":"Vascular health",      "Y":"Body fat composition"},
    {"path_id":"T05", "type":"simple", "X":"Sleep quality",           "M1":"Vascular health",      "Y":"Blood pressure lying"},
    {"path_id":"T06", "type":"simple", "X":"Sleep quality",           "M1":"Vascular health",      "Y":"Carotid IMT"},
    {"path_id":"T07", "type":"simple", "X":"Sleep quality",           "M1":"Body fat composition", "Y":"Carotid IMT"},

    {"path_id":"T08", "type":"simple", "X":"Sleep heart rate / HRV",  "M1":"Vascular health",      "Y":"Blood pressure lying"},
    {"path_id":"T09", "type":"simple", "X":"Sleep heart rate / HRV",  "M1":"Vascular health",      "Y":"Carotid IMT"},
    {"path_id":"T10", "type":"simple", "X":"Sleep heart rate / HRV",  "M1":"Vascular health",      "Y":"Body fat composition"},
    {"path_id":"T11", "type":"simple", "X":"Sleep heart rate / HRV",  "M1":"Body fat composition", "Y":"Blood pressure lying"},
    {"path_id":"T12", "type":"simple", "X":"Sleep heart rate / HRV",  "M1":"Body fat composition", "Y":"Carotid IMT"},

    {"path_id":"T13", "type":"simple", "X":"Sleep OSA",               "M1":"Vascular health",      "Y":"Blood pressure lying"},
    {"path_id":"T14", "type":"simple", "X":"Sleep OSA",               "M1":"Vascular health",      "Y":"Carotid IMT"},
    {"path_id":"T15", "type":"simple", "X":"Sleep OSA",               "M1":"Vascular health",      "Y":"Body fat composition"},
    {"path_id":"T16", "type":"simple", "X":"Sleep OSA",               "M1":"Body fat composition", "Y":"Blood pressure lying"},
    {"path_id":"T17", "type":"simple", "X":"Sleep OSA",               "M1":"Body fat composition", "Y":"Carotid IMT"},

    {"path_id":"T18", "type":"simple", "X":"Vascular health",         "M1":"Body fat composition", "Y":"Carotid IMT"},
    {"path_id":"T19", "type":"simple", "X":"Vascular health",         "M1":"Body fat composition", "Y":"Blood pressure lying"},
]

# ============================================================
# Part 2 paths (T20–T49): biology-driven set for Part2
# IMPORTANT: T20/T21 require M3 and are NOT compatible with current Stage04.
# They are stored in PATHS_PART2_NEEDS_M3 below.
# ============================================================
PATHS_PART2 = [
    # Serial (2 mediators)
    {"path_id":"T22","type":"serial","X":"Sleep quality","M1":"Blood pressure resting","M2":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T23","type":"serial","X":"Sleep quality","M1":"Blood pressure resting","M2":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T24","type":"serial","X":"Sleep quality","M1":"Blood pressure resting","M2":"Vascular health","Y":"Body fat composition"},
    {"path_id":"T25","type":"serial","X":"Sleep quality","M1":"Carotid IMT","M2":"Vascular health","Y":"Body fat composition"},
    {"path_id":"T26","type":"serial","X":"Blood pressure resting","M1":"Carotid IMT","M2":"Vascular health","Y":"Body fat composition"},

    {"path_id":"T27","type":"serial","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","M2":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T28","type":"serial","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","M2":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T29","type":"serial","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","M2":"Vascular health","Y":"Body fat composition"},
    {"path_id":"T30","type":"serial","X":"Sleep heart rate / HRV","M1":"Carotid IMT","M2":"Vascular health","Y":"Body fat composition"},
    {"path_id":"T31","type":"serial","X":"Blood pressure orthostatic","M1":"Carotid IMT","M2":"Vascular health","Y":"Body fat composition"},

    # Simple
    {"path_id":"T32","type":"simple","X":"Sleep quality","M1":"Blood pressure resting","Y":"Carotid IMT"},
    {"path_id":"T33","type":"simple","X":"Sleep quality","M1":"Blood pressure resting","Y":"Vascular health"},
    {"path_id":"T34","type":"simple","X":"Sleep quality","M1":"Blood pressure resting","Y":"Body fat composition"},
    {"path_id":"T35","type":"simple","X":"Sleep quality","M1":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T36","type":"simple","X":"Sleep quality","M1":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T37","type":"simple","X":"Sleep quality","M1":"Vascular health","Y":"Body fat composition"},

    {"path_id":"T38","type":"simple","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","Y":"Carotid IMT"},
    {"path_id":"T39","type":"simple","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","Y":"Vascular health"},
    {"path_id":"T40","type":"simple","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","Y":"Body fat composition"},
    {"path_id":"T41","type":"simple","X":"Sleep heart rate / HRV","M1":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T42","type":"simple","X":"Sleep heart rate / HRV","M1":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T43","type":"simple","X":"Sleep heart rate / HRV","M1":"Vascular health","Y":"Body fat composition"},

    {"path_id":"T44","type":"simple","X":"Blood pressure resting","M1":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T45","type":"simple","X":"Blood pressure resting","M1":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T46","type":"simple","X":"Blood pressure resting","M1":"Vascular health","Y":"Body fat composition"},

    {"path_id":"T47","type":"simple","X":"Blood pressure orthostatic","M1":"Carotid IMT","Y":"Vascular health"},
    {"path_id":"T48","type":"simple","X":"Blood pressure orthostatic","M1":"Carotid IMT","Y":"Body fat composition"},
    {"path_id":"T49","type":"simple","X":"Blood pressure orthostatic","M1":"Vascular health","Y":"Body fat composition"},
]

# ============================================================
# Part 3 paths (T50–T59): BP/Sleep -> IMT -> Gut microbiome
# ============================================================
PATHS_PART3 = [
    # Serial
    {"path_id":"T50","type":"serial","X":"Blood pressure standing","M1":"Nocturnal hypoxia burden","M2":"Carotid IMT","Y":"Gut microbiome abundance"},
    {"path_id":"T51","type":"serial","X":"Blood pressure standing","M1":"Sleep OSA","M2":"Body fat composition","Y":"Gut microbiome abundance"},

    # Simple
    {"path_id":"T52","type":"simple","X":"Blood pressure standing","M1":"Nocturnal hypoxia burden","Y":"Carotid IMT"},
    {"path_id":"T53","type":"simple","X":"Blood pressure standing","M1":"Nocturnal hypoxia burden","Y":"Gut microbiome abundance"},
    {"path_id":"T54","type":"simple","X":"Blood pressure standing","M1":"Carotid IMT","Y":"Gut microbiome abundance"},
    {"path_id":"T55","type":"simple","X":"Nocturnal hypoxia burden","M1":"Carotid IMT","Y":"Gut microbiome abundance"},
    {"path_id":"T56","type":"simple","X":"Blood pressure standing","M1":"Sleep OSA","Y":"Gut microbiome abundance"},
    {"path_id":"T57","type":"simple","X":"Blood pressure standing","M1":"Sleep OSA","Y":"Body fat composition"},
    {"path_id":"T58","type":"simple","X":"Blood pressure standing","M1":"Body fat composition","Y":"Gut microbiome abundance"},
    {"path_id":"T59","type":"simple","X":"Sleep OSA","M1":"Body fat composition","Y":"Gut microbiome abundance"},
]

# ============================================================
# Paths that require extending Stage04 (3 mediators / M3)
# Interpretation assumed: X -> M1 -> M2 -> M3 -> Y
# ============================================================
PATHS_NEED_M3_SUPPORT = [
    {"path_id":"T20","type":"serial","X":"Sleep quality","M1":"Blood pressure resting","M2":"Carotid IMT","M3":"Vascular health","Y":"Body fat composition"},
    {"path_id":"T21","type":"serial","X":"Sleep heart rate / HRV","M1":"Blood pressure resting","M2":"Carotid IMT","M3":"Vascular health","Y":"Body fat composition"},
]

# ============================================================
# Unified entrypoint for pipeline
# ============================================================
PATH_SETS = {
    "part1": PATHS_PART1,
    "part2": PATHS_PART2,
    "part3": PATHS_PART3,
}

# Optional: all compatible paths together (if you ever want a single run)
PATHS_ALL_COMPATIBLE = PATHS_PART1 + PATHS_PART2 + PATHS_PART3