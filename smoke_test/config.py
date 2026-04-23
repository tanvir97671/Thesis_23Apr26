"""Smoke test configuration — all constants in one place."""

# ---- Dataset ----
SCENARIO_NAME = "city_18_denver"   # 1 of 6 available (minimal download)
DATA_FRACTION = 0.01               # use 1% of scenario
MAX_SAMPLES = 500                  # hard cap

# ---- HuggingFace repos ----
MODEL_REPO_ID = "wi-lab/lwm"
DATASET_REPO_ID = "wi-lab/lwm"

# ---- LWM architecture constants ----
NUM_ANTENNAS = 32
NUM_SUBCARRIERS = 32
PATCH_SIZE = 16
EMBED_DIM = 64

# ---- Inference ----
BATCH_SIZE = 32
EMBEDDING_TYPE = "channel_emb"
GEN_RAW = True

# ---- Downstream task ----
NUM_BEAMS = 64

# ---- Validation thresholds ----
MIN_GPU_MEM_GB = 30
MAX_TEST_TIME_SEC = 300
