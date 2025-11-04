import os
import gdown
import pickle

# ===== CONFIGURE YOUR MODEL FILES =====
MODELS = {
    "rf": {
        "file_id": "https://drive.google.com/file/d/1cfB6a7-CBTAvSNWdLJi__DSFxN03ldRK/view?usp=sharing",
        "filename": "rf_model.pkl"
    },
    "xgb": {
        "file_id": "https://drive.google.com/file/d/1k7FZF1mdzhVVuN1rogplyhFGQncOcYEx/view?usp=sharing",
        "filename": "xgb_model.pkl"
    }
}

def download_model(file_id, filename):
    """Download model from Google Drive if not present locally."""
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists. Skipping download.")

def load_model(filename):
    """Load pickle model."""
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        print(f"{filename} loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return None

# ===== DOWNLOAD & LOAD MODELS =====
download_model(MODELS["rf"]["file_id"], MODELS["rf"]["filename"])
download_model(MODELS["xgb"]["file_id"], MODELS["xgb"]["filename"])

rf_model = load_model(MODELS["rf"]["filename"])
xgb_model = load_model(MODELS["xgb"]["filename"])
