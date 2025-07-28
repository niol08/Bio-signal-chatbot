
from keras.models import load_model
from graph import zeropad, zeropad_output_shape 
from pathlib import Path
import joblib
from download_models import ensure_models_downloaded

def load_mitbih_model():
    ensure_models_downloaded()
    return load_model(
        "models/MLII-latest.keras",
        custom_objects={
            "zeropad": zeropad,
            "zeropad_output_shape": zeropad_output_shape
        },
        compile=False
    )

def load_pcg_model():
    ensure_models_downloaded()
    model_path = Path("models/pcg_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"PCG model not found at {model_path.resolve()}")
    
    model = load_model(model_path, compile=False)
    model.compile()
    return model

def load_emg_model():
    ensure_models_downloaded()
    model_path = Path("models/emg_classifier_txt.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"EMG model not found at {model_path.resolve()}")
    model = load_model(model_path, compile=False)
    model.compile()         
    return model

from keras.models import load_model

def load_vag_model():
    ensure_models_downloaded()
    p = Path("models/vag_feature_classifier.pkl")
    if not p.exists():
        raise FileNotFoundError(f"No VAG model at {p.resolve()}")
    return joblib.load(p)    
    