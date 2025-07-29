
from tensorflow.keras.models import load_model
from graph import zeropad, zeropad_output_shape 
from pathlib import Path
import joblib
from download_models import ensure_models_downloaded

def load_mitbih_model():
    ensure_models_downloaded()
    
    models_dir = Path("models")
    if models_dir.exists():
        print(f"Models directory contents: {list(models_dir.glob('*'))}")
    else:
        print("Models directory doesn't exist!")
        models_dir.mkdir(exist_ok=True)

    possible_paths = [
            "models/MLII-latest.keras",
            "../models/MLII-latest.keras",
            "app/models/MLII-latest.keras"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            print(f"Found ECG model at: {path}")
            break
    
    if not model_path:
        raise FileNotFoundError(f"ECG model not found. Checked: {possible_paths}")
    
    return load_model(
        model_path,
        custom_objects={
            "zeropad": zeropad,
            "zeropad_output_shape": zeropad_output_shape
        },
        compile=False
    )

def load_pcg_model():
    ensure_models_downloaded()
    models_dir = Path("models")
    if models_dir.exists():
        print(f"Models directory for PCG: {list(models_dir.glob('*.h5'))}")
    
    possible_paths = [
        "models/pcg_model.h5",
        "../models/pcg_model.h5",
        "app/models/pcg_model.h5"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
        
        
    if not model_path:
        raise FileNotFoundError(f"PCG model not found at {possible_paths}")
    
    model = load_model(model_path, compile=False)
    model.compile()
    return model

def load_emg_model():
    ensure_models_downloaded()
    possible_paths = [
        "models/emg_classifier_txt.h5",
        "../models/emg_classifier_txt.h5",
        "app/models/emg_classifier_txt.h5"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
    if not model_path:
        raise FileNotFoundError(f"EMG model not found at {possible_paths}")
    model = load_model(model_path, compile=False)
    model.compile()         
    return model


def load_vag_model():
    ensure_models_downloaded()
    possible_paths = [
        "models/vag_feature_classifier.pkl",
        "../models/vag_feature_classifier.pkl",
        "app/models/vag_feature_classifier.pkl"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
        
    if not model_path:
        raise FileNotFoundError(f"No VAG model at {possible_paths}")
    
    return joblib.load(model_path)
