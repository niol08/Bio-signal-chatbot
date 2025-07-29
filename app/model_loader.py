
from keras.models import load_model
from graph import zeropad, zeropad_output_shape 
from pathlib import Path
import joblib
from huggingface_hub import hf_hub_download

def load_mitbih_model():
    
    model_path = hf_hub_download(
        repo_id="niol08/Bio-signal-models",
        filename="MLII-latest.keras"
    )
    
    return load_model(
        model_path,
        custom_objects={
            "zeropad": zeropad,
            "zeropad_output_shape": zeropad_output_shape
        },
        compile=False
    )

def load_pcg_model():
    model_path = hf_hub_download(
        repo_id="niol08/Bio-signal-models",
        filename="pcg_model.h5"
    )
    
    model = load_model(model_path, compile=False)
    model.compile()
    return model

def load_emg_model():
    model_path = hf_hub_download(
        repo_id="niol08/Bio-signal-models",
        filename="emg_model.h5"
    )
    
    model = load_model(model_path, compile=False)
    model.compile()         
    return model


def load_vag_model():
    model_path = hf_hub_download(
        repo_id="niol08/Bio-signal-models",
        filename="vag_feature_classifier.pkl"
    )
    
    return joblib.load(model_path)
