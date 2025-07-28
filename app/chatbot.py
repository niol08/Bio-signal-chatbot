
import numpy as np
from util import load_uploaded_file, segment_signal
from gemini import query_gemini_rest

CLASSES = ["N", "V", "/", "A", "F", "~"]
LABEL_MAP = {
    "N": "Normal sinus beat",
    "V": "Premature Ventricular Contraction (PVC)",
    "/": "Paced beat (pacemaker)",
    "A": "Atrial premature beat",
    "F": "Fusion of ventricular & normal beat",
    "~": "Unclassifiable / noise"
}

def analyze_signal(file, model, gemini_key="", signal_type="ECG"):
    signal = load_uploaded_file(file, signal_type)
    segments = segment_signal(signal)          

    preds = model.predict(segments, verbose=0)[0]    
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])                        
    label = CLASSES[idx]
    human = LABEL_MAP[label]

    gemini_txt = None
    if gemini_key:
        gemini_txt = query_gemini_rest(signal_type, human, conf, gemini_key)

    return label, human, conf, gemini_txt
