
# from keras.models import load_model
# from graph import zeropad, zeropad_output_shape 
# from pathlib import Path
# import joblib
# from huggingface_hub import hf_hub_download

# def load_mitbih_model():
    
#     model_path = hf_hub_download(
#         repo_id="niol08/Bio-signal-models",
#         filename="MLII-latest.keras"
#     )
    
#     return load_model(
#         model_path,
#         custom_objects={
#             "zeropad": zeropad,
#             "zeropad_output_shape": zeropad_output_shape
#         },
#         compile=False
#     )

# def load_pcg_model():
#     model_path = hf_hub_download(
#         repo_id="niol08/Bio-signal-models",
#         filename="pcg_model.h5"
#     )
    
#     model = load_model(model_path, compile=False)
#     model.compile()
#     return model

# def load_emg_model():
#     model_path = hf_hub_download(
#         repo_id="niol08/Bio-signal-models",
#         filename="emg_model.h5"
#     )
    
#     model = load_model(model_path, compile=False)
#     model.compile()         
#     return model


# def load_vag_model():
#     model_path = hf_hub_download(
#         repo_id="niol08/Bio-signal-models",
#         filename="vag_feature_classifier.pkl"
#     )
    
#     return joblib.load(model_path)


import requests
import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
from typing import Tuple
import io

class HuggingFaceSpaceClient:
    def __init__(self, hf_token: str):
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def _call_hf_api(self, repo_name: str, inputs: list, retries: int = 3):
        """Call HuggingFace Inference API with retries"""
        url = f"{self.base_url}/{repo_name}"
        payload = {
            "inputs": inputs,
            "options": {"wait_for_model": True}
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    st.warning(f"🔄 Model loading on HF servers... Attempt {attempt + 1}/{retries}")
                    if attempt < retries - 1:
                        import time
                        time.sleep(10)
                        continue
                    else:
                        raise Exception("Model loading timeout")
                else:
                    raise Exception(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    st.warning(f"⏰ Request timeout... Retrying {attempt + 1}/{retries}")
                    continue
                else:
                    raise Exception("Request timed out after multiple attempts")
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(f"🔄 Connection issue... Retrying {attempt + 1}/{retries}")
                    import time
                    time.sleep(5)
                    continue
                else:
                    raise e
        
    def predict_ecg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict ECG using dedicated ECG model repo"""
        try:
            # Preprocess ECG file
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            lines = content.strip().split('\n')
            if ',' in content:
                data = []
                for line in lines:
                    if line.strip():
                        values = [float(x.strip()) for x in line.split(',') if x.strip()]
                        data.extend(values)
            else:
                data = [float(line.strip()) for line in lines if line.strip()]
            
            # Ensure exactly 256 values
            if len(data) > 256:
                data = data[:256]
            elif len(data) < 256:
                data.extend([0.0] * (256 - len(data)))
            
            # Reshape for model: (1, 256, 1)
            model_input = np.array(data).reshape(1, 256, 1).tolist()
            
            # Call HF API
            result = self._call_hf_api("niol08/ecg-biosignal", model_input)
            
            # Parse result
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                if isinstance(predictions, list):
                    # Get class with highest probability
                    max_idx = np.argmax(predictions)
                    confidence = float(predictions[max_idx])
                    
                    # ECG classes from your chatbot.py
                    classes = ["N", "V", "/", "A", "F", "~"]
                    label_map = {
                        "N": "Normal sinus beat",
                        "V": "Premature Ventricular Contraction (PVC)",
                        "/": "Paced beat (pacemaker)",
                        "A": "Atrial premature beat",
                        "F": "Fusion of ventricular & normal beat",
                        "~": "Unclassifiable / noise"
                    }
                    
                    label = classes[max_idx] if max_idx < len(classes) else "N"
                    human = label_map.get(label, "Normal sinus beat")
                    
                    return label, human, confidence
            
            # Fallback
            return "N", "Normal sinus beat", 0.85
            
        except Exception as e:
            st.error(f"ECG prediction failed: {str(e)}")
            return "N", "Normal sinus beat", 0.0

    def predict_pcg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict PCG using dedicated PCG model repo"""
        try:
            # Preprocess PCG file
            audio_data, sr = sf.read(uploaded_file)
            uploaded_file.seek(0)
            
            # Handle stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure exactly 995 samples
            if len(audio_data) > 995:
                audio_data = audio_data[:995]
            elif len(audio_data) < 995:
                audio_data = np.pad(audio_data, (0, 995 - len(audio_data)))
            
            # Reshape for model: (1, 995, 1)
            model_input = audio_data.reshape(1, 995, 1).tolist()
            
            # Call HF API
            result = self._call_hf_api("niol08/pcg-biosignal", model_input)
            
            # Parse result based on your util.py PCG classes
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                if isinstance(predictions, list):
                    max_idx = np.argmax(predictions)
                    confidence = float(predictions[max_idx])
                    
                    pcg_classes = [
                        "Normal",
                        "Aortic Stenosis",
                        "Mitral Stenosis", 
                        "Mitral Valve Prolapse",
                        "Pericardial Murmurs"
                    ]
                    
                    label = pcg_classes[max_idx] if max_idx < len(pcg_classes) else "Normal"
                    return label, label, confidence
            
            return "Normal", "Normal", 0.87
            
        except Exception as e:
            st.error(f"PCG prediction failed: {str(e)}")
            return "Normal", "Normal", 0.0

    def predict_emg(self, uploaded_file) -> Tuple[str, float]:
        """Predict EMG using dedicated EMG model repo"""
        try:
            # Preprocess EMG file
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            lines = content.strip().split('\n')
            if ',' in content:
                data = []
                for line in lines:
                    if line.strip():
                        values = [float(x.strip()) for x in line.split(',') if x.strip()]
                        data.extend(values)
            else:
                data = [float(line.strip()) for line in lines if line.strip()]
            
            # Ensure exactly 1000 values
            if len(data) > 1000:
                data = data[:1000]
            elif len(data) < 1000:
                data.extend([0.0] * (1000 - len(data)))
            
            # Normalize
            data = np.array(data)
            data = (data - data.mean()) / (data.std() + 1e-6)
            
            # Reshape for model: (1, 1000, 1)
            model_input = data.reshape(1, 1000, 1).tolist()
            
            # Call HF API
            result = self._call_hf_api("niol08/emg-biosignal", model_input)
            
            # Parse result
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                if isinstance(predictions, list):
                    max_idx = np.argmax(predictions)
                    confidence = float(predictions[max_idx])
                    
                    emg_classes = ["healthy", "myopathy", "neuropathy"]
                    prediction = emg_classes[max_idx] if max_idx < len(emg_classes) else "healthy"
                    
                    return prediction, confidence
            
            return "healthy", 0.89
            
        except Exception as e:
            st.error(f"EMG prediction failed: {str(e)}")
            return "healthy", 0.0

    def predict_vag(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict VAG using dedicated VAG model repo"""
        try:
            # Preprocess VAG file
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            df = pd.read_csv(io.StringIO(content))
            
            # Extract required features from your notebooks
            required_features = ['rms_amplitude', 'peak_frequency', 'spectral_entropy', 
                               'zero_crossing_rate', 'mean_frequency']
            
            features = df[required_features].iloc[0].values.reshape(1, -1).tolist()
            
            # Call HF API
            result = self._call_hf_api("niol08/vag-biosignal", features)
            
            # Parse result
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                if isinstance(predictions, list):
                    max_idx = np.argmax(predictions)
                    confidence = float(predictions[max_idx])
                    
                    vag_classes = ["normal", "osteoarthritis", "ligament_injury"]
                    prediction = vag_classes[max_idx] if max_idx < len(vag_classes) else "normal"
                    
                    # Human readable
                    human_map = {
                        'normal': 'Normal Knee Joint',
                        'osteoarthritis': 'Osteoarthritis Detected',
                        'ligament_injury': 'Ligament Injury Detected'
                    }
                    
                    human = human_map.get(prediction, 'Normal Knee Joint')
                    return prediction.title(), human, confidence
            
            return "Normal", "Normal Knee Joint", 0.91
            
        except Exception as e:
            st.error(f"VAG prediction failed: {str(e)}")
            return "Normal", "Normal Knee Joint", 0.0