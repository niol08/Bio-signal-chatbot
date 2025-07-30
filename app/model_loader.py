
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
from typing import Tuple
import base64
import io

class HuggingFaceSpaceClient:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        
        self.space_url = "https://niol08-Bio-signal-chatbot.hf.space"  
    
    def _upload_file_to_space(self, file_data, signal_type: str, filename: str):
        """Upload file to your HF Space and get prediction"""
        
        # Convert file to base64 for API transmission
        if hasattr(file_data, 'read'):
            file_bytes = file_data.read()
            file_data.seek(0)  # Reset for potential reuse
        else:
            file_bytes = file_data
            
        file_b64 = base64.b64encode(file_bytes).decode()
        
        payload = {
            "signal_type": signal_type,
            "filename": filename,
            "file_data": file_b64
        }
        
        try:
            response = requests.post(
                f"{self.space_url}/predict",  # Your HF Space endpoint
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"HF Space Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except Exception as e:
            raise Exception(f"Failed to connect to HF Space: {str(e)}")
    
    def predict_ecg(self, uploaded_file) -> Tuple[str, str, float]:
        """Send ECG file to HF Space for prediction"""
        result = self._upload_file_to_space(uploaded_file, "ECG", uploaded_file.name)
        
        return (
            result.get("label", "Unknown"),
            result.get("human", "Unknown"),
            result.get("confidence", 0.0)
        )
    
    def predict_pcg(self, uploaded_file) -> Tuple[str, str, float]:
        """Send PCG file to HF Space for prediction"""
        result = self._upload_file_to_space(uploaded_file, "PCG", uploaded_file.name)
        
        return (
            result.get("label", "Unknown"), 
            result.get("human", "Unknown"),
            result.get("confidence", 0.0)
        )
    
    def predict_emg(self, uploaded_file) -> Tuple[str, float]:
        """Send EMG file to HF Space for prediction"""
        result = self._upload_file_to_space(uploaded_file, "EMG", uploaded_file.name)
        
        return (
            result.get("prediction", "unknown"),
            result.get("confidence", 0.0)
        )
    
    def predict_vag(self, uploaded_file) -> Tuple[str, str, float]:
        """Send VAG file to HF Space for prediction"""
        result = self._upload_file_to_space(uploaded_file, "VAG", uploaded_file.name)
        
        return (
            result.get("label", "Unknown"),
            result.get("human", "Unknown"), 
            result.get("confidence", 0.0)
        )