
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


import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
from typing import Tuple
import io
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow import keras
import joblib
import tempfile
import os

class HuggingFaceSpaceClient:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.repo_id = "niol08/Bio-signal-models"
        
        # Exact model filenames from your repo
        self.models = {
            "ECG": "MLII-latest.keras",
            "PCG": "pcg_model.h5", 
            "EMG": "emg_classifier_txt.h5",
            "VAG": "vag_feature_classifier.pkl"
        }
        
        # Cache for loaded models
        self.loaded_models = {}
    
    def _download_and_load_model(self, signal_type: str):
        """Download and load model from HuggingFace Hub"""
        if signal_type in self.loaded_models:
            return self.loaded_models[signal_type]
        
        model_filename = self.models[signal_type]
        
        st.info(f"🔄 Downloading {model_filename} from HuggingFace...")
        
        try:
            # Download model file using HuggingFace Hub
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=model_filename,
                token=self.hf_token
            )
            
            st.success(f"✅ Downloaded {model_filename}")
            
            # Load the appropriate model type
            if signal_type == "ECG":
                st.info("🧠 Loading ECG Keras model...")
                model = keras.models.load_model(model_path, compile=False)
                
            elif signal_type == "PCG":
                st.info("🧠 Loading PCG Keras model...")
                model = keras.models.load_model(model_path, compile=False)
                
            elif signal_type == "EMG":
                st.info("🧠 Loading EMG Keras model...")
                model = keras.models.load_model(model_path, compile=False)
                
            elif signal_type == "VAG":
                st.info("🧠 Loading VAG Scikit-learn model...")
                model = joblib.load(model_path)
            
            # Cache the loaded model
            self.loaded_models[signal_type] = model
            st.success(f"✅ {signal_type} model loaded successfully!")
            
            return model
            
        except Exception as e:
            st.error(f"❌ Failed to download/load {signal_type} model: {str(e)}")
            raise e

    def predict_ecg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict ECG using MLII-latest.keras from HuggingFace"""
        # Download and load ECG model
        model = self._download_and_load_model("ECG")
        
        # Process ECG data
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        # Parse ECG data
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                data.extend(values)
            else:
                try:
                    data.append(float(line.strip()))
                except ValueError:
                    continue
        
        if len(data) == 0:
            raise Exception("No numeric data found in ECG file")
        
        # Ensure exactly 256 values for MLII-latest.keras
        if len(data) > 256:
            data = data[:256]
        elif len(data) < 256:
            data.extend([0.0] * (256 - len(data)))
        
        # Prepare data for model (batch_size=1, sequence_length=256, features=1)
        model_input = np.array(data).reshape(1, 256, 1)
        
        st.info("🧠 Running ECG prediction with HuggingFace model...")
        
        # Make prediction
        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # ECG classes based on MIT-BIH database
        ecg_classes = ["N", "V", "/", "A", "F", "~"]
        class_names = {
            "N": "Normal sinus beat",
            "V": "Premature Ventricular Contraction (PVC)",
            "/": "Paced beat (pacemaker)",
            "A": "Atrial premature beat",
            "F": "Fusion of ventricular & normal beat",
            "~": "Unclassifiable / noise"
        }
        
        predicted_label = ecg_classes[predicted_class_idx]
        human_readable = class_names[predicted_label]
        
        return predicted_label, human_readable, confidence

    def predict_pcg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict PCG using pcg_model.h5 from HuggingFace"""
        # Download and load PCG model
        model = self._download_and_load_model("PCG")
        
        # Process PCG audio
        audio_data, sr = sf.read(uploaded_file)
        uploaded_file.seek(0)
        
        # Handle stereo to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure exactly 995 samples for pcg_model.h5
        if len(audio_data) > 995:
            audio_data = audio_data[:995]
        elif len(audio_data) < 995:
            audio_data = np.pad(audio_data, (0, 995 - len(audio_data)))
        
        # Prepare data for model (batch_size=1, sequence_length=995, features=1)
        model_input = audio_data.reshape(1, 995, 1)
        
        st.info("🧠 Running PCG prediction with HuggingFace model...")
        
        # Make prediction
        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # PCG classes (adjust based on your model's training)
        pcg_classes = [
            "Normal",
            "Aortic Stenosis", 
            "Mitral Stenosis",
            "Mitral Valve Prolapse",
            "Pericardial Murmurs"
        ]
        
        predicted_label = pcg_classes[predicted_class_idx] if predicted_class_idx < len(pcg_classes) else "Normal"
        
        return predicted_label, predicted_label, confidence

    def predict_emg(self, uploaded_file) -> Tuple[str, float]:
        """Predict EMG using emg_classifier_txt.h5 from HuggingFace"""
        # Download and load EMG model
        model = self._download_and_load_model("EMG")
        
        # Process EMG data
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                data.extend(values)
            else:
                try:
                    data.append(float(line.strip()))
                except ValueError:
                    continue
        
        if len(data) == 0:
            raise Exception("No numeric data found in EMG file")
        
        # Ensure exactly 1000 values for emg_classifier_txt.h5
        if len(data) > 1000:
            data = data[:1000]
        elif len(data) < 1000:
            data.extend([0.0] * (1000 - len(data)))
        
        # Normalize EMG data
        data_array = np.array(data)
        normalized_data = (data_array - data_array.mean()) / (data_array.std() + 1e-6)
        
        # Prepare data for model (batch_size=1, sequence_length=1000, features=1)
        model_input = normalized_data.reshape(1, 1000, 1)
        
        st.info("🧠 Running EMG prediction with HuggingFace model...")
        
        # Make prediction
        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # EMG classes (3 classes: healthy, myopathy, neuropathy)
        emg_classes = ["healthy", "myopathy", "neuropathy"]
        predicted_label = emg_classes[predicted_class_idx] if predicted_class_idx < len(emg_classes) else "healthy"
        
        return predicted_label, confidence

    def predict_vag(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict VAG using vag_feature_classifier.pkl from HuggingFace"""
        # Download and load VAG model
        model = self._download_and_load_model("VAG")
        
        # Process VAG features
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        df = pd.read_csv(io.StringIO(content))
        
        # Required features for vag_feature_classifier.pkl
        required_features = ['rms_amplitude', 'peak_frequency', 'spectral_entropy', 
                           'zero_crossing_rate', 'mean_frequency']
        
        if not all(feature in df.columns for feature in required_features):
            raise Exception(f"Missing required features. Need: {required_features}")
        
        features = df[required_features].iloc[0].values.reshape(1, -1)
        
        st.info("🧠 Running VAG prediction with HuggingFace model...")
        
        # Make prediction with scikit-learn model
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 0.85  # Default confidence for models without probability
        
        # VAG classes (adjust based on your model's training)
        vag_mapping = {
            0: ("Normal", "Normal Knee Joint"),
            1: ("Osteoarthritis", "Osteoarthritis Detected"), 
            2: ("Ligament Injury", "Ligament Injury Detected")
        }
        
        if isinstance(prediction, (int, np.integer)):
            label, human = vag_mapping.get(prediction, ("Normal", "Normal Knee Joint"))
        else:
            # If prediction is string
            prediction_lower = str(prediction).lower() 
            if 'osteo' in prediction_lower:
                label, human = "Osteoarthritis", "Osteoarthritis Detected"
            elif 'ligament' in prediction_lower:
                label, human = "Ligament Injury", "Ligament Injury Detected"
            else:
                label, human = "Normal", "Normal Knee Joint"
        
        return label, human, confidence