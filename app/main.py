import os
import streamlit as st
from dotenv import load_dotenv


from model_loader import load_mitbih_model, load_pcg_model, load_emg_model, load_vag_model
from chatbot import analyze_signal
from util import analyze_pcg_signal, analyze_emg_signal, predict_vag_from_features


load_dotenv()

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title="Biosignal Chatbot", page_icon="🩺", layout="centered")
st.title("🩺 Biosignal Diagnostic Chatbot")


@st.cache_resource
def get_models():
    return {
        "ECG": None,  
        "PCG": None,  
        "EMG": None,  
        "VAG": None,  
    }

MODELS = get_models()

FILE_TYPES = {
    "ECG": ["csv", "txt"],
    "EMG": ["csv", "txt"],
    "VAG": ["csv", "npy", "wav"],
    "PCG": ["wav"],               
}


tabs = st.tabs(["ECG", "EMG", "VAG", "PCG"])

for tab, sig in zip(tabs, ["ECG", "EMG", "VAG", "PCG"]):
    with tab:
        st.header(f"{sig} Analysis")

       
        if sig == "ECG":
            with st.expander("📄 ECG Data Requirements"):
                st.markdown(
                    "- Upload a `.csv` or `.txt` file containing **256 numeric values** (single row or single column).\n"
                    "- Example:\n"
                    "```csv\n0.12\n0.15\n-0.05\n...\n```"
                )
        elif sig == "VAG":
            with st.expander("📄 VAG Data Requirements"):
                st.markdown(
                "- Upload a `.csv` file **with headers** containing the following 5 features:\n"
                "  - `rms_amplitude`\n"
                "  - `peak_frequency`\n"
                "  - `spectral_entropy`\n"
                "  - `zero_crossing_rate`\n"
                "  - `mean_frequency`\n"
                "- Example file content:\n"
                "```csv\n"
                "rms_amplitude,peak_frequency,spectral_entropy,zero_crossing_rate,mean_frequency\n"
                "1.02,20,-1890.34,0.001,39.7\n"
                
                "```"
                )
        elif sig == "EMG":
            with st.expander("📄 EMG Data Requirements"):
                st.markdown(
                    "- Upload a `.txt` or `.csv` file containing **raw EMG signal samples**.\n"
                    "- The model expects **at least 1,000 values** (1-second window at 1 kHz sampling).\n"
                    "- You can provide:\n"
                    "  - A `.txt` file with one value per line.\n"
                    "  - A `.csv` file with a single column of numbers.\n\n"
                    "- Example `.txt` file:\n"
                    "```txt\n"
                    "0.034\n"
                    "0.056\n"
                    "-0.012\n"
                    "...\n"
                    "```"
                )
        elif sig == "PCG":
            with st.expander("📄 PCG Data Requirements"):
                st.markdown(
                    "- Upload a `.wav` file containing a **single-channel (mono) PCG signal**.\n"
                    "- The model expects **at least 995 audio samples** (≈0.025s of heart sound at 44.1 kHz).\n"
                    "- Files longer than 995 samples will be **trimmed**; shorter ones will be **zero-padded**.\n"
                    "- Ensure the signal is **clean and preprocessed** (no ambient noise).\n\n"
                    "- Example `.wav` properties:\n"
                    "  - Mono (1 channel)\n"
                    "  - 44.1 kHz sampling rate\n"
                    "  - 16-bit PCM or float32\n"
                    "\n"
                    "_Note: Do not upload `.mp3`, `.flac`, or stereo files—they may fail to process properly._"
                )


      
        uploaded = st.file_uploader(
            f"Upload {sig} file",
            type=FILE_TYPES[sig],
            key=f"upload_{sig}"
        )

       

        if sig == "ECG" and uploaded and st.button("Run Diagnostic", key=f"run_{sig}"):
            if MODELS["ECG"] is None:
                with st.spinner("Loading ECG model..."):
                    MODELS["ECG"] = load_mitbih_model()
            
            label, human, conf, gnote = analyze_signal(
                uploaded, MODELS["ECG"], GEMINI_API_KEY, signal_type="ECG"
            )
            st.success(f"**{label} – {human}**\n\nConfidence: {conf:.2%}")
            if gnote:
                st.markdown("### 🧠 Gemini Insight")
                st.write(gnote)
            elif not GEMINI_API_KEY:
                st.info("Gemini key missing – no explanation.")

        elif sig == "PCG" and uploaded and st.button("Run Diagnostic", key=f"run_{sig}"):
            if MODELS["PCG"] is None:
                with st.spinner("Loading PCG model..."):
                    MODELS["PCG"] = load_pcg_model()
            
            label, human, conf, gnote = analyze_pcg_signal(
                uploaded, MODELS["PCG"], GEMINI_API_KEY
            )
            st.success(f"**{label}**\n\nConfidence: {conf:.2%}")
            if gnote:
                st.markdown("### 🧠 Gemini Insight")
                st.write(gnote)
            elif not GEMINI_API_KEY:
                st.info("Gemini key missing – no explanation.")

        elif sig == "EMG" and uploaded and st.button("Run Diagnostic", key=f"run_{sig}"):
            if MODELS["EMG"] is None:
                with st.spinner("Loading EMG model..."):
                    MODELS["EMG"] = load_emg_model()
            
            human, conf, gnote = analyze_emg_signal(
                uploaded, MODELS["EMG"], GEMINI_API_KEY
            )
            st.success(f"**{human.upper()}**\n\nConfidence: {conf:.2%}")
            if gnote:
                st.markdown("### 🧠 Gemini Insight")
                st.write(gnote)
            elif not GEMINI_API_KEY:
                st.info("Gemini key missing – no explanation.")
                
        elif sig == "VAG" and uploaded and st.button("Run Diagnostic", key=f"run_{sig}"):
            if MODELS["VAG"] is None:
                with st.spinner("Loading VAG model..."):
                    MODELS["VAG"] = load_vag_model()
            
            label, human, conf, gnote = predict_vag_from_features(
                uploaded, MODELS["VAG"], GEMINI_API_KEY
            )
            st.success(f"**{label}**\n\nConfidence: {conf:.2%}")
            if gnote:
                st.markdown("### 🧠 Gemini Insight")
                st.write(gnote)
            elif not GEMINI_API_KEY:
                st.info("Gemini key missing – no explanation.")




        else:
            if not uploaded:
                st.info("Upload a file to begin.")


st.caption("© 2025 Biosignal Chatbot")
