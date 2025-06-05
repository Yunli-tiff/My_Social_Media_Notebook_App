import whisper
import streamlit as st

model = whisper.load_model("base")

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file.name)
    return result["text"]
