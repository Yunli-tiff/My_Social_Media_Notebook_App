import pytesseract
from PIL import Image
import streamlit as st

@st.cache_data(show_spinner=False)
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    return pytesseract.image_to_string(img, lang="chi_tra+eng")
