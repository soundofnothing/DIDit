import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from signature import Fingerprint
import streamlit as st
import requests
from bs4 import BeautifulSoup


def get_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Failed to retrieve the URL: {e}")
        return None

def compare_fingerprints(inputs):
    figures = []
    data_tables = []
    for input_type, input_value in inputs:
        if input_type == "URL":
            input_value = get_text_from_url(input_value)
            if input_value is None:
                continue  # Skip to the next input if URL retrieval failed
        
        fingerprint = Fingerprint.from_text(input_value)
        fig, df = visualize_fingerprint_identity(fingerprint)
        figures.append(fig)
        data_tables.append(df)
    
    for i, (fig, df) in enumerate(zip(figures, data_tables)):
        st.write(f'### Text Snippet {i + 1}')
        st.plotly_chart(fig)
        st.write(df)

st.title("Authorship Inference")

num_inputs = st.number_input("Number of Inputs", min_value=1, value=1)

inputs = []
for i in range(num_inputs):
    input_type = st.selectbox(f"Input Type {i+1}", ["Text", "URL"], key=f"input_type_{i}")
    input_value = st.text_area(f"Enter Text or URL {i+1}", key=f"input_value_{i}")
    if input_value:  # Only add non-empty inputs
        inputs.append((input_type, input_value))

if st.button('Analyze'):
    if inputs:
        compare_fingerprints(inputs)
    else:
        st.error("Please enter at least one text snippet or URL.")