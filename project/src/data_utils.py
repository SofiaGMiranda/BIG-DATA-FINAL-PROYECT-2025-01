import pandas as pd
import streamlit as st

@st.cache_data
def process_data(file):
    df = pd.read_csv(file)
    df.drop_duplicates(inplace=True)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days.clip(lower=0)
    df['Admission Day of Week'] = df['Date of Admission'].dt.day_name()
    df['Admission Month'] = df['Date of Admission'].dt.month_name()
    df['Admission Year'] = df['Date of Admission'].dt.year
    return df