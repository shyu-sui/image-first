import streamlit as st
from PIL import Image

option_radio = st.radio(
  "chose your favorit women"
  ('Miakarifa', 'FukadaEimi', 'EmaWatson', 'YagiNana'))

st.write('Your favorite is:', option_radio)

