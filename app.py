import streamlit as st
from functions import main, input_c, precipitation

st.title("田んぼダムの計算")

uploaded_file_input_c = st.file_uploader("input_cのCSVファイルをアップロードしてください", type="csv", key='input_c')
uploaded_file_precipitation = st.file_uploader("precipitationのCSVファイルをアップロードしてください", type="csv", key='precipitation')

if uploaded_file_input_c is not None and uploaded_file_precipitation is not None:
    dt = st.number_input("dtを入力してください", value=0)
    tout = st.number_input("toutを入力してください", value=0)
    runtime = st.number_input("runtimeを入力してください", value=0)

    if st.button("計算開始"):
        input_c(uploaded_file_input_c)
        precipitation(uploaded_file_precipitation)
        main(dt, tout, runtime)
