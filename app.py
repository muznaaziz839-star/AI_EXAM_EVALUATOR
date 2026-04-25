import streamlit as st
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load model
from keras.models import load_model

model = load_model("model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 30

st.set_page_config(page_title="AI Exam Evaluator")

st.title("📝 AI Exam Answer Evaluator")

question = st.text_input("Enter Question")
ideal = st.text_area("Enter Ideal Answer")
student = st.text_area("Enter Student Answer")

if st.button("Check Answer"):

    text = question + " " + ideal + " " + student

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)[0][0]

    score = round(float(pred), 2)

    if score < 0:
        score = 0
    if score > 10:
        score = 10

    st.success(f"Predicted Score = {score}/10")

    if score >= 8:
        st.info("Excellent Answer")
    elif score >= 5:
        st.warning("Average Answer")
    else:
        st.error("Poor Answer")