import streamlit as st

st.set_page_config(
    page_title="AI Exam Answer Evaluator",
    layout="centered"
)

st.title("📝 AI Exam Answer Evaluator")
st.write("Simple AI-based answer scoring system")

question = st.text_input("Enter Question")
ideal_answer = st.text_area("Enter Ideal Answer")
student_answer = st.text_area("Enter Student Answer")

def calculate_score(ideal, student):
    if not ideal or not student:
        return 0

    ideal_words = set(ideal.lower().split())
    student_words = set(student.lower().split())

    if len(ideal_words) == 0:
        return 0

    match = len(ideal_words.intersection(student_words))

    score = (match / len(ideal_words)) * 10

    return round(score, 2)

if st.button("Evaluate Answer"):

    score = calculate_score(ideal_answer, student_answer)

    st.success(f"📊 Score: {score}/10")

    if score >= 8:
        st.info("🌟 Excellent Answer")
    elif score >= 5:
        st.warning("👍 Average Answer")
    else:
        st.error("❌ Poor Answer")

    st.write("### 🔍 Analysis")
    st.write("Matching keywords:", len(set(ideal_answer.lower().split()).intersection(set(student_answer.lower().split()))))