import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FoodLoop AI", page_icon="🍽️", layout="wide")
st.title("🍽️ FoodLoop AI")
st.write("Machine learning based food-demand prediction for reducing avoidable cafeteria overproduction.")

try:
    model = joblib.load("models/food_demand_model.joblib")
except FileNotFoundError:
    st.error("Model not found. Run notebook 03 first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    expected = st.number_input("Expected attendance", 100, 1000, 430, 10)
    temp = st.number_input("Temperature (°C)", 5.0, 45.0, 28.0, 0.5)
    menu = st.selectbox("Menu type", ["North Indian","South Indian","Punjabi","Mixed","Light"])
with col2:
    exam = int(st.checkbox("Exam week"))
    event = int(st.checkbox("Special event"))
    if st.button("Predict meal demand", use_container_width=True):
        row = pd.DataFrame({
            "day_of_week":[day], "is_weekend":[int(day in ["Saturday","Sunday"])],
            "exam_week":[exam], "special_event":[event],
            "temperature_c":[temp], "menu_type":[menu],
            "expected_attendance":[expected]
        })
        predicted = round(float(model.predict(row)[0]))
        buffer = max(5, round(predicted * 0.025))
        st.metric("Predicted demand", f"{predicted} meals")
        st.metric("Recommended preparation", f"{predicted + buffer} meals")
        st.info(f"FoodLoop added a safety buffer of {buffer} meals.")

st.divider()
st.subheader("Why this matters")
st.write("The idea is to replace rough fixed preparation estimates with a data-driven demand prediction. Actual leftovers can be logged and used as future training data.")
st.caption("Student prototype using synthetic data. Real deployment needs validated cafeteria data and operational safety rules.")
