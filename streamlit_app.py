import streamlit as st
import pandas as pd
import pickle
import gzip
from predictor import BookingPredictor  # pastikan file ini ada di folder yang sama

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="centered")
st.title("Hotel Reservation Status Classifier 🏨")

# Load model dan scaler dari file .pkl.gz dan .pkl
with gzip.open("best_rf_model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    standard_scaler = pickle.load(f)

with open("robust_scaler.pkl", "rb") as f:
    robust_scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Inisialisasi objek predictor
predictor = BookingPredictor(
    model_path=None,
    standard_scaler_path=None,
    robust_scaler_path=None,
    columns_path=None
)

predictor.model = model
predictor.standard_scaler = standard_scaler
predictor.robust_scaler = robust_scaler
predictor.columns = columns

# Test case
test_case_1 = {
    "Booking_ID": "INN00001",
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 224,
    "arrival_year": 2017,
    "arrival_month": 10,
    "arrival_date": 2,
    "market_segment_type": "Offline",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 65.0,
    "no_of_special_requests": 0
}

test_case_2 = {
    "Booking_ID": "INN00003",
    "no_of_adults": 1,
    "no_of_children": 0,
    "no_of_weekend_nights": 2,
    "no_of_week_nights": 1,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 1,
    "arrival_year": 2018,
    "arrival_month": 2,
    "arrival_date": 28,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 60.0,
    "no_of_special_requests": 0
}

# Dropdown untuk pilih test case
test_options = {
    "Test Case 1": test_case_1,
    "Test Case 2": test_case_2
}

option = st.selectbox("🔍 Pilih Test Case", list(test_options.keys()))
selected_data = test_options[option]

# Tampilkan data input
st.write("### 📋 Data yang digunakan:")
st.dataframe(pd.DataFrame([selected_data]))

# Tombol prediksi
if st.button("Predict"):
    input_df = pd.DataFrame([selected_data])
    prediction = predictor.predict(input_df)
    result = "Canceled ❌" if prediction[0] == 1 else "Not Canceled ✅"
    st.success(f"Hasil Prediksi: **{result}**")
