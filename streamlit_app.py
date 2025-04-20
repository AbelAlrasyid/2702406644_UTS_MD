import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime

# Load model dan scaler
@st.cache_data
def load_all():
    with open("best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)
    with open("robust_scaler.pkl", "rb") as f:
        robust_scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        all_columns = pickle.load(f)
    return model, standard_scaler, robust_scaler, all_columns

model, standard_scaler, robust_scaler, all_columns = load_all()

# Input Manual
st.title("Prediksi Pembatalan Pemesanan Hotel")

st.sidebar.header("Input Fitur")
no_of_adults = st.sidebar.slider("Jumlah Dewasa", 0, 10, 2)
no_of_children = st.sidebar.slider("Jumlah Anak-anak", 0, 10, 0)
no_of_weekend_nights = st.sidebar.slider("Malam Weekend", 0, 10, 1)
no_of_week_nights = st.sidebar.slider("Malam Weekday", 0, 10, 1)
type_of_meal_plan = st.sidebar.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
required_car_parking_space = st.sidebar.selectbox("Butuh Parkir?", [0, 1])
room_type_reserved = st.sidebar.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.sidebar.slider("Lead Time", 0, 500, 50)
arrival_year = st.sidebar.selectbox("Arrival Year", list(range(2017, datetime.datetime.now().year + 1)))
arrival_month = st.sidebar.slider("Arrival Month", 1, 12, 6)
arrival_date = st.sidebar.slider("Arrival Date", 1, 31, 15)
market_segment_type = st.sidebar.selectbox("Segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
repeated_guest = st.sidebar.selectbox("Tamu Berulang?", [0, 1])
no_of_previous_cancellations = st.sidebar.slider("Jumlah Cancel Sebelumnya", 0, 10, 0)
no_of_previous_bookings_not_canceled = st.sidebar.slider("Booking Tidak Dibatalkan Sebelumnya", 0, 20, 0)
avg_price_per_room = st.sidebar.number_input("Harga Rata-Rata per Kamar", value=100.0)
no_of_special_requests = st.sidebar.slider("Jumlah Permintaan Khusus", 0, 5, 0)

# Input ke DataFrame
input_dict = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'type_of_meal_plan': type_of_meal_plan,
    'required_car_parking_space': required_car_parking_space,
    'room_type_reserved': room_type_reserved,
    'lead_time': lead_time,
    'arrival_year': arrival_year,
    'arrival_month': arrival_month,
    'arrival_date': arrival_date,
    'market_segment_type': market_segment_type,
    'repeated_guest': repeated_guest,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests
}

input_df = pd.DataFrame([input_dict])

# One-hot encode
cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
input_df = input_df.reindex(columns=all_columns, fill_value=0)

# Scaling
standard_cols = ['arrival_month', 'arrival_date']
robust_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
               'no_of_week_nights', 'required_car_parking_space', 'lead_time',
               'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
               'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
               'no_of_special_requests']

input_df[standard_cols] = standard_scaler.transform(input_df[standard_cols])
input_df[robust_cols] = robust_scaler.transform(input_df[robust_cols])

# Prediksi
prediction = model.predict(input_df)[0]
status = "Dibatalkan" if prediction == 1 else "Tidak Dibatalkan"

st.subheader("Hasil Prediksi")
st.write(f"Status Pemesanan: **{status}**")

st.subheader("Data Input")
st.dataframe(pd.DataFrame([input_dict]))

# =========================
# TEST CASES
# =========================
st.markdown("---")
st.subheader("Contoh Test Case")

test_case_1 = {
    'no_of_adults': 2,
    'no_of_children': 0,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 2,
    'type_of_meal_plan': "Meal Plan 1",
    'required_car_parking_space': 0,
    'room_type_reserved': "Room_Type 1",
    'lead_time': 34,
    'arrival_year': 2025,
    'arrival_month': 6,
    'arrival_date': 10,
    'market_segment_type': "Online",
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 90.0,
    'no_of_special_requests': 1
}

test_case_2 = {
    'no_of_adults': 1,
    'no_of_children': 1,
    'no_of_weekend_nights': 0,
    'no_of_week_nights': 3,
    'type_of_meal_plan': "Not Selected",
    'required_car_parking_space': 1,
    'room_type_reserved': "Room_Type 3",
    'lead_time': 150,
    'arrival_year': 2025,
    'arrival_month': 8,
    'arrival_date': 15,
    'market_segment_type': "Offline",
    'repeated_guest': 1,
    'no_of_previous_cancellations': 2,
    'no_of_previous_bookings_not_canceled': 5,
    'avg_price_per_room': 120.0,
    'no_of_special_requests': 2
}

def predict_test_case(case_dict):
    df = pd.DataFrame([case_dict])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=all_columns, fill_value=0)
    df[standard_cols] = standard_scaler.transform(df[standard_cols])
    df[robust_cols] = robust_scaler.transform(df[robust_cols])
    pred = model.predict(df)[0]
    return "Dibatalkan" if pred == 1 else "Tidak Dibatalkan"

st.markdown("**Test Case 1:**")
st.dataframe(pd.DataFrame([test_case_1]))
st.write(f"Hasil Prediksi: **{predict_test_case(test_case_1)}**")

st.markdown("**Test Case 2:**")
st.dataframe(pd.DataFrame([test_case_2]))
st.write(f"Hasil Prediksi: **{predict_test_case(test_case_2)}**")
